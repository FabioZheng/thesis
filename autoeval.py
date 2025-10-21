"""Evaluate autoencoding compression models on a text dataset.

This script compares an adaptive multi-rate model and three fixed-rate
models by measuring autoencoding quality (ROUGE-L F1 and BERTScore F1),
compression ratio, and latency. Results are saved as per-example JSONL
logs, an aggregated summary JSON, and an RD curve plot.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple
from itertools import chain
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import snapshot_download
from rouge import Rouge
from tqdm import tqdm

import evaluate
from datasets import load_dataset as hf_load_dataset

from modeling_cocom import COCOM
from cmab_agent import CompressionBanditAgent
from utils import prepare_auto_encoding
from metrics import batch_entropy


DEFAULT_DATASET_NAME = "wshuai190/kilt-128"
DEFAULT_DATASET_SPLIT = "train"
ADAPTIVE_MODEL_ID = "ielabgroup/tinyllama-compression-multi-rate-4-16-128"
FIXED_MODEL_IDS = {
    "fixed_4": ("ielabgroup/tinyllama-compression-single-rate-4", 4),
    "fixed_16": ("ielabgroup/tinyllama-compression-single-rate-16", 16),
    "fixed_128": ("ielabgroup/tinyllama-compression-single-rate-128", 128),
}


@dataclass
class EvaluationConfig:
    dataset_name: str
    dataset_split: str
    max_samples: Optional[int]
    save_dir: Path
    device: torch.device
    plot_bertscore: bool
    bandit_path: Optional[str]


@dataclass
class ExampleResult:
    doc_id: Optional[str]
    text: str
    reconstruction: str
    compression_rate: int
    tokens_in: int
    tokens_out: int
    compression_ratio: float
    encode_time: float
    decode_time: float
    rouge_l_f1: float
    bertscore_f1: float

    @property
    def total_time(self) -> float:
        return self.encode_time + self.decode_time


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Autoencoding compression evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name (default: wshuai190/kilt-128)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to evaluate (default: train)",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--save_dir", type=Path, default=Path("results"), help="Directory to save outputs")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument(
        "--plot_bertscore",
        action="store_true",
        help="If set, also plot BERTScore on the RD curve",
    )
    parser.add_argument(
        "--bandit_path",
        type=str,
        default="bandit_ckpt/bandit_agent.pkl",
        help="Path to the trained bandit agent",
    )

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    return EvaluationConfig(
        dataset_name=args.dataset,
        dataset_split=args.split,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        device=device,
        plot_bertscore=args.plot_bertscore,
        bandit_path=args.bandit_path,
    )


def load_dataset_records(
    dataset_name: str, split: str, max_samples: Optional[int]
) -> Callable[[], Iterator[Dict[str, str]]]:
    def record_iterator() -> Iterator[Dict[str, str]]:
        dataset = hf_load_dataset(dataset_name, split=split, streaming=True)
        count = 0
        for example in dataset:
            if max_samples is not None and count >= max_samples:
                break

            text = example.get("text")
            if text is None:
                continue

            record: Dict[str, str] = {"text": text}
            for candidate in ("id", "doc_id", "uid"):
                if candidate in example and example[candidate] is not None:
                    record["id"] = str(example[candidate])
                    break

            count += 1
            yield record

    return record_iterator


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def determine_default_max_length(tokenizer: Any, fallback: int = 2048) -> int:
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len is None or max_len <= 0 or max_len > 32000:
        return fallback
    return int(max_len)


def count_tokens(tokenizer: Any, text: str) -> int:
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    token_ids = tokens.get("input_ids", [])
    return len(token_ids)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clean_decoded_text(text: str, tokenizer: Any) -> str:
    special_tokens = []
    for attr in ("ae_token", "mem_token", "enc_token", "sep_token"):
        token = getattr(tokenizer, attr, None)
        if token:
            special_tokens.append(token)
    cleaned = text
    for token in special_tokens:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()


def download_model_snapshot(model_id: str) -> Path:
    cache_dir = snapshot_download(repo_id=model_id)
    return Path(cache_dir)


def load_cocom_model(model_id: str, device: torch.device, bandit_path: Optional[str] = None) -> COCOM:
    model_path = download_model_snapshot(model_id)
    dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
    model = COCOM.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
    model.to(device)
    model.eval()

    if bandit_path:
        print(f"Loading bandit agent from {bandit_path}")
        with open(bandit_path, "rb") as f:
            bandit_data = pickle.load(f)
        agent = CompressionBanditAgent(
            rates=bandit_data["rates"],
            alpha=bandit_data["alpha"],
            use_length_feature=bandit_data.get("use_length_feature", False),
        )
        agent.A = bandit_data["A"]
        agent.b = bandit_data["b"]
        model.set_bandit_agent(agent)
        print("Bandit agent loaded and attached to the model.")

    return model


def prepare_inputs(
    model: COCOM,
    text: str,
    rate: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    compressor_tokenizer = model.compr.tokenizer if getattr(model, "compr", None) is not None else model.decoder_tokenizer
    enc_tokens = compressor_tokenizer(
        text,
        return_attention_mask=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    token_ids = enc_tokens.get("input_ids", [])
    truncated_len = min(len(token_ids), max_length)
    enc_max_len = max(truncated_len, 1)

    batch = {"text": [text]}
    inputs = prepare_auto_encoding(
        batch,
        compressor_tokenizer=compressor_tokenizer,
        decoder_tokenizer=model.decoder_tokenizer,
        compression_rate=rate,
        enc_max_len=enc_max_len,
        train=False,
    )
    return inputs


def autoencode_example(
    model: COCOM,
    text: str,
    rate: int,
    device: torch.device,
    max_new_tokens: int = 512,
    agent: Optional[CompressionBanditAgent] = None,
) -> Tuple[str, int, float, float]:
    compressor_tokenizer = model.compr.tokenizer if getattr(model, "compr", None) is not None else model.decoder_tokenizer
    max_length = determine_default_max_length(compressor_tokenizer)

    if agent:
        # Calculate context for bandit
        enc_tokens = compressor_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc_tokens["input_ids"]
        attention_mask = enc_tokens["attention_mask"]
        entropy = batch_entropy(input_ids, attention_mask)[0]
        length = attention_mask.sum().item()
        rate = agent.select_rate(entropy, length)

    inputs = prepare_inputs(model, text, rate, max_length)

    enc_input_ids = inputs["enc_input_ids"].to(device)
    enc_attention_mask = inputs["enc_attention_mask"].to(device)
    dec_input_ids = inputs["dec_input_ids"].to(device)
    dec_attention_mask = inputs["dec_attention_mask"].to(device)

    if not hasattr(model, "current_rate"):
        raise AttributeError("Model does not expose current_rate")
    model.current_rate = rate

    with torch.no_grad():
        synchronize_if_needed(device)
        encode_start = time.perf_counter()
        compressed_embs = model.compr(
            enc_input_ids,
            enc_attention_mask,
            rate=rate,
        )
        synchronize_if_needed(device)
        encode_time = time.perf_counter() - encode_start

        indices = range(0, enc_input_ids.size(0) + 1, model.generation_top_k)
        inputs_embeds = model.replace_embeddings(compressed_embs, dec_input_ids, indices)

        synchronize_if_needed(device)
        decode_start = time.perf_counter()
        output_ids = model.decoder.generate(
            inputs_embeds=inputs_embeds.to(device),
            attention_mask=dec_attention_mask,
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens,
        )
        synchronize_if_needed(device)
        decode_time = time.perf_counter() - decode_start

    decoded = model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    decoded = clean_decoded_text(decoded, model.decoder_tokenizer)
    compressed_tokens = compressed_embs.shape[1]
    return decoded, compressed_tokens, encode_time, decode_time


def warmup_model(model: COCOM, text: str, rate: int, device: torch.device) -> None:
    try:
        autoencode_example(model, text, rate, device)
    except Exception:
        # If warmup fails (e.g., due to unsupported sequence), ignore and proceed.
        pass


def compute_rouge(rouge_metric: Rouge, prediction: str, reference: str) -> float:
    try:
        score = rouge_metric.get_scores(prediction, reference)
        return float(score[0]["rouge-l"]["f"])
    except ValueError:
        return 0.0


def evaluate_model(
    name: str,
    model: COCOM,
    default_rate: Optional[int],
    dataset: Iterable[Dict[str, str]],
    device: torch.device,
    token_counter: Any,
    save_dir: Path,
    agent: Optional[CompressionBanditAgent] = None,
) -> List[ExampleResult]:
    ensure_dir(save_dir)
    log_path = save_dir / f"{name}.jsonl"
    rouge_metric = Rouge(metrics=["rouge-l"])
    bert_metric = evaluate.load("bertscore")

    dataset_iter = iter(dataset)
    try:
        first_example = next(dataset_iter)
    except StopIteration:
        return []

    warmup_rate = default_rate if default_rate is not None else getattr(model, "current_rate", 4)
    warmup_model(model, first_example["text"], warmup_rate, device)

    results: List[ExampleResult] = []
    references: List[str] = []
    predictions: List[str] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        for example in tqdm(chain([first_example], dataset_iter), desc=f"Evaluating {name}"):
            text = example["text"]
            doc_id = example.get("id")

            if default_rate is not None:
                rate = default_rate
            else:
                # For the adaptive model, the rate is determined per example
                rate = int(getattr(model, "current_rate", 4))

            try:
                decoded, tokens_out, encode_time, decode_time = autoencode_example(
                    model, text, rate, device, agent=agent
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                error_record = {
                    "id": doc_id,
                    "text": text,
                    "error": str(exc),
                }
                log_file.write(json.dumps(error_record) + "\n")
                continue

            tokens_in = count_tokens(token_counter, text)
            compression_ratio = tokens_out / tokens_in if tokens_in > 0 else math.inf

            rouge_l_f1 = compute_rouge(rouge_metric, decoded, text)

            result = ExampleResult(
                doc_id=doc_id,
                text=text,
                reconstruction=decoded,
                compression_rate=rate,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                compression_ratio=compression_ratio,
                encode_time=encode_time,
                decode_time=decode_time,
                rouge_l_f1=rouge_l_f1,
                bertscore_f1=0.0,  # placeholder updated later
            )

            results.append(result)
            references.append(text)
            predictions.append(decoded)

    if results:
        bert_outputs = bert_metric.compute(predictions=predictions, references=references, lang="en")
        bert_f1 = bert_outputs.get("f1", [])
        for item, score in zip(results, bert_f1):
            item.bertscore_f1 = float(score)

        # Rewrite log with complete results including BERTScore
        with log_path.open("w", encoding="utf-8") as log_file:
            for item in results:
                log_file.write(
                    json.dumps(
                        {
                            "id": item.doc_id,
                            "text": item.text,
                            "reconstruction": item.reconstruction,
                            "compression_rate": item.compression_rate,
                            "tokens_in": item.tokens_in,
                            "tokens_out": item.tokens_out,
                            "compression_ratio": item.compression_ratio,
                            "encode_time": item.encode_time,
                            "decode_time": item.decode_time,
                            "total_time": item.total_time,
                            "rouge_l_f1": item.rouge_l_f1,
                            "bertscore_f1": item.bertscore_f1,
                        }
                    )
                    + "\n"
                )

    return results


def summarize_results(results: List[ExampleResult]) -> Dict[str, Dict[str, float]]:
    if not results:
        return {}

    def aggregate(values: Iterable[float]) -> Tuple[float, float]:
        arr = np.array(list(values), dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    summary: Dict[str, Dict[str, float]] = {}
    metrics = {
        "rouge_l_f1": [r.rouge_l_f1 for r in results],
        "bertscore_f1": [r.bertscore_f1 for r in results],
        "compression_ratio": [r.compression_ratio for r in results],
        "tokens_in": [r.tokens_in for r in results],
        "tokens_out": [r.tokens_out for r in results],
    }
    for key, values in metrics.items():
        mean, std = aggregate(values)
        summary[key] = {"mean": mean, "std": std}

    latencies = {
        "encode_time": [r.encode_time for r in results],
        "decode_time": [r.decode_time for r in results],
        "total_time": [r.total_time for r in results],
    }
    for key, values in latencies.items():
        arr = np.array(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary["num_examples"] = {"count": len(results)}
    return summary


def save_summary(
    save_dir: Path,
    tokenizer_name: str,
    dataset_name: str,
    dataset_split: str,
    model_summaries: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    payload = {
        "tokenizer": tokenizer_name,
        "dataset": {
            "name": dataset_name,
            "split": dataset_split,
        },
        "conditions": model_summaries,
    }
    summary_path = save_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_rd_curve(
    save_dir: Path,
    summaries: Dict[str, List[ExampleResult]],
    plot_bertscore: bool,
) -> None:
    plt.figure(figsize=(8, 6))
    for name, results in summaries.items():
        if not results:
            continue
        ratios = np.array([r.compression_ratio for r in results], dtype=np.float64)
        rouge_scores = np.array([r.rouge_l_f1 for r in results], dtype=np.float64)
        order = np.argsort(ratios)
        plt.scatter(ratios, rouge_scores, label=f"{name} (ROUGE-L)")
        plt.plot(ratios[order], rouge_scores[order], linestyle="--")

    if plot_bertscore:
        for name, results in summaries.items():
            if not results:
                continue
            ratios = np.array([r.compression_ratio for r in results], dtype=np.float64)
            bert_scores = np.array([r.bertscore_f1 for r in results], dtype=np.float64)
            order = np.argsort(ratios)
            plt.scatter(ratios, bert_scores, marker="x", label=f"{name} (BERTScore)")
            plt.plot(ratios[order], bert_scores[order], linestyle=":")

    plt.xlabel("Compression ratio (tokens_out / tokens_in)")
    plt.ylabel("Score")
    plt.title("Rate-Distortion Curve")
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(save_dir / "rd_curve.png", dpi=300)
    plt.close()


def main() -> None:
    config = parse_args()
    ensure_dir(config.save_dir)

    dataset_iterator_factory = load_dataset_records(
        config.dataset_name, config.dataset_split, config.max_samples
    )

    initial_iter = dataset_iterator_factory()
    try:
        first_record = next(initial_iter)
    except StopIteration as exc:
        raise ValueError("Dataset is empty or could not be parsed.") from exc
    adaptive_dataset = chain([first_record], initial_iter)

    all_results: Dict[str, List[ExampleResult]] = {}
    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    adaptive_model = load_cocom_model(
        ADAPTIVE_MODEL_ID, config.device, bandit_path=config.bandit_path
    )
    token_counter = adaptive_model.decoder_tokenizer

    adaptive_results = evaluate_model(
        name="adaptive",
        model=adaptive_model,
        default_rate=None,
        dataset=adaptive_dataset,
        device=config.device,
        token_counter=token_counter,
        save_dir=config.save_dir,
        agent=getattr(adaptive_model, "bandit_agent", None),
    )
    all_results["adaptive"] = adaptive_results
    summaries["adaptive"] = summarize_results(adaptive_results)

    del adaptive_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name, (model_id, rate) in FIXED_MODEL_IDS.items():
        model = load_cocom_model(model_id, config.device)
        results = evaluate_model(
            name=name,
            model=model,
            default_rate=rate,
            dataset=dataset_iterator_factory(),
            device=config.device,
            token_counter=token_counter,
            save_dir=config.save_dir,
        )
        all_results[name] = results
        summaries[name] = summarize_results(results)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_summary(
        config.save_dir,
        getattr(token_counter, "name_or_path", "unknown"),
        config.dataset_name,
        config.dataset_split,
        summaries,
    )
    plot_rd_curve(config.save_dir, all_results, config.plot_bertscore)


if __name__ == "__main__":
    main()
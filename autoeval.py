import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from bert_score import BERTScorer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

MODEL_SPECS = [
    ("adaptive", "ielabgroup/tinyllama-compression-multi-rate-4-16-128", None, "adaptive"),
    ("fixed@4", "ielabgroup/tinyllama-compression-single-rate-4", 4, "fixed"),
    ("fixed@16", "ielabgroup/tinyllama-compression-single-rate-16", 16, "fixed"),
    ("fixed@128", "ielabgroup/tinyllama-compression-single-rate-128", 128, "fixed"),
]

def load_any_model(model_name: str) -> Tuple[PreTrainedModel, str]:
    try:
        return AutoModelForSeq2SeqLM.from_pretrained(model_name), "seq2seq"
    except Exception as err:  # pragma: no cover - only hit when seq2seq loader fails
        logging.debug("Falling back to causal LM loader for %s (%s)", model_name, err)
    return AutoModelForCausalLM.from_pretrained(model_name), "causal"


class CompressionModelWrapper:
    def __init__(self, model_name: str, device: torch.device, metrics_tokenizer: PreTrainedTokenizerBase) -> None:
        self.model, self.kind = load_any_model(model_name)
        self.model_name = model_name
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.metrics_tokenizer = metrics_tokenizer
        self.warned = False
        self.max_new_tokens = 512

    def _token_count(self, text: str) -> int:
        return len(self.metrics_tokenizer(text, add_special_tokens=True)["input_ids"])

    def run(self, text: str, rate: Optional[int]) -> Dict[str, Any]:
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tokens_in = self._token_count(text)
        compressed = None
        tokens_out = tokens_in
        selected_rate: Optional[Any] = None
        sync = self.device.type == "cuda"

        if sync: torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        if hasattr(self.model, "compress"):
            try:  # pragma: no cover - depends on external model
                kwargs = dict(inputs)
                if rate is not None:
                    kwargs["rate"] = rate
                compressed = self.model.compress(**kwargs)
                if isinstance(compressed, dict):
                    if "compressed_length" in compressed:
                        tokens_out = int(compressed["compressed_length"])
                    for key in ("compressed_tokens", "codes", "compressed_ids"):
                        tensor = compressed.get(key)
                        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
                            tokens_out = int(tensor.shape[-1])
                            break
                    selected_rate = compressed.get("selected_rate") or compressed.get("rate")
                elif isinstance(compressed, torch.Tensor):
                    tokens_out = int(compressed.shape[-1])
            except Exception as exc:
                logging.warning("compress() failed for %s: %s", self.model_name, exc)
                compressed = None
        if compressed is None and not self.warned:
            logging.info("Model %s falling back to raw inputs for compression stats.", self.model_name)
            self.warned = True
        if sync: torch.cuda.synchronize(self.device)
        encode_time = time.perf_counter() - start

        if sync: torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        text_out = None
        if compressed is not None and hasattr(self.model, "decompress"):
            try:  # pragma: no cover - depends on external model
                dec_kwargs: Dict[str, Any] = {"compressed": compressed}
                if rate is not None:
                    dec_kwargs["rate"] = rate
                decoded = self.model.decompress(**dec_kwargs)
                if isinstance(decoded, dict) and "text" in decoded:
                    text_out = str(decoded["text"])
                elif isinstance(decoded, torch.Tensor):
                    text_out = self.tokenizer.batch_decode(decoded, skip_special_tokens=True)[0]
                elif isinstance(decoded, str):
                    text_out = decoded
            except Exception as exc:
                logging.warning("decompress() failed for %s: %s", self.model_name, exc)
        if text_out is None:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            if self.kind == "causal":
                prompt = inputs["input_ids"].shape[-1]
                outputs = outputs[:, prompt:]
            text_out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if not text_out:
                text_out = text
        if sync: torch.cuda.synchronize(self.device)
        decode_time = time.perf_counter() - start

        compression_ratio = tokens_out / tokens_in if tokens_in else 0.0
        if selected_rate is not None:
            return {
                "reconstruction": text_out,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "compression_ratio": compression_ratio,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "total_time": encode_time + decode_time,
                "selected_rate": selected_rate,
            }
        return {
            "reconstruction": text_out,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "compression_ratio": compression_ratio,
            "encode_time": encode_time,
            "decode_time": decode_time,
            "total_time": encode_time + decode_time,
        }

def compute_statistics(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return (
        {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
        if arr.size
        else {"mean": math.nan, "std": math.nan}
    )

def compute_latency(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    return (
        {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }
        if arr.size
        else {k: math.nan for k in ("mean", "std", "p50", "p95")}
    )


def evaluate_model(
    spec: Tuple[str, str, Optional[int], str],
    dataset: Iterable[Dict[str, Any]],
    wrapper: CompressionModelWrapper,
    rouge: rouge_scorer.RougeScorer,
    bert: BERTScorer,
    results_path: Path,
) -> Dict[str, Any]:
    label, model_name, rate, _ = spec
    logging.info("Evaluating %s (%s)", label, model_name)
    records: List[Dict[str, Any]] = []
    metric_keys = ["tokens_in", "tokens_out", "compression_ratio", "encode_time", "decode_time", "total_time"]
    with results_path.open("w", encoding="utf-8") as handle:
        for idx, sample in enumerate(tqdm(dataset, desc=label)):
            text = sample.get("text", "")
            sample_id = sample.get("id", idx)
            processed = wrapper.run(text, rate)
            reconstruction = processed["reconstruction"]
            rouge_result = rouge.score(target=text, prediction=reconstruction)["rougeL"]
            _, _, bert_f = bert.score([reconstruction], [text], verbose=False)
            record = {
                "id": sample_id,
                "text": text,
                "reconstruction": reconstruction,
                "rouge_l_f1": rouge_result.fmeasure,
                "bertscore_f1": float(bert_f.cpu().numpy()[0]),
                **{k: processed[k] for k in metric_keys},
                "rate": processed.get("selected_rate", rate),
                "model_name": model_name,
                "label": label,
            }
            records.append(record)
            handle.write(json.dumps(record) + "\n")

    return aggregate_results(spec, records)


def aggregate_results(spec: Tuple[str, str, Optional[int], str], records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stat_keys = ["rouge_l_f1", "bertscore_f1", "compression_ratio", "tokens_in", "tokens_out"]
    stats = {k: compute_statistics(r[k] for r in records) for k in stat_keys}
    lat_keys = ["encode_time", "decode_time", "total_time"]
    stats.update({k: compute_latency(r[k] for r in records) for k in lat_keys})
    label, model_name, rate, category = spec
    base = {
        "label": label,
        "model_name": model_name,
        "category": category,
        "rate": rate,
        "num_samples": len(records),
    }
    base.update(stats)
    return base

def plot_rd_curve(
    aggregates: Dict[str, Dict[str, Any]],
    output_path: Path,
    plot_bertscore: bool,
) -> None:
    plt.figure(figsize=(8, 6))
    for label, stats in aggregates.items():
        x = stats["compression_ratio"]["mean"]
        y = stats["rouge_l_f1"]["mean"]
        plt.scatter(x, y, marker="*" if stats.get("category") == "adaptive" else "o", label=f"{label} (ROUGE-L)")
    if plot_bertscore:
        for label, stats in aggregates.items():
            x = stats["compression_ratio"]["mean"]
            y = stats["bertscore_f1"]["mean"]
            plt.scatter(x, y, marker="X" if stats.get("category") == "adaptive" else "^", label=f"{label} (BERTScore)")
    plt.xlabel("Compression ratio (tokens_out / tokens_in)")
    plt.ylabel("Quality score")
    plt.title("Rate–Distortion Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compression models.")
    parser.add_argument("--dataset", type=str, default="data/docs.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--plot_bertscore",
        action="store_true",
        help="If set, also plot BERTScore points on the RD curve.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    dataset = load_dataset("json", data_files=args.dataset)["train"]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics_tokenizer = AutoTokenizer.from_pretrained(MODEL_SPECS[0][1])
    tokenizer_name = metrics_tokenizer.name_or_path

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bert_device = (
        f"cuda:{device.index}" if device.type == "cuda" and device.index is not None else device.type
    )
    bert = BERTScorer(lang="en", rescale_with_baseline=True, device=bert_device)

    summary = {
        "tokenizer": tokenizer_name,
        "device": str(device),
        "plot_bertscore": args.plot_bertscore,
        "models": {},
    }

    for spec in MODEL_SPECS:
        label, model_name, _, _ = spec
        wrapper = CompressionModelWrapper(model_name, device, metrics_tokenizer)
        results_path = save_dir / f"{label.replace('@', '_at_')}_results.jsonl"
        summary["models"][label] = evaluate_model(spec, dataset, wrapper, rouge, bert, results_path)

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved summary to %s", summary_path)

    plot_path = save_dir / "rd_curve.png"
    plot_rd_curve(summary["models"], plot_path, args.plot_bertscore)
    logging.info("Saved RD curve to %s", plot_path)

if __name__ == "__main__":
    main()


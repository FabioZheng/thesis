#!/usr/bin/env python
"""Evaluate autoencoding compression models across compression settings."""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from rouge import Rouge
from tqdm import tqdm

from modeling_cocom import COCOM
from metrics import batch_entropy
from utils import prepare_auto_encoding

try:
    import evaluate
except ImportError as exc:  # pragma: no cover - user environment is expected to have evaluate
    raise ImportError(
        "The `evaluate` package is required to compute BERTScore. "
        "Install it with `pip install evaluate bert-score`."
    ) from exc


TextSample = Tuple[str, str]


class SimpleAdaptiveAgent:
    """Heuristic rate selector compatible with :class:`CompressionBanditAgent`."""

    def __init__(self, rates: Sequence[int]):
        self.rates = sorted(set(int(r) for r in rates))
        if not self.rates:
            raise ValueError("At least one compression rate is required for adaptive evaluation.")
        self.use_length_feature = True
        # Rough calibration targets inspired by typical encoder statistics.
        self._max_entropy = 8.0
        self._max_length = 1024.0

    def select_rate(self, avg_entropy: float, avg_length: Optional[float] = None) -> int:
        # Normalise the signals into [0, 1]
        entropy_score = min(max(avg_entropy / self._max_entropy, 0.0), 1.0)
        if avg_length is None:
            length_score = entropy_score
        else:
            length_score = min(max(avg_length / self._max_length, 0.0), 1.0)

        # High score -> keep more information (use smaller compression factor)
        combined = 0.6 * entropy_score + 0.4 * length_score
        # Map to an index with higher compression for lower scores
        idx = len(self.rates) - 1 - int(round(combined * (len(self.rates) - 1)))
        idx = max(0, min(idx, len(self.rates) - 1))
        return self.rates[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compression quality across rates.")
    parser.add_argument("--model", required=True, help="Hugging Face model id or local checkpoint directory.")
    parser.add_argument("--dataset", required=True, help="Path to JSON/JSONL dataset with text samples.")
    parser.add_argument("--rates", type=int, nargs="+", default=[4, 8, 16],
                        help="Fixed compression rates to evaluate (e.g., 4 8 16).")
    parser.add_argument("--adaptive", action="store_true",
                        help="Include an adaptive compression policy in the evaluation.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate.")
    parser.add_argument("--save_dir", required=True, help="Directory to save evaluation outputs.")
    parser.add_argument("--encoder_max_length", type=int, default=512,
                        help="Maximum encoder sequence length used for tokenisation.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate during decoding.")
    return parser.parse_args()


def _extract_text(record: object) -> Optional[str]:
    if isinstance(record, str):
        text = record.strip()
        return text if text else None
    if isinstance(record, dict):
        for key in ("text", "document", "content", "body", "passage", "answer"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in record.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_text_samples(path: str, limit: Optional[int] = None) -> List[TextSample]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples: List[TextSample] = []
    if dataset_path.suffix.lower() == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = _extract_text(record)
                if text:
                    identifier = str(record.get("id") or record.get("doc_id") or record.get("guid") or idx)
                    samples.append((identifier, text))
                if limit is not None and len(samples) >= limit:
                    break
    else:
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for idx, (key, value) in enumerate(data.items()):
                text = _extract_text(value)
                if text:
                    samples.append((str(key), text))
                if limit is not None and len(samples) >= limit:
                    break
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                text = _extract_text(item)
                if text:
                    samples.append((str(idx), text))
                if limit is not None and len(samples) >= limit:
                    break
        else:
            raise ValueError("Unsupported JSON structure. Expected a dict or list of examples.")

    if limit is not None:
        samples = samples[:limit]

    if not samples:
        raise ValueError("No text samples were loaded from the dataset.")

    return samples


def prepare_inputs(texts: Sequence[str], compressor_tokenizer, decoder_tokenizer, compression_rate: int,
                   enc_max_len: int) -> Dict[str, torch.Tensor]:
    example = {"text": list(texts)}
    features = prepare_auto_encoding(example, compressor_tokenizer, decoder_tokenizer,
                                     compression_rate=compression_rate,
                                     enc_max_len=enc_max_len, train=False)
    return features


def _count_input_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        mem_token_id: Optional[int]) -> int:
    mask = attention_mask.bool()
    if mem_token_id is not None:
        mem_mask = input_ids.eq(mem_token_id) & mask
        return int((mask & ~mem_mask).sum().item())
    return int(mask.sum().item())


def evaluate_setting(name: str, model: COCOM, samples: List[TextSample],
                     compressor_tokenizer, decoder_tokenizer, compression_rate: int,
                     enc_max_len: int, max_new_tokens: int, device: torch.device,
                     agent: Optional[SimpleAdaptiveAgent] = None) -> Tuple[List[Dict], Dict[str, float]]:
    bertscore_metric = evaluate.load("bertscore")
    rouge = Rouge()

    per_example: List[Dict] = []
    predictions: List[str] = []
    references: List[str] = []
    compression_ratios: List[float] = []
    encode_times: List[float] = []
    decode_times: List[float] = []
    tokens_in_values: List[int] = []
    tokens_out_values: List[int] = []
    rates_used: List[int] = []

    min_rate = compression_rate

    model.eval()
    mem_token_id = getattr(model.decoder_tokenizer, "mem_token_id", None)

    with torch.no_grad():
        for sample_id, text in tqdm(samples, desc=f"Evaluating {name}"):
            texts = [text]
            features = prepare_inputs(texts, compressor_tokenizer, decoder_tokenizer,
                                      compression_rate=min_rate,
                                      enc_max_len=enc_max_len)

            enc_input_ids = features["enc_input_ids"].to(device)
            enc_attention_mask = features["enc_attention_mask"].to(device)
            dec_input_ids = features["dec_input_ids"].to(device)
            dec_attention_mask = features["dec_attention_mask"].to(device)

            if agent is not None:
                entropies = batch_entropy(features["enc_input_ids"], features["enc_attention_mask"])
                avg_entropy = sum(entropies) / len(entropies)
                lengths = features["enc_attention_mask"].sum(dim=1).tolist()
                avg_length = sum(lengths) / len(lengths) if lengths else None
                selected_rate = agent.select_rate(avg_entropy, avg_length)
                model.current_rate = selected_rate
            else:
                model.current_rate = compression_rate
                selected_rate = compression_rate

            encode_start = time.perf_counter()
            if model.compr is not None:
                compressed_embs = model.compr(enc_input_ids, enc_attention_mask, rate=model.current_rate)
            else:
                compressed_embs = model.compr_decoder(enc_input_ids, enc_attention_mask)
            indices = range(0, enc_input_ids.size(0) + 1, model.generation_top_k)
            inputs_embeds = model.replace_embeddings(compressed_embs, dec_input_ids, indices)
            encode_time = time.perf_counter() - encode_start

            decode_start = time.perf_counter()
            output_ids = model.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=dec_attention_mask,
                do_sample=False,
                top_p=None,
                max_new_tokens=max_new_tokens,
            )
            decode_time = time.perf_counter() - decode_start

            decoded = model.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

            tokens_in = _count_input_tokens(features["enc_input_ids"], features["enc_attention_mask"], mem_token_id)
            tokens_out = compressed_embs.size(1)
            compression_ratio = (tokens_out / tokens_in) if tokens_in else float("inf")

            predictions.append(decoded)
            references.append(text)
            compression_ratios.append(compression_ratio)
            encode_times.append(encode_time)
            decode_times.append(decode_time)
            tokens_in_values.append(tokens_in)
            tokens_out_values.append(tokens_out)
            rates_used.append(selected_rate)

            per_example.append({
                "sample_id": sample_id,
                "reference": text,
                "prediction": decoded,
                "rate": selected_rate,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "compression_ratio": compression_ratio,
                "encode_time": encode_time,
                "decode_time": decode_time,
                "latency": encode_time + decode_time,
            })

    # Compute quality metrics
    rouge_l_scores: List[float] = []
    for pred, ref in zip(predictions, references):
        try:
            rouge_l = rouge.get_scores(pred, ref, avg=True)["rouge-l"]["f"]
        except ValueError:
            rouge_l = 0.0
        rouge_l_scores.append(rouge_l)

    bert_scores = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
    bert_f1 = bert_scores["f1"]

    for item, rouge_l, bert_f in zip(per_example, rouge_l_scores, bert_f1):
        item["rouge_l"] = rouge_l
        item["bertscore_f1"] = bert_f

    summary = {
        "configuration": name,
        "num_samples": len(per_example),
        "avg_rate": sum(rates_used) / len(rates_used) if rates_used else None,
        "avg_tokens_in": sum(tokens_in_values) / len(tokens_in_values) if tokens_in_values else None,
        "avg_tokens_out": sum(tokens_out_values) / len(tokens_out_values) if tokens_out_values else None,
        "avg_compression_ratio": sum(compression_ratios) / len(compression_ratios) if compression_ratios else None,
        "avg_rouge_l": sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else None,
        "avg_bertscore_f1": sum(bert_f1) / len(bert_f1) if bert_f1 else None,
        "avg_encode_time": sum(encode_times) / len(encode_times) if encode_times else None,
        "avg_decode_time": sum(decode_times) / len(decode_times) if decode_times else None,
        "avg_latency": sum(e + d for e, d in zip(encode_times, decode_times)) / len(encode_times) if encode_times else None,
    }

    return per_example, summary


def save_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json_file(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_rd_curve(output_path: Path, summaries: Dict[str, Dict[str, float]], adaptive_key: Optional[str]) -> None:
    fixed_points = [s for key, s in summaries.items() if key != adaptive_key]
    fixed_points = [s for s in fixed_points if s.get("avg_compression_ratio") is not None]
    fixed_points.sort(key=lambda s: s["avg_compression_ratio"])

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    if fixed_points:
        ratios = [s["avg_compression_ratio"] for s in fixed_points]
        rouge_scores = [s["avg_rouge_l"] for s in fixed_points]
        bert_scores = [s["avg_bertscore_f1"] for s in fixed_points]
        ax.plot(ratios, rouge_scores, marker="o", label="Fixed (ROUGE-L)")
        ax.plot(ratios, bert_scores, marker="^", label="Fixed (BERTScore)")

    if adaptive_key and adaptive_key in summaries:
        adaptive_summary = summaries[adaptive_key]
        ratio = adaptive_summary.get("avg_compression_ratio")
        rouge_l = adaptive_summary.get("avg_rouge_l")
        bert_f1 = adaptive_summary.get("avg_bertscore_f1")
        if ratio is not None:
            ax.scatter([ratio], [rouge_l], marker="s", s=120, label="Adaptive (ROUGE-L)")
            ax.scatter([ratio], [bert_f1], marker="x", s=120, label="Adaptive (BERTScore)")

    ax.set_xlabel("Compression ratio (tokens_out / tokens_in)")
    ax.set_ylabel("Quality score")
    ax.set_title("Rate–Distortion curve")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    samples = load_text_samples(args.dataset, args.max_samples)

    print(f"Loaded {len(samples)} samples for evaluation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model {args.model} on device {device}...")
    model = COCOM.from_pretrained(args.model)
    model.to(device)
    model.generation_top_k = 1

    compressor_tokenizer = model.compr.tokenizer if model.compr is not None else model.decoder_tokenizer
    decoder_tokenizer = model.decoder_tokenizer

    summaries: Dict[str, Dict[str, float]] = {}

    adaptive_key: Optional[str] = None

    if args.adaptive:
        agent = SimpleAdaptiveAgent(args.rates)
        adaptive_key = "adaptive"
        per_example, summary = evaluate_setting(
            name="adaptive",
            model=model,
            samples=samples,
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compression_rate=min(args.rates),
            enc_max_len=args.encoder_max_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
            agent=agent,
        )
        summaries[adaptive_key] = summary
        save_jsonl(Path(args.save_dir) / f"per_example_{adaptive_key}.jsonl", per_example)

    for rate in args.rates:
        key = f"fixed_{rate}x"
        per_example, summary = evaluate_setting(
            name=key,
            model=model,
            samples=samples,
            compressor_tokenizer=compressor_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            compression_rate=rate,
            enc_max_len=args.encoder_max_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        summaries[key] = summary
        save_jsonl(Path(args.save_dir) / f"per_example_{key}.jsonl", per_example)

    summary_path = Path(args.save_dir) / "summary.json"
    save_json_file(summary_path, summaries)
    print(f"Saved summary to {summary_path}")

    plot_path = Path(args.save_dir) / "rd_curve.png"
    plot_rd_curve(plot_path, summaries, adaptive_key)
    print(f"Saved RD curve plot to {plot_path}")


if __name__ == "__main__":
    main()

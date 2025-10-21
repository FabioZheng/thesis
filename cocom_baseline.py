#!/usr/bin/env python3
"""Baseline RAG evaluation script for COCOM models on MS MARCO.

This script evaluates a COCOM model using the MS MARCO dataset in
streaming mode. It generates answers using (a) the query only and (b)
query plus retrieved passages, and reports BERTScore and token-level F1
for each setting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import islice
from typing import Iterator, List, Optional, Sequence

import torch
from datasets import IterableDataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from bert_score import score as bert_scorer
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'bert-score'. Install with `pip install bert-score`."
    ) from exc


DATASET_NAME = "ms_marco"
DATASET_CONFIG = "v1.1"
DEFAULT_SPLIT = "validation"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_NUM_EXAMPLES = 32
DOCUMENT_SEPARATOR = "\n\n"
MAX_CONTEXT_DOCS = 5


@dataclass
class Example:
    query_id: Optional[str]
    query: str
    answers: List[str]
    documents: List[str]


def take(dataset: IterableDataset, count: int) -> Iterator[dict]:
    """Yield at most ``count`` examples from an iterable dataset."""

    yield from islice(dataset, count)


def extract_ms_marco_fields(example: dict) -> Example:
    """Extract the relevant MS MARCO fields from an example.

    Mirrors the behaviour in :mod:`baseline.py` and :mod:`save_json.py`.
    """

    query = str(example.get("query", "")).strip()
    answers_raw = example.get("answers") or []
    answers = [str(ans).strip() for ans in answers_raw if str(ans).strip()]

    passages = example.get("passages") or {}
    documents = passages.get("passage_text") or []
    documents = [str(doc).strip() for doc in documents if str(doc).strip()]

    query_id = example.get("query_id") or example.get("id")
    if isinstance(query_id, (list, tuple)):
        query_id = next((str(item) for item in query_id if item is not None), None)
    elif query_id is not None:
        query_id = str(query_id)

    return Example(query_id=query_id, query=query, answers=answers, documents=documents)


def format_messages(query: str, documents: Optional[Sequence[str]], tokenizer) -> List[dict]:
    """Create chat messages suitable for ``apply_chat_template`` if available."""

    if documents:
        context = DOCUMENT_SEPARATOR.join(documents)
        user_content = (
            "You are a helpful assistant. Answer the question using the provided context.\n"
            f"Question: {query}\n\nContext:\n{context}"
        )
    else:
        user_content = (
            "You are a helpful assistant. Answer the question concisely.\n"
            f"Question: {query}"
        )

    return [{"role": "user", "content": user_content}]


def build_prompt(tokenizer, messages: List[dict]) -> str:
    """Build a prompt for the model, using the chat template when available."""

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # pragma: no cover - fallback path
            pass

    # Fallback: concatenate the messages manually.
    return "\n\n".join(message["content"] for message in messages)


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate an answer for a given prompt."""

    if not prompt.strip():
        return ""

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
    }

    with torch.no_grad():
        output = model.generate(**encoded, **model_kwargs)

    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    prompt_decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    answer = decoded[len(prompt_decoded) :].strip()
    return answer


def bert_f1(candidate: str, reference: str, device: str) -> float:
    if not candidate or not reference:
        return 0.0
    _, _, f1 = bert_scorer([candidate], [reference], device=device, verbose=False)
    return float(f1.item())


def token_f1(candidate: str, reference: str) -> float:
    """Compute token-level F1 using whitespace tokenisation."""

    candidate_tokens = candidate.split()
    reference_tokens = reference.split()
    if not candidate_tokens or not reference_tokens:
        return 0.0

    candidate_counts = {}
    for token in candidate_tokens:
        candidate_counts[token] = candidate_counts.get(token, 0) + 1

    reference_counts = {}
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = 0
    for token, count in candidate_counts.items():
        overlap += min(count, reference_counts.get(token, 0))

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_example(
    example: Example,
    model: AutoModelForCausalLM,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    bert_device: str,
) -> dict:
    """Evaluate the model on a single example."""

    gold_answer = example.answers[0] if example.answers else ""

    query_only_messages = format_messages(example.query, None, tokenizer)
    query_only_prompt = build_prompt(tokenizer, query_only_messages)
    query_only_answer = generate_answer(model, tokenizer, query_only_prompt, device, max_new_tokens)

    context_documents = example.documents[:MAX_CONTEXT_DOCS]
    rag_messages = format_messages(example.query, context_documents, tokenizer)
    rag_prompt = build_prompt(tokenizer, rag_messages)
    rag_answer = generate_answer(model, tokenizer, rag_prompt, device, max_new_tokens)

    result = {
        "query_id": example.query_id,
        "query": example.query,
        "gold_answer": gold_answer,
        "query_only_answer": query_only_answer,
        "rag_answer": rag_answer,
        "query_only_metrics": {
            "bertscore_f1": bert_f1(query_only_answer, gold_answer, bert_device),
            "token_f1": token_f1(query_only_answer, gold_answer),
        },
        "rag_metrics": {
            "bertscore_f1": bert_f1(rag_answer, gold_answer, bert_device),
            "token_f1": token_f1(rag_answer, gold_answer),
        },
    }
    return result


def aggregate_metrics(results: Sequence[dict]) -> dict:
    """Aggregate metrics across all results by taking the arithmetic mean."""

    def mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    query_only_bert = [item["query_only_metrics"]["bertscore_f1"] for item in results]
    query_only_f1 = [item["query_only_metrics"]["token_f1"] for item in results]
    rag_bert = [item["rag_metrics"]["bertscore_f1"] for item in results]
    rag_f1 = [item["rag_metrics"]["token_f1"] for item in results]

    return {
        "query_only": {
            "bertscore_f1": mean(query_only_bert),
            "token_f1": mean(query_only_f1),
        },
        "rag": {
            "bertscore_f1": mean(rag_bert),
            "token_f1": mean(rag_f1),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a COCOM model on MS MARCO.")
    parser.add_argument("model_path", help="Hugging Face model path for the COCOM model.")
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to evaluate (default: validation).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help="Number of examples to evaluate (default: 32).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum number of new tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save per-example results as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    model.eval()

    device = next(model.parameters()).device
    bert_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG}:{args.split} in streaming mode "
        f"(first {args.num_examples} examples)."
    )
    dataset_iterable = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=args.split,
        streaming=True,
    )

    results = []
    for raw_example in take(dataset_iterable, args.num_examples):
        example = extract_ms_marco_fields(raw_example)
        if not example.query:
            continue
        result = evaluate_example(
            example,
            model,
            tokenizer,
            device,
            args.max_new_tokens,
            bert_device,
        )
        results.append(result)
        print(
            f"Processed query_id={example.query_id} | "
            f"Query-only F1={result['query_only_metrics']['token_f1']:.3f} | "
            f"RAG F1={result['rag_metrics']['token_f1']:.3f}"
        )

    summary = aggregate_metrics(results)
    print("\n=== Aggregate Metrics ===")
    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"results": results, "summary": summary}, handle, indent=2)
        print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()

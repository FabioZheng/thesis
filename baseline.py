"""Baseline script for MS MARCO RAG QA using the COCOM decoder."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import torch
from datasets import IterableDataset, load_dataset

from modeling_cocom import COCOM, COCOMConfig
from save_json import (
    _extract_answer_texts,
    _extract_passage_texts,
    _extract_query_id,
    _extract_query_text,
    _filter_placeholder_answers,
    _is_placeholder_only_answers,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class Sample:
    """Container for the MS MARCO data required by the baseline."""

    query_id: Optional[str]
    query: str
    answers: List[str]
    documents: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        default="ms_marco",
        help="Hugging Face dataset identifier (default: ms_marco)",
    )
    parser.add_argument(
        "--dataset-config",
        default="v2.1",
        help="Dataset configuration on the Hugging Face Hub (default: v2.1)",
    )
    parser.add_argument(
        "--dataset-split",
        default="validation",
        help="Dataset split to stream from the Hugging Face Hub (default: validation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of initial dataset rows to skip before evaluation",
    )
    parser.add_argument(
        "--decoder-model-name",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Decoder model identifier to load for the COCOM architecture",
    )
    parser.add_argument(
        "--quantization",
        choices=("no", "int4", "int8"),
        default="no",
        help="Quantisation strategy passed to :class:`COCOM` (default: no)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto-detect)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate per answer (default: 64)",
    )
    parser.add_argument(
        "--documents-per-sample",
        type=int,
        default=1,
        help="Number of documents to include in the contextual prompt (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the baseline script (default: INFO)",
    )
    return parser.parse_args()


def stream_samples(
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    limit: Optional[int],
    offset: int,
) -> Iterator[Sample]:
    """Stream MS MARCO rows and yield :class:`Sample` instances."""

    load_kwargs = {"streaming": True}
    if dataset_config is not None:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split, **load_kwargs)

    if not isinstance(dataset, IterableDataset):
        raise RuntimeError("Streaming mode should return an IterableDataset instance")

    iterator: Iterable[dict] = dataset.skip(offset) if offset else dataset

    count = 0
    for row in iterator:
        query = _extract_query_text(row)
        if not query:
            LOGGER.debug("Skipping row without a query: %s", row)
            continue

        answers = _extract_answer_texts(row)
        if _is_placeholder_only_answers(answers):
            continue
        answers = _filter_placeholder_answers(answers)

        if not answers:
            LOGGER.debug("Skipping row without usable answers: %s", row)
            continue

        passages = _extract_passage_texts(row)
        if not passages:
            LOGGER.debug("Skipping row without passages: %s", row)
            continue

        query_id = _extract_query_id(row)
        sample = Sample(
            query_id=str(query_id) if query_id is not None else None,
            query=query,
            answers=list(answers),
            documents=list(passages),
        )

        yield sample
        count += 1
        if limit is not None and count >= limit:
            break


def load_decoder(args: argparse.Namespace) -> COCOM:
    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model_name,
        quantization=args.quantization,
        generation_top_k=args.documents_per_sample,
        sep=True,
        compr_model_name=None,
        compr_rates=[1],
        training_form="decoder",
    )
    model = COCOM(cfg)
    model.to(args.device)
    model.eval()
    return model


def build_prompt_only_query(model: COCOM, query: str) -> str:
    bos = model.decoder_tokenizer.bos_token or ""
    return f"{bos}[INST]\n{query}\n[/INST]\n"


def build_prompt_with_documents(model: COCOM, query: str, documents: List[str]) -> str:
    bos = model.decoder_tokenizer.bos_token or ""
    context = "\n\n".join(documents)
    context_instruction = (
        "Use the following document to help answer:\n"
        f"{context}\n"
    )
    return f"{bos}[INST]\n{query}\n\n{context_instruction}[/INST]\n"


def generate_text(model: COCOM, prompt: str, max_new_tokens: int, device: str) -> str:
    encoded = model.decoder_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    output = model.decoder.generate(
        **encoded,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    text = model.decoder_tokenizer.decode(output[0], skip_special_tokens=True)
    return text.strip()


def normalise_text(text: str) -> str:
    return " ".join(text.lower().split())


def has_answer(prediction: str, answers: Iterable[str]) -> bool:
    prediction_norm = normalise_text(prediction)
    for answer in answers:
        answer_norm = normalise_text(answer)
        if answer_norm and (
            answer_norm in prediction_norm or prediction_norm in answer_norm
        ):
            return True
    return False


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    LOGGER.info(
        "Streaming dataset %s/%s split=%s", args.dataset_name, args.dataset_config, args.dataset_split
    )

    model = load_decoder(args)

    stats_only_query: List[bool] = []
    stats_with_docs: List[bool] = []

    for sample in stream_samples(
        args.dataset_name,
        args.dataset_config,
        args.dataset_split,
        args.limit,
        args.offset,
    ):
        LOGGER.info("Processing query_id=%s", sample.query_id)

        prompt_only = build_prompt_only_query(model, sample.query)
        generated_only = generate_text(model, prompt_only, args.max_new_tokens, args.device)

        selected_docs = sample.documents[: args.documents_per_sample]
        prompt_docs = build_prompt_with_documents(model, sample.query, selected_docs)
        generated_docs = generate_text(model, prompt_docs, args.max_new_tokens, args.device)

        stats_only_query.append(has_answer(generated_only, sample.answers))
        stats_with_docs.append(has_answer(generated_docs, sample.answers))

        LOGGER.info("Query: %s", sample.query)
        LOGGER.info("Gold answers: %s", sample.answers)
        LOGGER.info("Prediction (query only): %s", generated_only)
        LOGGER.info("Prediction (with docs): %s", generated_docs)

    def _summarise(matches: List[bool]) -> float:
        return sum(matches) / len(matches) if matches else 0.0

    LOGGER.info("=== Summary ===")
    LOGGER.info("Total samples: %d", len(stats_only_query))
    LOGGER.info("Accuracy (query only): %.3f", _summarise(stats_only_query))
    LOGGER.info("Accuracy (with docs): %.3f", _summarise(stats_with_docs))


if __name__ == "__main__":
    main()

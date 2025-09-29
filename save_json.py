#!/usr/bin/env python3
"""Utilities for flattening and saving Hugging Face datasets.

This module can be executed as a script to download a dataset split and
materialise it as a JSON document indexed by ``doc_id`` where each entry
contains a ``query_id`` (if available) and the associated passage text.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Union

import pandas as pd
from datasets import Dataset, IterableDataset, load_dataset


TEXT_CONTAINER_KEYS: Sequence[str] = (
    "passages",
    "passage_text",
    "passage",
    "documents",
    "document",
    "contexts",
    "context",
    "paragraphs",
    "paragraph",
    "content",
    "body",
    "text",
)

QUERY_ID_KEYS: Sequence[str] = (
    "query_id",
    "queryid",
    "qid",
    "question_id",
    "questionid",
    "id",
)


def _to_scalar(value: Any) -> Union[str, int, float, None]:
    """Extract a scalar identifier from potentially nested containers."""

    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (list, tuple, set)):
        for item in value:
            scalar = _to_scalar(item)
            if scalar is not None:
                return scalar
        return None
    if isinstance(value, Mapping):
        for item in value.values():
            scalar = _to_scalar(item)
            if scalar is not None:
                return scalar
        return None
    return None


def _flatten_text_container(value: Any) -> list[str]:
    """Recursively extract textual content from heterogeneous containers."""

    texts: list[str] = []
    if value is None:
        return texts

    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            texts.append(cleaned)
        return texts

    if isinstance(value, (list, tuple, set)):
        for item in value:
            texts.extend(_flatten_text_container(item))
        return texts

    if isinstance(value, Mapping):
        # Prioritise well-known keys before traversing the rest of the mapping.
        for key in TEXT_CONTAINER_KEYS:
            if key in value:
                texts.extend(_flatten_text_container(value[key]))
        for item in value.values():
            texts.extend(_flatten_text_container(item))
        return texts

    # Fallback: attempt to coerce to string if it is not a mapping or iterable.
    try:
        string_value = str(value).strip()
    except Exception:  # pragma: no cover - highly defensive
        string_value = ""
    if string_value:
        texts.append(string_value)
    return texts


def _extract_query_id(row: Mapping[str, Any]) -> Union[str, int, float, None]:
    for key in QUERY_ID_KEYS:
        if key in row:
            scalar = _to_scalar(row[key])
            if scalar is not None:
                return scalar
    return None


def _extract_passage_texts(row: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in TEXT_CONTAINER_KEYS:
        if key in row:
            texts.extend(_flatten_text_container(row[key]))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_texts: list[str] = []
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


RowIterable = Union[
    str,
    Dataset,
    IterableDataset,
    Iterable[Mapping[str, Any]],
]


def _iter_rows(dataset_source: RowIterable) -> Iterable[Mapping[str, Any]]:
    if isinstance(dataset_source, str):
        if not os.path.exists(dataset_source):
            raise FileNotFoundError(f"Dataset file not found: {dataset_source}")
        frame = pd.read_json(dataset_source, lines=True)
        for row in frame.to_dict(orient="records"):
            yield row
        return

    if isinstance(dataset_source, Dataset):
        for row in dataset_source:
            yield dict(row)
        return

    if isinstance(dataset_source, IterableDataset):
        for row in dataset_source:
            yield dict(row)
        return

    for row in dataset_source:
        if isinstance(row, MutableMapping):
            yield row
        else:
            yield dict(row)


def load_and_flatten(dataset_source: RowIterable) -> Dict[int, Dict[str, Union[str, int, float, None]]]:
    """Load a dataset and flatten it into doc-indexed records."""

    docs: Dict[int, Dict[str, Union[str, int, float, None]]] = {}
    doc_id = 0

    for row in _iter_rows(dataset_source):
        query_id = _extract_query_id(row)
        passage_texts = _extract_passage_texts(row)
        if not passage_texts and "text" in row:
            passage_texts = _flatten_text_container(row["text"])

        for passage_text in passage_texts:
            docs[doc_id] = {"query_id": query_id, "text": passage_text}
            doc_id += 1

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save a Hugging Face dataset split as a JSON mapping of doc_id to "
            "{query_id, text}."
        )
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name or path (e.g. 'squad', 'dmrau/multi_qa')",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to download"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Optional dataset configuration name"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Destination JSON file. Defaults to <dataset>_<split>_flattened.json"
        ),
    )

    args = parser.parse_args()

    print(f"🔄 Loading dataset: {args.dataset}, split: {args.split}")
    dataset = load_dataset(args.dataset, split=args.split, name=args.name)

    print("🧹 Flattening dataset structure")
    docs = load_and_flatten(dataset)

    out_file = args.out or f"{args.dataset.replace('/', '_')}_{args.split}_flattened.json"
    print(f"💾 Saving {len(docs)} passages to {out_file}")
    with open(out_file, "w", encoding="utf-8") as handle:
        json.dump(docs, handle, ensure_ascii=False)

    print("✅ Done!")


if __name__ == "__main__":
    main()

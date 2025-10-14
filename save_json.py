#!/usr/bin/env python3
"""Utilities for flattening and saving Hugging Face datasets.

This module can be executed as a script to download a dataset split and
materialise it as a JSON document indexed by ``doc_id`` where each entry
contains a ``query_id`` (if available), the associated passage text, and any
detected answers.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from itertools import islice
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from tqdm import tqdm


TEXT_CONTAINER_KEYS: Sequence[str] = (
    "passages",
    "passage_text",
    "passage_texts",
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

NON_TEXT_KEYS: Sequence[str] = (
    "url",
    "urls",
    "passage_url",
    "passage_urls",
    "is_selected",
    "is_selecteds",
)

QUERY_ID_KEYS: Sequence[str] = (
    "query_id",
    "queryid",
    "qid",
    "question_id",
    "questionid",
    "id",
)


QUERY_TEXT_KEYS: Sequence[str] = (
    "query",
    "question",
    "query_text",
    "question_text",
    "title",
    "text",
)


ANSWER_CONTAINER_KEYS: Sequence[str] = (
    "answers",
    "answer",
    "gold_answers",
    "ground_truth",
    "ground_truths",
    "groundtruth",
    "groundtruths",
    "label",
    "labels",
    "response",
    "responses",
    "target",
    "targets",
    "completion",
    "completions",
    "output",
    "outputs",
    "answer_text",
    "answer_texts",
)


TEXT_CONTAINER_KEYS_LOWER: Tuple[str, ...] = tuple(key.lower() for key in TEXT_CONTAINER_KEYS)
NON_TEXT_KEYS_LOWER: Tuple[str, ...] = tuple(key.lower() for key in NON_TEXT_KEYS)
QUERY_ID_KEYS_LOWER: Tuple[str, ...] = tuple(key.lower() for key in QUERY_ID_KEYS)
QUERY_TEXT_KEYS_LOWER: Tuple[str, ...] = tuple(key.lower() for key in QUERY_TEXT_KEYS)
ANSWER_CONTAINER_KEYS_LOWER: Tuple[str, ...] = tuple(key.lower() for key in ANSWER_CONTAINER_KEYS)

NO_ANSWER_PLACEHOLDER_NORMALISED: frozenset[str] = frozenset(("no answer present.",))


def _find_matching_value(mapping: Mapping[str, Any], ordered_keys: Sequence[str]) -> Any:
    for candidate in ordered_keys:
        for key, value in mapping.items():
            if isinstance(key, str) and key.lower() == candidate:
                return value
    return None


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
        for candidate in TEXT_CONTAINER_KEYS_LOWER:
            matched = _find_matching_value(value, (candidate,))
            if matched is not None:
                texts.extend(_flatten_text_container(matched))
        for key, item in value.items():
            lowered = key.lower() if isinstance(key, str) else key
            if isinstance(lowered, str) and (
                lowered in TEXT_CONTAINER_KEYS_LOWER or lowered in NON_TEXT_KEYS_LOWER
            ):
                continue
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


def _normalise_answer(answer: str) -> str:
    return answer.strip().lower()


def _is_placeholder_only_answers(answers: Sequence[str]) -> bool:
    if not answers:
        return False
    normalised = [_normalise_answer(answer) for answer in answers if answer.strip()]
    if not normalised:
        return False
    return all(answer in NO_ANSWER_PLACEHOLDER_NORMALISED for answer in normalised)


def _filter_placeholder_answers(answers: Sequence[str]) -> list[str]:
    filtered: list[str] = []
    for answer in answers:
        if _normalise_answer(answer) in NO_ANSWER_PLACEHOLDER_NORMALISED:
            continue
        filtered.append(answer)
    return filtered


def _extract_query_id(row: Mapping[str, Any]) -> Union[str, int, float, None]:
    for candidate in QUERY_ID_KEYS_LOWER:
        value = _find_matching_value(row, (candidate,)) if hasattr(row, "items") else None
        if value is None:
            continue
        scalar = _to_scalar(value)
        if scalar is not None:
            return scalar
    return None


def _extract_passage_texts(row: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    for candidate in TEXT_CONTAINER_KEYS_LOWER:
        value = _find_matching_value(row, (candidate,))
        if value is not None:
            texts.extend(_flatten_text_container(value))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_texts: list[str] = []
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts


def _extract_answer_texts(row: Mapping[str, Any]) -> list[str]:
    answers: list[str] = []
    for candidate in ANSWER_CONTAINER_KEYS_LOWER:
        value = _find_matching_value(row, (candidate,))
        if value is None:
            continue
        answers.extend(_flatten_text_container(value))
    seen: set[str] = set()
    unique_answers: list[str] = []
    for answer in answers:
        if answer not in seen:
            unique_answers.append(answer)
            seen.add(answer)
    return unique_answers


RowIterable = Union[
    str,
    Dataset,
    DatasetDict,
    IterableDataset,
    Iterable[Mapping[str, Any]],
]


def _iter_rows(dataset_source: RowIterable) -> Iterable[Mapping[str, Any]]:
    if isinstance(dataset_source, str):
        if not os.path.isfile(dataset_source):
            raise FileNotFoundError(f"Dataset file not found: {dataset_source}")
        frame = pd.read_json(dataset_source, lines=True)
        for row in frame.to_dict(orient="records"):
            yield row
        return

    if isinstance(dataset_source, Dataset):
        for row in dataset_source:
            yield dict(row)
        return

    if isinstance(dataset_source, DatasetDict):
        for split_name in sorted(dataset_source.keys()):
            split = dataset_source[split_name]
            for row in split:
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


def load_and_flatten(
    dataset_source: RowIterable,
    *,
    split: str | None = None,
    name: str | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
    limit: int | None = None,
) -> Dict[int, Dict[str, Union[str, int, float, None, list[str]]]]:
    """Load a dataset-like object and flatten it into doc-indexed records.

    ``dataset_source`` can be a path to a JSON/JSONL file, a Hugging Face
    ``Dataset``/``DatasetDict``/``IterableDataset`` instance, the name of a
    dataset on the Hugging Face Hub, or any iterable of mapping-like rows. When
    ``dataset_source`` is a Hugging Face dataset identifier the optional
    ``split``, ``name`` and ``load_dataset_kwargs`` arguments are forwarded to
    :func:`datasets.load_dataset`. The optional ``limit`` argument constrains the
    number of rows that will be consumed from ``dataset_source`` before
    flattening passages.

    Each flattened record captures the detected query identifier (if any), the
    associated passage text, and any answer texts that can be inferred from the
    original row.
    """

    if isinstance(dataset_source, str) and not os.path.isfile(dataset_source):
        dataset_identifier = dataset_source
        load_kwargs = dict(load_dataset_kwargs or {})
        if split is not None:
            load_kwargs.setdefault("split", split)
        if name is not None:
            load_kwargs.setdefault("name", name)

        try:
            dataset_source = load_dataset(dataset_identifier, **load_kwargs)
        except Exception as error:  # pragma: no cover - network/IO heavy
            raise ValueError(
                "Failed to load dataset using datasets.load_dataset; "
                f"dataset='{dataset_identifier}', split='{split}', name='{name}'."
            ) from error

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    docs: Dict[int, Dict[str, Union[str, int, float, None, list[str]]]] = {}
    doc_id = 0

    row_iterable = _iter_rows(dataset_source)
    if limit is not None:
        row_iterable = islice(row_iterable, limit)

    for row in tqdm(row_iterable, desc="Flattening dataset", unit="row"):
        query_id = _extract_query_id(row)
        passage_texts = _extract_passage_texts(row)
        if not passage_texts and "text" in row:
            passage_texts = _flatten_text_container(row["text"])
        answer_texts = _extract_answer_texts(row)
        if _is_placeholder_only_answers(answer_texts):
            continue
        answer_texts = _filter_placeholder_answers(answer_texts)

        for passage_text in passage_texts:
            record: Dict[str, Union[str, int, float, None, list[str]]] = {
                "query_id": query_id,
                "text": passage_text,
            }
            if answer_texts:
                record["answers"] = answer_texts
            docs[doc_id] = record
            doc_id += 1

    return docs


def _extract_query_text(row: Mapping[str, Any]) -> Union[str, None]:
    for candidate in QUERY_TEXT_KEYS_LOWER:
        value = _find_matching_value(row, (candidate,)) if hasattr(row, "items") else None
        if value is None:
            continue
        texts = _flatten_text_container(value)
        if texts:
            return texts[0]
    return None


def load_queries(
    dataset_source: RowIterable,
    *,
    split: str | None = None,
    name: str | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, Dict[str, str]]:
    """Load a dataset-like object and extract a mapping of query_id to text."""

    if isinstance(dataset_source, str) and not os.path.isfile(dataset_source):
        dataset_identifier = dataset_source
        load_kwargs = dict(load_dataset_kwargs or {})
        if split is not None:
            load_kwargs.setdefault("split", split)
        if name is not None:
            load_kwargs.setdefault("name", name)

        try:
            dataset_source = load_dataset(dataset_identifier, **load_kwargs)
        except Exception as error:  # pragma: no cover - network/IO heavy
            raise ValueError(
                "Failed to load dataset using datasets.load_dataset; "
                f"dataset='{dataset_identifier}', split='{split}', name='{name}'."
            ) from error

    queries: Dict[str, Dict[str, str]] = {}

    for row in _iter_rows(dataset_source):
        query_id = _extract_query_id(row)
        if query_id is None:
            continue
        answers = _extract_answer_texts(row)
        if _is_placeholder_only_answers(answers):
            continue
        query_text = _extract_query_text(row)
        if not query_text:
            continue
        queries[str(query_id)] = {"text": query_text}

    return queries


def load_answers(
    dataset_source: RowIterable,
    *,
    split: str | None = None,
    name: str | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
) -> Dict[str, list[str]]:
    """Load a dataset-like object and map each query_id to its answer texts."""

    if isinstance(dataset_source, str) and not os.path.isfile(dataset_source):
        dataset_identifier = dataset_source
        load_kwargs = dict(load_dataset_kwargs or {})
        if split is not None:
            load_kwargs.setdefault("split", split)
        if name is not None:
            load_kwargs.setdefault("name", name)

        try:
            dataset_source = load_dataset(dataset_identifier, **load_kwargs)
        except Exception as error:  # pragma: no cover - network/IO heavy
            raise ValueError(
                "Failed to load dataset using datasets.load_dataset; "
                f"dataset='{dataset_identifier}', split='{split}', name='{name}'."
            ) from error

    answers: Dict[str, list[str]] = {}

    for row in _iter_rows(dataset_source):
        query_id = _extract_query_id(row)
        if query_id is None:
            continue
        answer_texts = _extract_answer_texts(row)
        if _is_placeholder_only_answers(answer_texts):
            continue
        answer_texts = _filter_placeholder_answers(answer_texts)
        if not answer_texts:
            continue

        key = str(query_id)
        existing = answers.setdefault(key, [])
        for answer in answer_texts:
            if answer not in existing:
                existing.append(answer)

    return answers


def _filter_payloads_by_query_ids(
    payloads: Mapping[Any, Any], query_ids: Iterable[Any] | None
) -> Dict[Any, Any]:
    if query_ids is None:
        return dict(payloads)

    allowed_ids = {query_id for query_id in query_ids if query_id is not None}
    if not allowed_ids:
        return {}

    allowed_ids_str = {str(query_id) for query_id in allowed_ids}

    def _matches(candidate: Any) -> bool:
        if candidate in allowed_ids:
            return True
        try:
            candidate_str = str(candidate)
        except Exception:  # pragma: no cover - defensive
            return False
        return candidate_str in allowed_ids_str

    return {
        query_id: payload
        for query_id, payload in payloads.items()
        if _matches(query_id)
    }


def _estimate_memory_usage(obj: Any) -> Dict[str, float]:
    """Approximate the in-memory footprint of ``obj`` and its serialised size."""

    base_size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, list):
        items = list(enumerate(obj))
    else:
        items = []

    sample_size = min(len(items), 100)
    sampled_bytes = 0
    for key, value in items[:sample_size]:
        sampled_bytes += sys.getsizeof(key)
        sampled_bytes += sys.getsizeof(value)

    multiplier = (len(items) / sample_size) if sample_size else 1
    approx_bytes = base_size + sampled_bytes * multiplier

    try:
        serialized = pickle.dumps(obj)
        pickle_bytes = len(serialized)
    except Exception:  # pragma: no cover - defensive serialisation attempt
        pickle_bytes = 0

    return {
        "approx_memory_mb": approx_bytes / (1024 * 1024),
        "pickle_disk_mb": pickle_bytes / (1024 * 1024),
    }


def save_json_to_path(
    data: Mapping[Any, Any],
    path: str,
    *,
    ensure_directory: bool = True,
) -> Tuple[str, Dict[str, float]]:
    """Persist ``data`` to ``path`` and report memory/disk usage statistics."""

    directory = os.path.dirname(path)
    if ensure_directory and directory:
        os.makedirs(directory, exist_ok=True)

    mem_stats = _estimate_memory_usage(data)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)

    mem_stats["json_disk_mb"] = os.path.getsize(path) / (1024 * 1024)
    return path, mem_stats


def save_json(
    data: Mapping[Any, Any], directory: str, filename: str
) -> Tuple[str, Dict[str, float]]:
    """Persist ``data`` inside ``directory`` and report memory/disk usage."""

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    return save_json_to_path(data, path, ensure_directory=False)


def save_queries_json(
    dataset_source: RowIterable,
    directory: str,
    filename: str = "queries.json",
    query_ids: Iterable[Any] | None = None,
    *,
    split: str | None = None,
    name: str | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Extract queries and persist them as ``{"query_id": {"text": ...}}`` JSON."""

    queries = load_queries(
        dataset_source,
        split=split,
        name=name,
        load_dataset_kwargs=load_dataset_kwargs,
    )

    filtered_queries = _filter_payloads_by_query_ids(queries, query_ids)
    path, mem_stats = save_json(filtered_queries, directory, filename)
    mem_stats["count"] = len(filtered_queries)
    return path, mem_stats


def save_answers_json(
    dataset_source: RowIterable,
    directory: str,
    filename: str = "answers.json",
    query_ids: Iterable[Any] | None = None,
    *,
    split: str | None = None,
    name: str | None = None,
    load_dataset_kwargs: Mapping[str, Any] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Extract answers and persist them as ``{"query_id": [...]}`` JSON."""

    answers = load_answers(
        dataset_source,
        split=split,
        name=name,
        load_dataset_kwargs=load_dataset_kwargs,
    )

    filtered_answers = _filter_payloads_by_query_ids(answers, query_ids)
    path, mem_stats = save_json(filtered_answers, directory, filename)
    mem_stats["count"] = len(filtered_answers)
    return path, mem_stats


def trim_json_file(
    input_path: str,
    output_path: str,
    limit: int,
    *,
    ensure_directory: bool = True,
) -> Tuple[str, Dict[str, float]]:
    """Trim a JSON document to ``limit`` entries and persist the result."""

    if limit < 0:
        raise ValueError("limit must be non-negative")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        if limit == 0:
            trimmed = {}
        else:
            items = list(data.items())[:limit]
            trimmed = dict(items)
    elif isinstance(data, list):
        if limit == 0:
            trimmed = []
        else:
            trimmed = data[:limit]
    else:
        raise TypeError(
            "Unsupported JSON structure; expected an object or array at the root"
        )

    return save_json_to_path(trimmed, output_path, ensure_directory=ensure_directory)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save a Hugging Face dataset split as a JSON mapping of doc_id to "
            "{query_id, text, answers}."
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
        default="data/docs.json",
        help=(
            "Destination JSON file. Defaults to <dataset>_<split>_flattened.json"
        ),
    )
    parser.add_argument(
        "--queries-out",
        type=str,
        default="data/queries.json",
        help="Optional path to persist query_id to query text mapping as JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of rows to process when flattening",
    )

    args = parser.parse_args()

    print(f"🔄 Loading dataset: {args.dataset}, split: {args.split}")
    dataset = load_dataset(args.dataset, split=args.split, name=args.name)

    print("🧹 Flattening dataset structure")
    docs = load_and_flatten(dataset, split=args.split, name=args.name, limit=args.limit)

    out_file = args.out or f"{args.dataset.replace('/', '_')}_{args.split}_flattened.json"
    path, mem_stats = save_json_to_path(docs, out_file)
    print(
        "💾 Saved {count} passages to {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=len(docs),
            path=path,
            mem=mem_stats.get("approx_memory_mb", 0.0),
            pickle=mem_stats.get("pickle_disk_mb", 0.0),
            json=mem_stats.get("json_disk_mb", 0.0),
        )
    )

    if args.queries_out:
        print("🧾 Extracting queries")
        queries = load_queries(dataset)
        if not queries:
            print("⚠️ No queries found in dataset; skipping query export.")
        else:
            q_path, q_stats = save_json_to_path(queries, args.queries_out)
            print(
                "📝 Saved {count} queries to {path} "
                "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
                    count=len(queries),
                    path=q_path,
                    mem=q_stats.get("approx_memory_mb", 0.0),
                    pickle=q_stats.get("pickle_disk_mb", 0.0),
                    json=q_stats.get("json_disk_mb", 0.0),
                )
            )

    print("✅ Done!")


if __name__ == "__main__":
    main()
import json
import math
import os
import random
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import h5py
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, TrainerCallback
from fine_tuning_parser import get_fine_tuning_args
from datasets.fingerprint import Hasher
from metrics import exact_match_score, f1_score
from modeling_cocom import COCOM, COCOMConfig
from transformers.trainer_utils import get_last_checkpoint

import wandb

random.seed(42)


def _to_str_doc_id(doc_id: object) -> str:
    try:
        if isinstance(doc_id, bytes):
            return doc_id.decode("utf-8")
        if isinstance(doc_id, (np.integer, np.int64, np.int32)):
            return str(int(doc_id))
        return str(doc_id)
    except Exception:
        return str(doc_id)


def load_context_store(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Context store not found at {path}")

    context_index: Dict[str, int] = {}

    with h5py.File(path, "r") as handle:
        if "contexts" not in handle or "shapes" not in handle or "doc_ids" not in handle:
            raise KeyError("Context store is missing required datasets: 'contexts', 'shapes', or 'doc_ids'.")

        contexts_ds = handle["contexts"]
        shapes_ds = handle["shapes"]
        doc_ids_raw = handle["doc_ids"]

        if len(contexts_ds) != len(doc_ids_raw) or len(shapes_ds) != len(doc_ids_raw):
            raise ValueError("Mismatch between contexts, shapes, and doc id entries in context store")

        for idx, raw_id in enumerate(doc_ids_raw):
            doc_id = _to_str_doc_id(raw_id)
            context_index[doc_id] = idx

    return context_index


@dataclass
class FaissRetriever:
    index: "faiss.Index"
    index_to_doc_id: List[str]
    doc_id_to_index: Dict[str, int]

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[str]:
        results = self.search_batch(query_embedding, k=k)
        return results[0] if results else []

    def search_batch(
        self, query_embeddings: np.ndarray | Iterable[np.ndarray], k: int = 5
    ) -> List[List[str]]:
        vectors = np.asarray(query_embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim > 2:
            vectors = vectors.reshape(vectors.shape[0], -1)
        vectors = np.ascontiguousarray(vectors)

        faiss.normalize_L2(vectors)
        _scores, indices = self.index.search(vectors, k)

        batch_doc_ids: List[List[str]] = []
        for row in indices:
            doc_ids: List[str] = []
            for idx in row:
                if idx < 0 or idx >= len(self.index_to_doc_id):
                    continue
                doc_ids.append(self.index_to_doc_id[idx])
            batch_doc_ids.append(doc_ids)

        return batch_doc_ids


def load_retriever(embeddings_path: str, docs_path: Optional[str]) -> FaissRetriever:
    with np.load(embeddings_path) as embeddings_file:
        doc_ids = [_to_str_doc_id(doc_id) for doc_id in embeddings_file["doc_ids"]]
        embeddings = embeddings_file["embeddings"]

    if embeddings.shape[0] != len(doc_ids):
        raise ValueError("Number of embeddings does not match number of document IDs.")

    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_to_doc_id = list(doc_ids)
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(index_to_doc_id)}

    return FaissRetriever(index=index, index_to_doc_id=index_to_doc_id, doc_id_to_index=doc_id_to_index)


def load_query_embedding_store(
    embeddings_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Query embeddings not found at {embeddings_path}")

    with np.load(embeddings_path) as embeddings_file:
        query_ids_raw = embeddings_file["query_ids"]
        query_texts_raw = embeddings_file["query_texts"]
        embeddings = np.asarray(embeddings_file["embeddings"], dtype=np.float32)

    if len(query_ids_raw) != len(embeddings) or len(query_texts_raw) != len(embeddings):
        raise ValueError("Mismatch between query ids, texts, and embeddings in query store.")

    faiss.normalize_L2(embeddings)

    query_vectors: Dict[str, np.ndarray] = {}
    query_text_lookup: Dict[str, str] = {}
    for idx, raw_id in enumerate(query_ids_raw):
        query_id = _to_str_doc_id(raw_id)
        query_vectors[query_id] = embeddings[idx]
        query_text_lookup[query_id] = str(query_texts_raw[idx])

    return query_vectors, query_text_lookup


def _extract_doc_text(payload: object) -> Optional[str]:
    if payload is None:
        return None
    if isinstance(payload, str):
        text = payload.strip()
        return text or None
    if isinstance(payload, dict):
        for key in ("text", "contents", "content", "body", "document", "passage"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_docs_lookup(docs_path: Optional[str]) -> Dict[str, str]:
    if not docs_path or not os.path.exists(docs_path):
        return {}

    with open(docs_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    lookup: Dict[str, str] = {}
    if isinstance(payload, dict):
        for doc_id, value in payload.items():
            text = _extract_doc_text(value)
            if text:
                lookup[_to_str_doc_id(doc_id)] = text
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            doc_id = entry.get("doc_id") or entry.get("id") or entry.get("document_id")
            if doc_id is None:
                continue
            text = _extract_doc_text(entry)
            if text:
                lookup[_to_str_doc_id(doc_id)] = text
    return lookup


def _format_doc_preview(text: str, max_lines: int = 3, max_chars: int = 200) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    selected = lines[:max_lines]
    preview = "\n".join(selected)
    if len(preview) > max_chars:
        preview = preview[: max_chars - 3].rstrip() + "..."
    return preview


def preview_retrieval(
    split_name: str,
    examples: List[Dict[str, object]],
    retriever: FaissRetriever,
    query_vectors: Dict[str, np.ndarray],
    docs_lookup: Dict[str, str],
    top_k: int,
) -> None:
    if not examples:
        print(f"No examples available in the {split_name} split for retrieval preview.")
        return

    sample = random.choice(examples)
    question = sample.get("question") or sample.get("query") or ""
    if not question:
        print(f"Selected example from {split_name} split does not contain a question.")
        return

    query_id = sample.get("query_id")
    if query_id is None:
        print(f"Selected example from {split_name} split is missing a query id.")
        return

    lookup_key = _to_str_doc_id(query_id)
    if lookup_key not in query_vectors:
        print(
            f"Query id {lookup_key} does not have a precomputed embedding; unable to preview retrieval."
        )
        return

    retrieved_doc_ids = retriever.search(query_vectors[lookup_key], k=top_k)

    print("\n=== Retrieval preview ({split}) ===".format(split=split_name))
    print(f"Query: {question}")
    if not retrieved_doc_ids:
        print("No documents retrieved for the preview query.")
        return

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        doc_key = _to_str_doc_id(doc_id)
        print(f"{rank}. doc_id={doc_key}")
        doc_text = docs_lookup.get(doc_key)
        if not doc_text:
            print("   [document text unavailable]")
            continue
        preview = _format_doc_preview(doc_text)
        if preview:
            indented = textwrap.indent(preview, prefix="   ")
            print(indented)
        else:
            print("   [document text unavailable]")


def _flatten_answer_texts(payload: object) -> List[str]:
    if payload is None:
        return []
    if isinstance(payload, str):
        text = payload.strip()
        return [text] if text else []
    if isinstance(payload, (list, tuple, set)):
        texts: List[str] = []
        for item in payload:
            texts.extend(_flatten_answer_texts(item))
        return texts
    if isinstance(payload, dict):
        for key in ("text", "answers", "answer", "labels"):
            if key in payload:
                texts = _flatten_answer_texts(payload[key])
                if texts:
                    return texts
        texts: List[str] = []
        for value in payload.values():
            texts.extend(_flatten_answer_texts(value))
        return texts
    return []


def _normalise_answers_payload(payload: object) -> Optional[Dict[str, List[str]]]:
    texts: List[str] = []
    for answer in _flatten_answer_texts(payload):
        if answer and answer not in texts:
            texts.append(answer)
    if not texts:
        return None
    return {"text": texts}


def _extract_answer(example: Dict) -> str:
    answers = example.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or answers.get("answers")
        if texts and isinstance(texts, (list, tuple)):
            first = texts[0]
            if isinstance(first, dict):
                return first.get("text", "")
            return first
    if isinstance(answers, (list, tuple)) and answers:
        first = answers[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return first
    return example.get("answer", "") or ""


def _extract_question_text(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("question", "query", "text", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for value in payload.values():
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _coerce_query_mapping(data: object) -> Dict[str, object]:
    mapping: Dict[str, object] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            mapping[_to_str_doc_id(key)] = value
    elif isinstance(data, list):
        for index, value in enumerate(data):
            query_id: Optional[object] = None
            if isinstance(value, dict):
                query_id = (
                    value.get("query_id")
                    or value.get("id")
                    or value.get("qid")
                    or value.get("question_id")
                )
            mapping[_to_str_doc_id(query_id if query_id is not None else index)] = value
    return mapping


def _coerce_queries_by_split(queries_payload: object) -> Dict[str, Dict[str, object]]:
    if isinstance(queries_payload, dict):
        lower_keys = {str(key).lower() for key in queries_payload.keys()}
        if any(key in {"train", "validation", "val", "dev", "test"} for key in lower_keys):
            splits: Dict[str, Dict[str, object]] = {}
            for split_name, value in queries_payload.items():
                splits[str(split_name).lower()] = _coerce_query_mapping(value)
            return splits
        return {"train": _coerce_query_mapping(queries_payload)}
    if isinstance(queries_payload, list):
        return {"train": _coerce_query_mapping(queries_payload)}
    raise ValueError("Unsupported queries JSON structure. Expected dict or list.")


def _coerce_answers_lookup(answers_payload: object) -> Dict[str, object]:
    if answers_payload is None:
        return {}
    lookup: Dict[str, object] = {}
    if isinstance(answers_payload, dict):
        for key, value in answers_payload.items():
            lookup[_to_str_doc_id(key)] = value
    elif isinstance(answers_payload, list):
        for entry in answers_payload:
            if isinstance(entry, dict):
                query_id = (
                    entry.get("query_id")
                    or entry.get("id")
                    or entry.get("qid")
                    or entry.get("question_id")
                )
                if query_id is None:
                    continue
                lookup[_to_str_doc_id(query_id)] = entry
    return lookup


def _lookup_answer_payload(answers_lookup: Dict[str, object], query_id: str) -> Optional[object]:
    if query_id in answers_lookup:
        return answers_lookup[query_id]
    try:
        numeric_key = str(int(float(query_id)))
    except Exception:
        numeric_key = None
    if numeric_key and numeric_key in answers_lookup:
        return answers_lookup[numeric_key]
    return None


def _build_split_examples(
    queries: Dict[str, object],
    answers_lookup: Dict[str, object],
) -> List[Dict[str, object]]:
    examples: List[Dict[str, object]] = []
    for query_id, payload in queries.items():
        question = _extract_question_text(payload)
        if not question:
            continue

        answer_payload: Optional[object] = None
        if isinstance(payload, dict):
            for key in ("answers", "answer"):
                if key in payload:
                    answer_payload = payload[key]
                    break
        if answer_payload is None:
            answer_payload = _lookup_answer_payload(answers_lookup, query_id)

        answers_dict = _normalise_answers_payload(answer_payload)
        if answers_dict is None:
            continue

        examples.append({"query_id": query_id, "question": question, "answers": answers_dict})
    return examples


def load_local_qa_splits(
    queries_path: str,
    answers_path: Optional[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not os.path.exists(queries_path):
        raise FileNotFoundError(f"Queries JSON not found at {queries_path}.")

    with open(queries_path, "r", encoding="utf-8") as handle:
        queries_payload = json.load(handle)

    answers_payload: Optional[object] = None
    if answers_path and os.path.exists(answers_path):
        with open(answers_path, "r", encoding="utf-8") as handle:
            answers_payload = json.load(handle)

    queries_by_split = _coerce_queries_by_split(queries_payload)
    answers_lookup = _coerce_answers_lookup(answers_payload)

    split_examples: Dict[str, List[Dict[str, object]]] = {}
    for split_name, split_queries in queries_by_split.items():
        split_examples[split_name] = _build_split_examples(split_queries, answers_lookup)

    if not split_examples:
        raise ValueError("No queries with answers found in the provided JSON files.")

    train_candidates = [
        key for key in ("train", "training") if key in split_examples and split_examples[key]
    ]
    if train_candidates:
        train_examples = split_examples.pop(train_candidates[0])
    else:
        first_key = next(iter(split_examples))
        train_examples = split_examples.pop(first_key)

    eval_candidates = [
        key
        for key in ("validation", "val", "dev", "test")
        if key in split_examples and split_examples[key]
    ]

    if eval_candidates:
        eval_examples = split_examples.pop(eval_candidates[0])
    else:
        if len(train_examples) < 2:
            eval_examples = list(train_examples)
        else:
            eval_count = min(16, int(round(len(train_examples) * 0.1)))
            eval_indices = set(random.sample(range(len(train_examples)), eval_count))
            eval_examples = [
                example for idx, example in enumerate(train_examples) if idx in eval_indices
            ]
            train_examples = [
                example for idx, example in enumerate(train_examples) if idx not in eval_indices
            ]

    if not train_examples:
        raise ValueError("No training queries available after processing local JSON files.")
    if not eval_examples:
        raise ValueError("No evaluation queries available after processing local JSON files.")

    return train_examples, eval_examples


class RAGFineTuneDataset(Dataset):
    def __init__(
        self,
        qa_split: Iterable[Dict],
        retriever: FaissRetriever,
        query_vectors: Dict[str, np.ndarray],
        query_text_lookup: Dict[str, str],
        context_store_path: str,
        doc_id_to_index: Dict[str, int],
        top_k: int,
    ) -> None:
        if not os.path.exists(context_store_path):
            raise FileNotFoundError(f"Context store not found at {context_store_path}")

        self.context_store_path = context_store_path
        self.doc_id_to_index = doc_id_to_index
        self.top_k = top_k
        self.samples: List[Dict[str, object]] = []

        skipped_due_to_contexts = 0
        skipped_due_to_embeddings = 0
        mismatched_query_ids: set[str] = set()
        batch_examples: List[Dict[str, object]] = []
        batch_vectors: List[np.ndarray] = []
        batch_size = 64

        def process_batch() -> None:
            nonlocal batch_examples, batch_vectors, skipped_due_to_contexts
            if not batch_examples:
                return

            vectors = np.asarray(batch_vectors, dtype=np.float32)
            retrieved_batches = retriever.search_batch(vectors, k=top_k)

            for example_info, retrieved_doc_ids in zip(batch_examples, retrieved_batches):
                doc_ids: List[str] = []
                for doc_id in retrieved_doc_ids:
                    key = _to_str_doc_id(doc_id)
                    if key in self.doc_id_to_index:
                        doc_ids.append(key)
                if len(doc_ids) < top_k:
                    skipped_due_to_contexts += 1
                    continue
                self.samples.append(
                    {
                        "query_id": example_info["query_id"],
                        "question": example_info["question"],
                        "answer": example_info["answer"],
                        "doc_ids": doc_ids[:top_k],
                    }
                )

            batch_examples = []
            batch_vectors = []

        for example in tqdm(qa_split, desc="Preparing RAG QA split"):
            question = example.get("question") or example.get("query") or ""
            if not question:
                continue
            answer = _extract_answer(example)
            query_id_raw = example.get("query_id")
            if query_id_raw is None:
                skipped_due_to_embeddings += 1
                continue
            query_id = _to_str_doc_id(query_id_raw)
            vector = query_vectors.get(query_id)
            if vector is None:
                skipped_due_to_embeddings += 1
                continue

            stored_text = query_text_lookup.get(query_id)
            if stored_text and stored_text.strip() and question.strip():
                if stored_text.strip() != question.strip() and query_id not in mismatched_query_ids:
                    tqdm.write(
                        f"Warning: Question text for query id {query_id} does not match stored query text."
                    )
                    mismatched_query_ids.add(query_id)

            batch_examples.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "answer": answer,
                }
            )
            batch_vectors.append(vector)

            if len(batch_examples) >= batch_size:
                process_batch()

        process_batch()

        retained = len(self.samples)
        if retained == 0:
            raise ValueError("No training examples available after retrieval. Check retrieval resources.")
        print(
            "Prepared {retained} examples for RAG fine-tuning (skipped {skipped_ctx} without sufficient contexts, {skipped_emb} without embeddings).".format(
                retained=retained,
                skipped_ctx=skipped_due_to_contexts,
                skipped_emb=skipped_due_to_embeddings,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        doc_ids: Iterable[str] = sample["doc_ids"]

        context_tensors: List[torch.Tensor] = []
        with h5py.File(self.context_store_path, "r") as handle:
            contexts_ds = handle["contexts"]
            shapes_ds = handle["shapes"]

            for doc_id in doc_ids:
                if doc_id not in self.doc_id_to_index:
                    raise KeyError(f"Document id {doc_id} not found in context store index")

                context_idx = self.doc_id_to_index[doc_id]
                flattened = np.asarray(contexts_ds[context_idx], dtype=np.float32)
                shape_arr = np.asarray(shapes_ds[context_idx], dtype=np.int64)
                shape = tuple(int(dim) for dim in shape_arr.tolist())
                if not shape:
                    raise ValueError(f"Missing shape information for context index {context_idx}")

                context_array = flattened.reshape(shape)
                context_tensor = torch.from_numpy(context_array)
                if context_tensor.dim() == 3 and context_tensor.size(0) == 1:
                    context_tensor = context_tensor.squeeze(0)
                tensor = context_tensor.to(dtype=torch.float32)
                context_tensors.append(tensor)

        if not context_tensors:
            raise ValueError("Missing context tensors for retrieved document ids.")


        max_mem = max(tensor.size(0) for tensor in context_tensors)
        hidden = context_tensors[0].size(1)
        padded_contexts = []
        for tensor in context_tensors:
            if tensor.size(0) < max_mem:
                pad = torch.zeros((max_mem - tensor.size(0), hidden), dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded_contexts.append(tensor)
        stacked = torch.stack(padded_contexts, dim=0)

        return {
            "question": question,
            "answer": answer,
            "context_embeddings": stacked,
        }


@dataclass
class RAGDataCollator:
    tokenizer: any
    top_k: int
    decoder_max_length: int

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        context_embeddings = [feature["context_embeddings"] for feature in features]
        questions = [feature["question"] for feature in features]
        answers = [feature["answer"] for feature in features]

        top_k = context_embeddings[0].size(0)
        if top_k != self.top_k:
            raise ValueError(
                f"Batch contexts use {top_k} documents but collator was initialised for {self.top_k}."
            )
        if any(tensor.size(0) != top_k for tensor in context_embeddings):
            raise ValueError("All samples must have the same number of retrieved contexts.")

        hidden = context_embeddings[0].size(2)
        bos = self.tokenizer.bos_token or ""
        mem_token_symbol = getattr(self.tokenizer, "mem_token", "") or ""

        trimmed_contexts: List[torch.Tensor] = []
        mem_per_context: List[int] = []
        for tensor, question in zip(context_embeddings, questions):
            sample_max_mem = tensor.size(1)
            base_prompt = f"{bos}[INST]{question}\n[/INST]\n"
            base_encoding = self.tokenizer(
                base_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                padding=False,
                truncation=False,
            )
            base_ids = base_encoding.get("input_ids", [])
            if base_ids and isinstance(base_ids[0], list):
                base_ids = base_ids[0]
            base_len = len(base_ids)
            available_mem_tokens = max(0, self.decoder_max_length - base_len)
            mem_tokens_per_context = min(sample_max_mem, available_mem_tokens // top_k)

            trimmed = tensor[:, :mem_tokens_per_context, :]
            trimmed_contexts.append(trimmed)
            mem_per_context.append(mem_tokens_per_context)

        max_effective_mem = max(mem_per_context) if mem_per_context else 0

        padded_contexts: List[torch.Tensor] = []
        for tensor, effective_mem in zip(trimmed_contexts, mem_per_context):
            if effective_mem < max_effective_mem:
                pad = torch.zeros(
                    (top_k, max_effective_mem - effective_mem, hidden),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                tensor = torch.cat([tensor, pad], dim=1)
            padded_contexts.append(tensor)
        context_batch = torch.stack(padded_contexts, dim=0)

        prompts: List[str] = []
        for question, mem_count in zip(questions, mem_per_context):
            mem_tokens = mem_token_symbol * mem_count if mem_count > 0 else ""
            mem_block = mem_tokens * top_k
            prompts.append(f"{bos}{mem_block}[INST]{question}\n[/INST]\n")

        dec_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.decoder_max_length,
            truncation=True,
            add_special_tokens=False,
        )

        labels = self.tokenizer(
            answers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.decoder_max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "dec_input_ids": dec_inputs["input_ids"],
            "dec_attention_mask": dec_inputs["attention_mask"],
            "labels": labels,
            "context_embeddings": context_batch,
        }
        return batch


class FineTuningTrainer(Trainer):
    def __init__(self, *args, save_only_final: bool = True, **kwargs):
        self.save_only_final = save_only_final
        super().__init__(*args, **kwargs)

    def training_step(self, model, *args):
        inputs = args[0] if len(args) > 0 else None
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        nan_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_gradients = True
                param.grad = None

        if nan_gradients:
            print("NaN gradient detected, skipping optimizer step.")
            return torch.tensor(0.0, requires_grad=False).to(loss.device)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        context_embeddings = inputs.get("context_embeddings")
        if context_embeddings is not None:
            outputs = model.module.forward_with_context_embeddings(
                context_embeddings=context_embeddings,
                dec_input_ids=inputs["dec_input_ids"],
                dec_attention_mask=inputs["dec_attention_mask"],
                labels=inputs.get("labels"),
            )
        else:
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        if not self.save_only_final or self.state.global_step >= self.state.max_steps:
            super()._save_checkpoint(model, trial, metrics)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not self.save_only_final or self.state.global_step >= self.state.max_steps:
            super().save_model(output_dir=output_dir, _internal_call=_internal_call)


class RetrievalPreviewCallback(TrainerCallback):
    def __init__(
        self,
        accelerator: Accelerator,
        split_name: str,
        examples: List[Dict[str, object]],
        retriever: FaissRetriever,
        query_vectors: Dict[str, np.ndarray],
        docs_lookup: Dict[str, str],
        top_k: int,
    ) -> None:
        self.accelerator = accelerator
        self.split_name = split_name
        self.examples = examples
        self.retriever = retriever
        self.query_vectors = query_vectors
        self.docs_lookup = docs_lookup
        self.top_k = top_k

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.accelerator.is_main_process:
            return control
        if not self.examples or not self.query_vectors:
            return control
        preview_retrieval(
            self.split_name,
            self.examples,
            self.retriever,
            self.query_vectors,
            self.docs_lookup,
            top_k=self.top_k,
        )
        return control


def log_save_event(log_path: str, step: int, save_dir: str, event_type: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "step": int(step),
        "path": save_dir,
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry) + "\n")


def overwrite_and_log_model(
    accelerator: Accelerator,
    model: torch.nn.Module,
    target_dir: str,
    log_path: str,
    step: int,
    event_type: str,
    config_path: Optional[str] = None,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(target_dir)
        if config_path and os.path.exists(config_path):
            shutil.copy2(config_path, os.path.join(target_dir, os.path.basename(config_path)))
        log_save_event(log_path, step, target_dir, event_type)
        print(f"Saved model checkpoint to: {target_dir} (step {step}, event={event_type})")
    accelerator.wait_for_everyone()


class PeriodicModelSaver(TrainerCallback):
    def __init__(
        self,
        accelerator: Accelerator,
        save_interval: Optional[int],
        target_dir: str,
        log_path: str,
        config_path: Optional[str] = None,
    ) -> None:
        self.accelerator = accelerator
        self.save_interval = max(int(save_interval), 0) if save_interval is not None else 0
        self.target_dir = target_dir
        self.log_path = log_path
        self.config_path = config_path

    def on_step_end(self, args, state, control, **kwargs):
        if self.save_interval <= 0:
            return control
        if state.global_step == 0 or state.global_step % self.save_interval != 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        overwrite_and_log_model(
            accelerator=self.accelerator,
            model=model,
            target_dir=self.target_dir,
            log_path=self.log_path,
            step=state.global_step,
            event_type="periodic",
            config_path=self.config_path,
        )
        return control


def compute_metrics(eval_pred, model):
    logits_list, labels = eval_pred
    if isinstance(logits_list, tuple):
        logits_list = logits_list[0]

    logits = logits_list[0] if isinstance(logits_list, list) else logits_list

    preds = np.argmax(logits, axis=-1)
    original_model = model.module if hasattr(model, "module") else model
    ignore_positions = labels == -100

    labels = labels.copy()
    preds = preds.copy()
    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    metrics = {}
    em = exact_match_score(preds_str, labels_str)
    f1 = f1_score(preds_str, labels_str)
    metrics.update({"EM": em, "F1": f1})

    sample_pairs = list(zip(preds_str, labels_str))
    if sample_pairs:
        print("Sample predictions vs references:")
        for index, (prediction, reference) in enumerate(sample_pairs[:3]):
            print(f"[{index}] Predicted: {prediction}")
            print(f"    Reference: {reference}")

    return metrics


def build_cycle_config(model_source: str, args) -> COCOMConfig:
    cfg = COCOMConfig.from_pretrained(model_source)
    if args.compression_rates is not None:
        cfg.compr_rates = list(args.compression_rates)
    if args.compression_linear_type is not None:
        cfg.compr_linear_type = args.compression_linear_type
    cfg.generation_top_k = args.retriever_top_k
    cfg.lora = args.lora.lower() == "true"
    return cfg


def compute_cycle_offsets(args) -> List[int]:
    if not getattr(args, "auto_continue_training", False):
        return []
    total_cycles = max(int(getattr(args, "auto_continue_cycles", 1)), 1)
    initial_offset = int(getattr(args, "compress_initial_offset", 0))
    increment = int(getattr(args, "compress_offset_increment", 100000))
    return [initial_offset + idx * increment for idx in range(total_cycles)]


def run_compress_command(args, offset: int, cycle_index: int, total_cycles: int) -> None:
    script_path = getattr(args, "compress_script_path", "compress.py")
    command: List[str] = ["python3", script_path]

    hf_model_name = getattr(args, "compress_hf_model_name", None)
    if hf_model_name:
        command.extend(["--hf_model_name", hf_model_name])

    limit = getattr(args, "compress_limit", None)
    if limit is not None:
        command.extend(["--limit", str(limit)])

    context_batch_size = getattr(args, "compress_context_batch_size", None)
    if context_batch_size is not None:
        command.extend(["--context_batch_size", str(context_batch_size)])

    if getattr(args, "compress_use_hf_dataset", True):
        command.append("--use-hf-dataset")

    dataset_name = getattr(args, "compress_hf_dataset_name", None)
    if dataset_name:
        command.extend(["--hf-dataset-name", dataset_name])

    dataset_config = getattr(args, "compress_hf_dataset_config", None)
    if dataset_config:
        command.extend(["--hf-dataset-config", dataset_config])

    dataset_split = getattr(args, "compress_hf_dataset_split", None)
    if dataset_split:
        command.extend(["--hf-dataset-split", dataset_split])

    command.extend(["--hf-offset", str(offset)])

    additional_args = getattr(args, "compress_additional_args", None)
    if additional_args:
        command.extend(additional_args)

    env = os.environ.copy()
    cuda_devices = getattr(args, "compress_cuda_devices", None)
    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    print(
        f"Running compress.py for cycle {cycle_index + 1}/{total_cycles} with offset {offset}."
    )
    print("Compress command:", " ".join(command))
    subprocess.run(command, check=True, env=env)


def run_training_cycle(
    accelerator: Accelerator,
    args,
    cfg: COCOMConfig,
    model_source: str,
    cycle_index: int,
    total_cycles: int,
    base_tmp_output_dir: str,
    final_model_dir: str,
    save_log_path: str,
    offset: Optional[int],
) -> None:
    cycle_tmp_dir = os.path.join(base_tmp_output_dir, f"cycle_{cycle_index:02d}")
    model_output_dir = os.path.join(cycle_tmp_dir, "checkpoints")
    config_path = os.path.join(cycle_tmp_dir, "training_config.json")

    if accelerator.is_main_process:
        os.makedirs(args.experiment_folder, exist_ok=True)
        if os.path.exists(cycle_tmp_dir):
            shutil.rmtree(cycle_tmp_dir)
        os.makedirs(model_output_dir, exist_ok=True)
        offset_display = offset if offset is not None else "N/A"
        print(
            f"Starting training cycle {cycle_index + 1}/{total_cycles} (offset={offset_display})"
        )
        print(f"Temporary outputs: {cycle_tmp_dir}")
        print(f"Final model path: {final_model_dir}")
        print(f"Checkpoint log path: {save_log_path}")

    accelerator.wait_for_everyone()
    os.makedirs(model_output_dir, exist_ok=True)

    context_index = load_context_store(args.rag_contexts_path)
    retriever = load_retriever(args.rag_embeddings_path, args.rag_docs_path)

    query_embeddings_path = getattr(args, "rag_query_embeddings_path", None)
    if not query_embeddings_path:
        query_embeddings_path = "data/query_embeddings.npz"
    query_vectors, query_text_lookup = load_query_embedding_store(query_embeddings_path)
    print(
        f"Loaded {len(query_vectors)} query embeddings from {query_embeddings_path}"
    )

    train_examples, eval_examples = load_local_qa_splits(
        queries_path=args.rag_queries_path,
        answers_path=args.rag_answers_path,
    )

    docs_lookup = load_docs_lookup(args.rag_docs_path)

    train_dataset = RAGFineTuneDataset(
        train_examples,
        retriever,
        query_vectors,
        query_text_lookup,
        args.rag_contexts_path,
        context_index,
        top_k=args.retriever_top_k,
    )
    eval_dataset = RAGFineTuneDataset(
        eval_examples,
        retriever,
        query_vectors,
        query_text_lookup,
        args.rag_contexts_path,
        context_index,
        top_k=args.retriever_top_k,
    )

    model = COCOM.from_pretrained(model_source, config=cfg)
    model.generation_top_k = args.retriever_top_k

    if accelerator.is_main_process:
        print(model)

    data_collator = RAGDataCollator(
        tokenizer=model.decoder_tokenizer,
        top_k=args.retriever_top_k,
        decoder_max_length=args.decoder_max_length,
    )

    evaluation_strategy = "steps" if args.eval_every_steps > 0 else "no"
    eval_steps = max(args.eval_every_steps, 1)
    logging_steps = max(args.eval_every_steps if args.eval_every_steps > 0 else 10, 1)

    epochs = max(args.epochs, 1)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=args.lr,
        eval_accumulation_steps=args.gradient_accumulation,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(args.per_device_batch_size // 4, 1),
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy=evaluation_strategy,
        save_strategy="no",
        report_to=None,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=16,
        do_eval=evaluation_strategy != "no",
        max_grad_norm=1.0,
        remove_unused_columns=False,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        num_train_epochs=epochs,
    )

    accelerator_state = getattr(accelerator, "state", None)
    world_size = getattr(accelerator_state, "num_processes", None)
    if world_size is None:
        world_size = getattr(accelerator, "num_processes", 1)
    world_size = max(int(world_size), 1)
    total_batch_size = args.per_device_batch_size * world_size * args.gradient_accumulation
    steps_per_epoch = max(math.ceil(len(train_dataset) / total_batch_size), 1)
    total_steps = steps_per_epoch * epochs

    trainer = FineTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda e: compute_metrics(e, model=model),
    )

    if args.show_retrieval_preview:
        trainer.add_callback(
            RetrievalPreviewCallback(
                accelerator=accelerator,
                split_name="eval",
                examples=eval_examples,
                retriever=retriever,
                query_vectors=query_vectors,
                docs_lookup=docs_lookup,
                top_k=args.retriever_top_k,
            )
        )

    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        trainer.optimizer,
        trainer.get_train_dataloader(),
        trainer.get_eval_dataloader(),
    )

    trainer.model = model
    trainer.optimizer = optimizer
    trainer._train_dataloader = train_dataloader
    trainer._eval_dataloader = eval_dataloader

    checkpoint = None
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        os.makedirs(cycle_tmp_dir, exist_ok=True)
        with open(config_path, "w") as json_file:
            json.dump(vars(args), json_file, indent=4)

    trainer.add_callback(
        PeriodicModelSaver(
            accelerator=accelerator,
            save_interval=args.save_every_steps,
            target_dir=final_model_dir,
            log_path=save_log_path,
            config_path=config_path,
        )
    )

    trainer.train(resume_from_checkpoint=checkpoint)

    overwrite_and_log_model(
        accelerator=accelerator,
        model=model,
        target_dir=final_model_dir,
        log_path=save_log_path,
        step=trainer.state.global_step,
        event_type="final",
        config_path=config_path,
    )

    if accelerator.is_main_process:
        print(f"Saved fine-tuned model to: {final_model_dir}")
        if os.path.exists(cycle_tmp_dir):
            shutil.rmtree(cycle_tmp_dir)

    accelerator.wait_for_everyone()


def main():
    args = get_fine_tuning_args()

    model_source = args.checkpoint_path if args.checkpoint_path else args.model_name_or_path
    if model_source is None:
        raise ValueError("Either --model_name_or_path or --checkpoint_path must be provided.")

    initial_cfg = build_cycle_config(model_source, args)
    compression_rates = (
        list(initial_cfg.compr_rates) if initial_cfg.compr_rates is not None else []
    )

    compressor_model_name = initial_cfg.compr_model_name or "decoder"
    decoder_model_name = initial_cfg.decoder_model_name
    lora_flag = args.lora.lower() == "true"

    folder_name = f"{Hasher.hash(str(args))}"
    tmp_output_dir = os.path.join(args.experiment_folder, "finetune_tmp")
    final_model_dir = os.path.join(args.experiment_folder, "fine_tuned_model")
    save_log_path = os.path.join(args.experiment_folder, "save_log.jsonl")

    accelerator = Accelerator()

    if accelerator.is_main_process:
        run_name = (
            f"{compressor_model_name}_{decoder_model_name}_{compression_rates}_QA_"
            f"{lora_flag}_{args.lr}_{folder_name}"
        )
        wandb.init(project="COCOM QA Finetune", name=run_name)
        os.makedirs(args.experiment_folder, exist_ok=True)
        if os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)
        if os.path.exists(final_model_dir):
            checkpoint_path = getattr(args, "checkpoint_path", None)
            if not checkpoint_path or (
                os.path.abspath(checkpoint_path) != os.path.abspath(final_model_dir)
            ):
                shutil.rmtree(final_model_dir)
        if os.path.exists(save_log_path):
            os.remove(save_log_path)
        print(f"Experiment folder: {args.experiment_folder}")
        print(f"Final model path: {final_model_dir}")
        print(f"Checkpoint log path: {save_log_path}")

    accelerator.wait_for_everyone()

    offsets = compute_cycle_offsets(args)
    total_cycles = len(offsets) if offsets else 1

    current_model_source = model_source
    for cycle_index in range(total_cycles):
        offset = offsets[cycle_index] if offsets else None
        if offset is not None:
            if accelerator.is_main_process:
                run_compress_command(args, offset, cycle_index, total_cycles)
            accelerator.wait_for_everyone()

        cfg = build_cycle_config(current_model_source, args)
        run_training_cycle(
            accelerator=accelerator,
            args=args,
            cfg=cfg,
            model_source=current_model_source,
            cycle_index=cycle_index,
            total_cycles=total_cycles,
            base_tmp_output_dir=tmp_output_dir,
            final_model_dir=final_model_dir,
            save_log_path=save_log_path,
            offset=offset,
        )

        current_model_source = final_model_dir

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir)
        print(f"Training complete. Latest model available at: {final_model_dir}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
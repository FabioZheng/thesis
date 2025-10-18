import argparse
import json
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import torch
import numpy as np

from datasets import load_dataset
from analyse.retrieval import TextEmbedder
from modeling_cocom import COCOM
from train_cmab import load_model_safely
from cmab_agent import CompressionBanditAgent
from metrics import batch_entropy
from utils import pad_tokens_to_rate
from save_json import load_and_flatten, save_json, save_queries_json, save_answers_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compressed contexts and embeddings for MS MARCO passages"
    )
    parser.add_argument(
        "--dataset",
        help=(
            "Path to MS MARCO dataset file (JSON/JSONL). If omitted or when"
            " --use-hf-dataset is specified the dataset will be streamed from"
            " the Hugging Face Hub."
        ),
        default=None,
    )
    parser.add_argument(
        "--use-hf-dataset",
        action="store_true",
        help=(
            "Download the MS MARCO dataset split from the Hugging Face Hub"
            " using streaming mode instead of reading a local file."
        ),
    )
    parser.add_argument(
        "--hf-dataset-name",
        default="ms_marco",
        help="Hugging Face dataset identifier to stream (default: 'ms_marco')",
    )
    parser.add_argument(
        "--hf-dataset-config",
        default="v1.1",
        help="Optional Hugging Face dataset configuration (default: 'v1.1')",
    )
    parser.add_argument(
        "--hf-dataset-split",
        default="train",
        help="Dataset split to stream from Hugging Face (default: 'train')",
    )
    parser.add_argument(
        "--hf-offset",
        type=int,
        default=0,
        help=(
            "Number of records to skip from the streamed Hugging Face dataset"
            " before processing."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained COCOM checkpoint directory for context generation",
    )
    parser.add_argument(
        "--context_batch_size",
        type=int,
        default=8,
        help="Batch size for context generation",
    )
    parser.add_argument(
        "--hf_model_name",
        default=None,
        help="Hugging Face model id to load instead of a local checkpoint",
    )
    parser.add_argument("--limit", type=int, help="Max number of rows", default=None)
    parser.add_argument("--compression_rate", type=int, help="Fallback rate", default=4)
    parser.add_argument("--docs_out", help="Directory to save flattened documents", default="data")
    parser.add_argument("--contexts_out", help="Directory to save compressed contexts", default="data/contexts")
    parser.add_argument("--embeddings_out", help="Directory to save document embeddings", default="data/embeddings")
    parser.add_argument(
        "--query-embeddings-out",
        help="Directory to save query embeddings",
        default="data",
    )
    parser.add_argument(
        "--use-bandit",
        dest="use_bandit",
        action="store_true",
        help="Enable the bandit agent for dynamic compression rates (default)",
    )
    parser.add_argument(
        "--no-use-bandit",
        dest="use_bandit",
        action="store_false",
        help="Disable the bandit agent and use a fixed compression rate",
    )
    parser.set_defaults(use_bandit=True)
    parser.add_argument(
        "--bandit-agent",
        default="bandit_ckpt/bandit_agent.pkl",
        help="Path to the trained bandit agent pickle (e.g., bandit_ckpt/bandit_agent.pkl)",
    )
    parser.add_argument(
        "--embedder-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name for document embeddings",
    )
    parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=256,
        help="Batch size for the TextEmbedder",
    )
    parser.add_argument(
        "--embedder-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for TextEmbedder (e.g., 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--no-embedder-normalize",
        action="store_true",
        help="Disable embedding normalization in TextEmbedder",
    )
    return parser.parse_args()

def load_bandit_agent(path: str, rates: List[int]) -> CompressionBanditAgent:
    with open(path, "rb") as f:
        agent_data = pickle.load(f)

    agent_rates = agent_data.get("rates", rates)
    agent_alpha = agent_data.get("alpha", 1.0)
    agent_use_length = agent_data.get("use_length_feature", False)
    agent = CompressionBanditAgent(
        agent_rates,
        alpha=agent_alpha,
        use_length_feature=agent_use_length,
    )

    # Restore learned parameters if available
    if "A" in agent_data:
        agent.A = agent_data["A"]
    if "b" in agent_data:
        agent.b = agent_data["b"]

    return agent


from tqdm import tqdm


def resolve_base_compression_rate(model: COCOM, fallback_rate: int) -> int:
    """Determine the compression rate to use when no bandit agent is present."""

    compr_rates = getattr(model, "compr_rates", None)
    if compr_rates:
        # Normalise to integers if possible
        try:
            normalised_rates = [int(rate) for rate in compr_rates]
        except Exception:
            normalised_rates = list(compr_rates)

        if len(normalised_rates) == 1:
            return normalised_rates[0]

        if fallback_rate in normalised_rates:
            return fallback_rate

        current_rate = getattr(model, "current_rate", None)
        if current_rate is not None:
            try:
                current_rate_int = int(current_rate)
            except Exception:
                current_rate_int = current_rate
            if current_rate_int in normalised_rates:
                return current_rate_int

        return normalised_rates[0]

    current_rate = getattr(model, "current_rate", None)
    if current_rate is not None:
        try:
            return int(current_rate)
        except Exception:
            return current_rate

    return fallback_rate


def generate_contexts(
        docs: Dict[int, Dict[str, str]],
        model: COCOM,
        fallback_rate: int,
        batch_size: Optional[int] = None,
) -> Dict[int, Dict[str, Any]]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if batch_size is None:
        batch_size = getattr(model, "context_batch_size", None)
    if batch_size is None or batch_size <= 0:
        batch_size = 8

    contexts: Dict[int, Dict[str, Any]] = {}
    agent = getattr(model, "bandit_agent", None)
    agent_name = agent.__class__.__name__ if agent is not None else "None"

    print(
        f"Generating contexts with agent={agent_name} and base compression rate={fallback_rate}"
    )

    if agent is None and hasattr(model, "current_rate"):
        try:
            model.current_rate = int(fallback_rate)
        except Exception:
            model.current_rate = fallback_rate

    doc_items = list(docs.items())
    pad_token_id = model.compr.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            model.compr.tokenizer.eos_token_id
            if model.compr.tokenizer.eos_token_id is not None
            else 0
        )

    pbar = tqdm(total=len(doc_items), desc="Generating contexts")

    for batch_start in range(0, len(doc_items), batch_size):
        batch_slice = doc_items[batch_start: batch_start + batch_size]
        batch_doc_ids = [doc_id for doc_id, _ in batch_slice]
        batch_queries = [item["query_id"] for _, item in batch_slice]
        texts = [item["text"] for _, item in batch_slice]

        tokens = model.compr.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="longest",
        )

        entropies: Optional[List[float]] = None
        lengths: Optional[List[float]] = None

        if agent is not None:
            entropy_values = batch_entropy(tokens["input_ids"], tokens["attention_mask"])
            entropies = [float(val) for val in entropy_values]
            if agent.use_length_feature:
                length_tensor = tokens["attention_mask"].sum(dim=1).to(dtype=torch.float32)
                lengths = [float(val) for val in length_tensor.tolist()]

        doc_infos: List[Dict[str, Any]] = []
        rate_to_indices: Dict[int, List[int]] = {}

        for idx, doc_id in enumerate(batch_doc_ids):
            entropy_val = entropies[idx] if entropies is not None else None
            length_val = lengths[idx] if lengths is not None else None
            selected_rate = fallback_rate
            if agent is not None:
                try:
                    selected_rate = agent.select_rate(entropy_val, length_val)
                except Exception:
                    selected_rate = fallback_rate

            doc_infos.append(
                {
                    "doc_id": doc_id,
                    "query_id": batch_queries[idx],
                    "rate": selected_rate,
                    "entropy": entropy_val,
                    "length": length_val,
                }
            )
            rate_to_indices.setdefault(selected_rate, []).append(idx)

        for rate, indices in rate_to_indices.items():
            rate_tokens = {
                key: value[indices].contiguous()
                for key, value in tokens.items()
            }
            rate_tokens = pad_tokens_to_rate(rate_tokens, rate, pad_token_id)
            rate_tokens = {k: v.to(device) for k, v in rate_tokens.items()}

            with torch.no_grad():
                batch_emb = model.compr(
                    input_ids=rate_tokens["input_ids"],
                    attention_mask=rate_tokens["attention_mask"],
                    rate=rate,
                )

            for idx_in_rate, context_tensor in zip(indices, batch_emb.detach().cpu()):
                info = doc_infos[idx_in_rate]
                payload: Dict[str, Any] = {
                    "query_id": info["query_id"],
                    "context": context_tensor,
                    "compression_rate": info["rate"],
                }
                if info["entropy"] is not None:
                    payload["entropy"] = info["entropy"]
                if info["length"] is not None:
                    payload["length"] = info["length"]

                contexts[info["doc_id"]] = payload
                postfix = {
                    "rate": info["rate"],
                    "entropy": info["entropy"],
                    "agent": agent_name,
                }
                if info["length"] is not None:
                    postfix["length"] = info["length"]
                pbar.update(1)
                pbar.set_postfix(postfix)

    return contexts


def save_contexts(
    contexts: Dict[int, Dict[str, Any]], directory: str, filename: str = "contexts.h5"
) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, filename)
    metadata_path = os.path.join(directory, os.path.splitext(filename)[0] + "_metadata.json")

    doc_items = sorted(
        contexts.items(),
        key=lambda kv: (
            int(kv[0]) if isinstance(kv[0], (int, np.integer)) else str(kv[0])
        ),
    )

    doc_ids: List[str] = []
    flattened_contexts: List[np.ndarray] = []
    shapes: List[Tuple[int, ...]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    approx_bytes = 0

    for doc_id, payload in doc_items:
        context = payload.get("context")
        if context is None:
            raise ValueError(f"Missing context tensor for document id {doc_id}")

        tensor_context = (
            context.detach().cpu() if isinstance(context, torch.Tensor) else torch.as_tensor(context)
        )
        tensor_context = tensor_context.to(torch.float32).contiguous()

        approx_bytes += tensor_context.element_size() * tensor_context.nelement()

        np_context = tensor_context.numpy().reshape(-1).astype(np.float32, copy=True)
        flattened_contexts.append(np_context)
        shapes.append(tuple(int(dim) for dim in tensor_context.shape))

        doc_id_str = str(doc_id)
        doc_ids.append(doc_id_str)
        metadata_payload: Dict[str, Any] = {}
        for key, value in payload.items():
            if key == "context":
                continue
            if isinstance(value, (np.integer, np.int64, np.int32)):
                metadata_payload[key] = int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                metadata_payload[key] = float(value)
            elif isinstance(value, torch.Tensor):
                metadata_payload[key] = value.detach().cpu().tolist()
            else:
                metadata_payload[key] = value
        metadata[doc_id_str] = metadata_payload

    with h5py.File(path, "w") as h5:
        context_dtype = h5py.vlen_dtype(np.dtype("float32"))
        contexts_ds = h5.create_dataset("contexts", (len(flattened_contexts),), dtype=context_dtype)
        for idx, np_context in enumerate(flattened_contexts):
            contexts_ds[idx] = np_context

        shape_dtype = h5py.vlen_dtype(np.dtype("int32"))
        shapes_ds = h5.create_dataset("shapes", (len(shapes),), dtype=shape_dtype)
        for idx, shape in enumerate(shapes):
            shapes_ds[idx] = np.asarray(shape, dtype=np.int32)

        doc_id_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("doc_ids", (len(doc_ids),), dtype=doc_id_dtype, data=doc_ids)

    with open(metadata_path, "w", encoding="utf-8") as metadata_handle:
        json.dump(metadata, metadata_handle, ensure_ascii=False)

    memory_stats: Dict[str, Any] = {
        "approx_memory_mb": approx_bytes / (1024 * 1024),
        "h5_disk_mb": os.path.getsize(path) / (1024 * 1024),
        "metadata_disk_mb": os.path.getsize(metadata_path) / (1024 * 1024),
        "metadata_path": metadata_path,
    }
    return path, memory_stats


def generate_doc_embeddings(
    docs: Dict[int, Dict[str, str]],
    embedder: TextEmbedder,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    doc_ids: List[int] = sorted(docs.keys())
    texts: List[str] = [docs[doc_id]["text"] for doc_id in doc_ids]
    query_ids: List[Any] = [docs[doc_id]["query_id"] for doc_id in doc_ids]

    embeddings_array = embedder.encode(texts)

    return (
        np.asarray(doc_ids, dtype=np.int64),
        np.asarray([str(qid) for qid in query_ids], dtype=np.str_),
        np.asarray(embeddings_array, dtype=np.float32),
    )


def _sorted_query_items(
    queries: Dict[Any, Dict[str, Any]]
) -> List[Tuple[str, Dict[str, Any]]]:
    def _normalise_key(key: Any) -> Tuple[int, Any]:
        try:
            numeric = int(str(key))
            return (0, numeric)
        except Exception:
            return (1, str(key))

    items: List[Tuple[str, Dict[str, Any]]] = []
    for key, value in queries.items():
        items.append((str(key), value if isinstance(value, dict) else {"text": value}))
    return sorted(items, key=lambda item: _normalise_key(item[0]))


def generate_query_embeddings(
    queries: Dict[Any, Dict[str, Any]],
    embedder: TextEmbedder,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered_items = _sorted_query_items(queries)

    query_ids: List[str] = []
    query_texts: List[str] = []
    for query_id, payload in ordered_items:
        text = ""
        if isinstance(payload, dict):
            text_val = payload.get("text")
            if isinstance(text_val, str):
                text = text_val.strip()
        if not text:
            continue
        query_ids.append(query_id)
        query_texts.append(text)

    if not query_ids:
        raise ValueError("No queries with textual content were provided for embedding.")

    embeddings_array = embedder.encode(query_texts)

    return (
        np.asarray(query_ids, dtype=np.str_),
        np.asarray(query_texts, dtype=np.str_),
        np.asarray(embeddings_array, dtype=np.float32),
    )


def save_embeddings_npz(
    doc_ids: np.ndarray,
    query_ids: np.ndarray,
    embeddings: np.ndarray,
    output_dir: str,
    filename: str = "embeddings.npz",
) -> Tuple[str, Dict[str, Any]]:
    """Persist embeddings alongside their document and query identifiers."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(
        output_path,
        doc_ids=doc_ids,
        query_ids=query_ids,
        embeddings=embeddings,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    memory_stats: Dict[str, Any] = {
        "embeddings_shape": tuple(int(dim) for dim in embeddings.shape),
        "approx_disk_mb": file_size_mb,
    }
    return output_path, memory_stats


def save_query_embeddings_npz(
    query_ids: np.ndarray,
    query_texts: np.ndarray,
    embeddings: np.ndarray,
    output_dir: str,
    filename: str = "query_embeddings.npz",
) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    np.savez_compressed(
        output_path,
        query_ids=query_ids,
        query_texts=query_texts,
        embeddings=embeddings,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    memory_stats: Dict[str, Any] = {
        "embeddings_shape": tuple(int(dim) for dim in embeddings.shape),
        "approx_disk_mb": file_size_mb,
    }
    return output_path, memory_stats


def _create_dataset_builder(
    args: argparse.Namespace,
) -> Tuple[Callable[[], Any], Optional[int], str]:
    """Prepare a factory that yields the dataset source for processing."""

    use_hf_dataset = args.use_hf_dataset or not args.dataset

    if use_hf_dataset:
        if args.hf_offset < 0:
            raise ValueError("--hf-offset must be non-negative")

        def _builder() -> Any:
            load_kwargs = {"streaming": True}
            dataset = load_dataset(
                args.hf_dataset_name,
                name=args.hf_dataset_config,
                split=args.hf_dataset_split,
                **load_kwargs,
            )
            dataset = dataset.with_format("python")
            if args.hf_offset:
                dataset = dataset.skip(args.hf_offset)
            if args.limit is not None:
                dataset = dataset.take(args.limit)
            return dataset

        dataset_descriptor = "hf://" + args.hf_dataset_name
        if args.hf_dataset_config:
            dataset_descriptor += f"/{args.hf_dataset_config}"
        dataset_descriptor += f":{args.hf_dataset_split}"
        dataset_descriptor += f" (offset={args.hf_offset}, limit={args.limit})"

        return _builder, None, dataset_descriptor

    if not args.dataset:
        raise ValueError(
            "Either provide --dataset or enable --use-hf-dataset to stream"
            " from the Hugging Face Hub."
        )

    def _builder() -> Any:
        return args.dataset

    return _builder, args.limit, args.dataset


def main() -> None:
    args = parse_args()

    dataset_builder, docs_limit, dataset_descriptor = _create_dataset_builder(args)

    print(f"Loading dataset source: {dataset_descriptor}")

    docs_source = dataset_builder()
    docs = load_and_flatten(docs_source, limit=docs_limit)
    docs = {
        doc_id: {k: v for k, v in payload.items() if k in ("query_id", "text")}
        for doc_id, payload in docs.items()
    }
    docs_path, docs_mem = save_json(docs, args.docs_out, "docs.json")
    print(
        "Extracted {count} MS MARCO passages -> {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=len(docs),
            path=docs_path,
            mem=docs_mem.get("approx_memory_mb", 0.0),
            pickle=docs_mem.get("pickle_disk_mb", 0.0),
            json=docs_mem.get("json_disk_mb", 0.0),
        )
    )

    query_ids = {
        payload["query_id"]
        for payload in docs.values()
        if payload.get("query_id") is not None
    }
    queries_source = dataset_builder()
    queries_path, queries_mem = save_queries_json(
        queries_source,
        args.docs_out,
        "queries.json",
        query_ids=query_ids or None,
    )
    print(
        "Extracted {count} queries -> {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=queries_mem.get("count", 0),
            path=queries_path,
            mem=queries_mem.get("approx_memory_mb", 0.0),
            pickle=queries_mem.get("pickle_disk_mb", 0.0),
            json=queries_mem.get("json_disk_mb", 0.0),
        )
    )

    answers_source = dataset_builder()
    answers_path, answers_mem = save_answers_json(
        answers_source,
        args.docs_out,
        "answers.json",
        query_ids=query_ids or None,
    )
    print(
        "Extracted {count} answers -> {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=answers_mem.get("count", 0),
            path=answers_path,
            mem=answers_mem.get("approx_memory_mb", 0.0),
            pickle=answers_mem.get("pickle_disk_mb", 0.0),
            json=answers_mem.get("json_disk_mb", 0.0),
        )
    )

    if args.checkpoint and args.hf_model_name:
        raise ValueError("Specify either --checkpoint or --hf_model_name, not both")

    model_source = args.checkpoint or args.hf_model_name
    if model_source is None:
        raise ValueError("You must provide either --checkpoint or --hf_model_name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_safely(model_source)
    model.to(device)
    model.eval()
    print(device)

    agent = None
    agent_name = "None"
    if args.use_bandit and args.bandit_agent:
        bandit_path = args.bandit_agent
        if os.path.isdir(bandit_path):
            bandit_path = os.path.join(bandit_path, "bandit_agent.pkl")
        if os.path.exists(bandit_path):
            agent = load_bandit_agent(bandit_path, getattr(model, "compr_rates", []))
            model.set_bandit_agent(agent)
            agent_name = agent.__class__.__name__
            print(f"Loaded bandit agent from {bandit_path}")
        else:
            print(
                f"Bandit agent not found at {bandit_path}; proceeding with fixed compression rate."
            )
    elif not args.use_bandit:
        print("Bandit agent disabled via --no-use-bandit flag; using fixed compression rate.")
    if agent is None:
        print("No bandit agent active; using a single compression rate during generation.")

    base_rate = resolve_base_compression_rate(model, args.compression_rate)
    print(f"Compression configuration -> agent={agent_name}, base_rate={base_rate}")

    contexts = generate_contexts(docs, model, base_rate, args.context_batch_size)
    contexts_path, ctx_mem = save_contexts(contexts, args.contexts_out, "contexts.h5")
    print(
        "Generated contexts for {count} MS MARCO passages using model {model_src} -> {path} "
        "(approx memory: {mem:.2f} MB, h5: {h5:.2f} MB, metadata: {meta:.2f} MB)".format(
            count=len(contexts),
            model_src=model_source,
            path=contexts_path,
            mem=ctx_mem.get("approx_memory_mb", 0.0),
            h5=ctx_mem.get("h5_disk_mb", 0.0),
            meta=ctx_mem.get("metadata_disk_mb", 0.0),
        )
    )
    print(
        "Context metadata saved to {metadata_path}".format(
            metadata_path=ctx_mem.get("metadata_path")
        )
    )

    embedder = TextEmbedder(
        model_name=args.embedder_model,
        batch_size=args.embedder_batch_size,
        device=args.embedder_device,
        normalize=not args.no_embedder_normalize,
    )

    doc_ids, query_ids, embeddings = generate_doc_embeddings(docs, embedder)
    emb_path, emb_mem = save_embeddings_npz(
        doc_ids,
        query_ids,
        embeddings,
        args.embeddings_out,
    )
    print(
        "Generated embeddings for {count} MS MARCO passages -> {path} "
        "(shape: {shape}, approx disk: {disk:.2f} MB)".format(
            count=len(doc_ids),
            path=emb_path,
            shape=emb_mem.get("embeddings_shape"),
            disk=emb_mem.get("approx_disk_mb", 0.0),
        )
    )

    with open(queries_path, "r", encoding="utf-8") as handle:
        queries_payload = json.load(handle)

    query_ids_array, query_texts_array, query_embeddings = generate_query_embeddings(
        queries_payload,
        embedder,
    )

    query_embeddings_dir = (
        args.query_embeddings_out if args.query_embeddings_out is not None else args.embeddings_out
    )

    query_emb_path, query_emb_mem = save_query_embeddings_npz(
        query_ids_array,
        query_texts_array,
        query_embeddings,
        query_embeddings_dir,
    )
    print(
        "Generated embeddings for {count} queries -> {path} "
        "(shape: {shape}, approx disk: {disk:.2f} MB)".format(
            count=len(query_ids_array),
            path=query_emb_path,
            shape=query_emb_mem.get("embeddings_shape"),
            disk=query_emb_mem.get("approx_disk_mb", 0.0),
        )
    )


if __name__ == "__main__":
    main()

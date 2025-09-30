import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import h5py
import numpy as np
import torch

from analyse.retrieval import TextEmbedder
from modeling_cocom import COCOM
from train_cmab import load_model_safely
from cmab_agent import CompressionBanditAgent
from metrics import batch_entropy
from utils import pad_tokens_to_rate
from save_json import load_and_flatten, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compressed contexts and embeddings for MS MARCO passages"
    )
    parser.add_argument("--dataset", help="Path to MS MARCO dataset file (JSON/JSONL)", default="ms_marco_train.json")
    parser.add_argument(
        "--checkpoint",
        help="Path to a trained COCOM checkpoint directory for context generation",
    )
    parser.add_argument("--limit", type=int, help="Max number of rows", default=None)
    parser.add_argument("--compression_rate", type=int, help="Fallback rate", default=4)
    parser.add_argument("--docs_out", help="Directory to save flattened documents", default="data")
    parser.add_argument("--contexts_out", help="Directory to save compressed contexts", default="data/contexts")
    parser.add_argument("--embeddings_out", help="Directory to save document embeddings", default="data/embeddings")
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
    agent = CompressionBanditAgent(agent_rates, alpha=agent_alpha)

    # Restore learned parameters if available
    if "A" in agent_data:
        agent.A = agent_data["A"]
    if "b" in agent_data:
        agent.b = agent_data["b"]

    return agent


from tqdm import tqdm


def generate_contexts(
    docs: Dict[int, Dict[str, str]],
    model: COCOM,
    fallback_rate: int,
    output_path: str,
) -> Dict[str, Any]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    doc_items: List[Any] = list(docs.items())
    doc_ids = np.array([doc_id for doc_id, _ in doc_items], dtype=np.int64)
    agent = getattr(model, "bandit_agent", None)

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    context_dtype = h5py.vlen_dtype(np.float32)
    shape_dtype = h5py.vlen_dtype(np.int32)

    with h5py.File(output_path, "w") as h5f:
        h5f.create_dataset("doc_id", data=doc_ids)
        query_ds = h5f.create_dataset(
            "query_id", shape=(len(doc_items),), dtype=string_dtype
        )
        compression_ds = h5f.create_dataset(
            "compression_rate", shape=(len(doc_items),), dtype=np.int32
        )
        entropy_ds = h5f.create_dataset(
            "entropy", shape=(len(doc_items),), dtype=np.float32
        )
        if len(entropy_ds) > 0:
            entropy_ds[...] = np.nan
        contexts_ds = h5f.create_dataset(
            "context", shape=(len(doc_items),), dtype=context_dtype
        )
        shapes_ds = h5f.create_dataset(
            "context_shape", shape=(len(doc_items),), dtype=shape_dtype
        )

        pbar = tqdm(doc_items, total=len(doc_items), desc="Generating contexts")
        for idx, (doc_id, item) in enumerate(pbar):
            tokens = model.compr.tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",  # Add padding to fix size issues
            )
            selected_rate = fallback_rate
            entropy: Optional[float] = None
            if agent is not None:
                entropy = batch_entropy(tokens["input_ids"], tokens["attention_mask"])[0]
                try:
                    selected_rate = agent.select_rate(float(entropy))
                except Exception:
                    selected_rate = fallback_rate

            pad_token_id = model.compr.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = (
                    model.compr.tokenizer.eos_token_id
                    if model.compr.tokenizer.eos_token_id is not None
                    else 0
                )
            tokens = pad_tokens_to_rate(tokens, selected_rate, pad_token_id)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                emb = model.compr(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    rate=selected_rate,
                )

            context_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
            contexts_ds[idx] = context_np.ravel()
            shapes_ds[idx] = np.array(context_np.shape, dtype=np.int32)
            query_ds[idx] = str(item["query_id"])
            compression_ds[idx] = selected_rate
            if entropy is not None:
                entropy_ds[idx] = float(entropy)

            # Update progress description
            pbar.set_postfix({"rate": selected_rate, "entropy": entropy})

        h5f.flush()

    context_size_mb = (
        os.path.getsize(output_path) / (1024 ** 2) if os.path.exists(output_path) else 0.0
    )

    return {
        "path": output_path,
        "count": len(doc_items),
        "file_size_mb": context_size_mb,
    }


def generate_embeddings(
    docs: Dict[int, Dict[str, str]],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
    index_path: str,
    metadata_path: str,
) -> Dict[str, Any]:
    embedder = TextEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
    )

    doc_ids: np.ndarray = np.array(sorted(docs.keys()), dtype=np.int64)
    texts: List[str] = [docs[int(doc_id)]["text"] for doc_id in doc_ids]
    embeddings_array = np.asarray(embedder.encode(texts), dtype=np.float32)

    if embeddings_array.size == 0:
        raise ValueError("No documents provided for embedding generation")

    dimension = embeddings_array.shape[1]
    if normalize:
        base_index = faiss.IndexFlatIP(dimension)
    else:
        base_index = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings_array, doc_ids)

    index_dir = os.path.dirname(index_path) or "."
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, index_path)

    query_ids = [str(docs[int(doc_id)]["query_id"]) for doc_id in doc_ids]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(metadata_path, "w") as meta_h5:
        meta_h5.create_dataset("doc_id", data=doc_ids)
        meta_h5.create_dataset(
            "query_id",
            data=np.array(query_ids, dtype=object),
            dtype=string_dtype,
        )
        meta_h5.create_dataset("embedding_dim", data=np.array([dimension], dtype=np.int32))
        meta_h5.flush()

    index_size_mb = (
        os.path.getsize(index_path) / (1024 ** 2) if os.path.exists(index_path) else 0.0
    )
    metadata_size_mb = (
        os.path.getsize(metadata_path) / (1024 ** 2)
        if os.path.exists(metadata_path)
        else 0.0
    )

    return {
        "index_path": index_path,
        "metadata_path": metadata_path,
        "count": len(doc_ids),
        "index_size_mb": index_size_mb,
        "metadata_size_mb": metadata_size_mb,
    }


def main() -> None:
    args = parse_args()

    docs = load_and_flatten(args.dataset, limit=args.limit)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_safely(args.checkpoint)
    model.to(device)
    model.eval()
    print(device)

    if args.bandit_agent:
        bandit_path = args.bandit_agent
        if os.path.isdir(bandit_path):
            bandit_path = os.path.join(bandit_path, "bandit_agent.pkl")
        if not os.path.exists(bandit_path):
            raise FileNotFoundError(
                f"Bandit agent not found at {bandit_path}. Provide a valid path to bandit_agent.pkl"
            )
        agent = load_bandit_agent(bandit_path, getattr(model, "compr_rates", []))
        model.set_bandit_agent(agent)
        print(f"Loaded bandit agent from {bandit_path}")
    else:
        print("No bandit agent provided; defaulting to fallback compression rate")

    contexts_output_path = os.path.join(args.contexts_out, "contexts.h5")
    contexts_info = generate_contexts(
        docs,
        model,
        args.compression_rate,
        contexts_output_path,
    )
    print(
        "Generated contexts for {count} MS MARCO passages using checkpoint {ckpt} -> {path} "
        "(file size: {size:.2f} MB)".format(
            count=contexts_info["count"],
            ckpt=args.checkpoint,
            path=contexts_info["path"],
            size=contexts_info["file_size_mb"],
        )
    )

    embeddings_index_path = os.path.join(args.embeddings_out, "embeddings.faiss")
    embeddings_metadata_path = os.path.join(
        args.embeddings_out, "embeddings_meta.h5"
    )
    embeddings_info = generate_embeddings(
        docs,
        args.embedder_model,
        args.embedder_batch_size,
        args.embedder_device,
        not args.no_embedder_normalize,
        embeddings_index_path,
        embeddings_metadata_path,
    )
    print(
        "Generated embeddings for {count} MS MARCO passages -> {index_path} (index size: {idx_size:.2f} MB, metadata: {meta_path}, metadata size: {meta_size:.2f} MB)".format(
            count=embeddings_info["count"],
            index_path=embeddings_info["index_path"],
            idx_size=embeddings_info["index_size_mb"],
            meta_path=embeddings_info["metadata_path"],
            meta_size=embeddings_info["metadata_size_mb"],
        )
    )


if __name__ == "__main__":
    main()

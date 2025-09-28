import argparse
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from analyse.retrieval import TextEmbedder
from modeling_cocom import COCOM
from train_cmab import load_model_safely
from cmab_agent import CompressionBanditAgent
from metrics import batch_entropy
from utils import pad_tokens_to_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compressed contexts and embeddings for MS MARCO passages"
    )
    parser.add_argument("--dataset", help="Path to MS MARCO dataset file (JSON/JSONL)", default="ms_marco_train.json")
    parser.add_argument(
        "--checkpoint",
        help="Path to a trained COCOM checkpoint directory for context generation",
    )
    parser.add_argument(
        "--compression_rate", type=int, help="Fallback rate", default=4
    )
    parser.add_argument(
        "--compression-batch-size",
        type=int,
        default=8,
        help="Number of documents processed concurrently when generating contexts",
    )
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


def load_and_flatten(dataset_path: str) -> Dict[int, Dict[str, str]]:
    df = pd.read_json(dataset_path, lines=True)

    docs: Dict[int, Dict[str, str]] = {}
    doc_id = 0
    for _, row in df.iterrows():
        query_id = row.get("query_id")
        passages_field = row.get("passages", {})
        passage_texts: List[str] = []
        if isinstance(passages_field, dict):
            if "passages" in passages_field:
                passage_texts = passages_field.get("passages", [])
            else:
                passage_texts = passages_field.get("passage_text", [])
        for passage in passage_texts:
            if passage is None:
                continue
            text = passage if isinstance(passage, str) else str(passage)
            docs[doc_id] = {"query_id": query_id, "text": text}
            doc_id += 1
    return docs


def _estimate_memory_usage(obj: Any) -> Dict[str, float]:
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
    except Exception:
        pickle_bytes = 0

    return {
        "approx_memory_mb": approx_bytes / (1024 * 1024),
        "pickle_disk_mb": pickle_bytes / (1024 * 1024),
    }


def save_json(data: Dict, directory: str, filename: str) -> tuple[str, Dict[str, float]]:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    mem_stats = _estimate_memory_usage(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    mem_stats["json_disk_mb"] = os.path.getsize(path) / (1024 * 1024)
    return path, mem_stats


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
    batch_size: int = 8,
) -> Dict[int, Dict[str, Any]]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contexts: Dict[int, Dict[str, Any]] = {}
    agent = getattr(model, "bandit_agent", None)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    doc_items = list(docs.items())
    pad_token_id = model.compr.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            model.compr.tokenizer.eos_token_id
            if model.compr.tokenizer.eos_token_id is not None
            else 0
        )

    pbar = tqdm(total=len(doc_items), desc="Generating contexts")

    for start in range(0, len(doc_items), batch_size):
        batch = doc_items[start : start + batch_size]
        texts = [item["text"] for _, item in batch]

        token_batch_encoding = model.compr.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        token_batch = {k: v for k, v in token_batch_encoding.items()}

        entropies: List[Optional[float]] = [None] * len(batch)
        if agent is not None:
            try:
                entropy_values = batch_entropy(
                    token_batch["input_ids"], token_batch["attention_mask"]
                )
                entropies = [float(ent) for ent in entropy_values]
            except Exception:
                entropies = [None] * len(batch)

        selected_rates: List[int] = []
        for entropy in entropies:
            rate = fallback_rate
            if agent is not None and entropy is not None:
                try:
                    rate = agent.select_rate(entropy)
                except Exception:
                    rate = fallback_rate
            selected_rates.append(rate)

        rate_to_indices: Dict[int, List[int]] = {}
        for idx, rate in enumerate(selected_rates):
            rate_to_indices.setdefault(rate, []).append(idx)

        for rate, indices in rate_to_indices.items():
            rate_tokens = {k: v[indices] for k, v in token_batch.items()}
            rate_tokens = pad_tokens_to_rate(rate_tokens, rate, pad_token_id)
            rate_tokens = {k: v.to(device) for k, v in rate_tokens.items()}

            with torch.no_grad():
                emb = model.compr(
                    input_ids=rate_tokens["input_ids"],
                    attention_mask=rate_tokens["attention_mask"],
                    rate=rate,
                )

            for output_idx, batch_idx in enumerate(indices):
                doc_id, item = batch[batch_idx]
                contexts[doc_id] = {
                    "query_id": item["query_id"],
                    "context": emb[output_idx].cpu().tolist(),
                    "compression_rate": rate,
                }
                entropy = entropies[batch_idx]
                if entropy is not None:
                    contexts[doc_id]["entropy"] = entropy

        if entropies:
            last_entropy = next((ent for ent in reversed(entropies) if ent is not None), None)
        else:
            last_entropy = None
        pbar.update(len(batch))
        if selected_rates:
            pbar.set_postfix({"rate": selected_rates[-1], "entropy": last_entropy})

    pbar.close()

    return contexts


def generate_embeddings(
    docs: Dict[int, Dict[str, str]],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
) -> Dict[int, Dict[str, Any]]:
    embedder = TextEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
    )

    doc_ids: List[int] = sorted(docs.keys())
    texts: List[str] = [docs[doc_id]["text"] for doc_id in doc_ids]
    embeddings_array = embedder.encode(texts)

    embeddings: Dict[int, Dict[str, Any]] = {}
    for idx, doc_id in enumerate(doc_ids):
        embeddings[doc_id] = {
            "query_id": docs[doc_id]["query_id"],
            "embedding": embeddings_array[idx].tolist(),
        }
    return embeddings


def main() -> None:
    args = parse_args()

    docs = load_and_flatten(args.dataset)
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

    contexts = generate_contexts(
        docs, model, args.compression_rate, batch_size=args.compression_batch_size
    )
    contexts_path, ctx_mem = save_json(contexts, args.contexts_out, "contexts.json")
    print(
        "Generated contexts for {count} MS MARCO passages using checkpoint {ckpt} -> {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=len(contexts),
            ckpt=args.checkpoint,
            path=contexts_path,
            mem=ctx_mem.get("approx_memory_mb", 0.0),
            pickle=ctx_mem.get("pickle_disk_mb", 0.0),
            json=ctx_mem.get("json_disk_mb", 0.0),
        )
    )

    embeddings = generate_embeddings(
        docs,
        args.embedder_model,
        args.embedder_batch_size,
        args.embedder_device,
        not args.no_embedder_normalize,
    )
    emb_path, emb_mem = save_json(embeddings, args.embeddings_out, "embeddings.json")
    print(
        "Generated embeddings for {count} MS MARCO passages -> {path} "
        "(approx memory: {mem:.2f} MB, pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=len(embeddings),
            path=emb_path,
            mem=emb_mem.get("approx_memory_mb", 0.0),
            pickle=emb_mem.get("pickle_disk_mb", 0.0),
            json=emb_mem.get("json_disk_mb", 0.0),
        )
    )


if __name__ == "__main__":
    main()
import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np

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
    parser.add_argument("--dataset", help="Path to MS MARCO dataset file (JSON/JSONL)", default="ms_marco_train.json")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a trained COCOM checkpoint directory for context generation",
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


def generate_contexts(
        docs: Dict[int, Dict[str, str]], model: COCOM, fallback_rate: int
) -> Dict[int, Dict[str, Any]]:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contexts: Dict[int, Dict[str, Any]] = {}
    agent = getattr(model, "bandit_agent", None)

    # Add progress bar
    pbar = tqdm(docs.items(), total=len(docs), desc="Generating contexts")

    for doc_id, item in pbar:
        tokens = model.compr.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",  # Add padding to fix size issues
        )
        selected_rate = fallback_rate
        entropy: Optional[float] = None
        length: Optional[float] = None
        if agent is not None:
            entropy = batch_entropy(tokens["input_ids"], tokens["attention_mask"])[0]
            if agent.use_length_feature:
                length = float(tokens["attention_mask"].sum().item())
            try:
                selected_rate = agent.select_rate(float(entropy), length)
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
        context_tensor = emb.detach().cpu()
        contexts[doc_id] = {
            "query_id": item["query_id"],
            "context": context_tensor,
            "compression_rate": selected_rate,
        }
        if entropy is not None:
            contexts[doc_id]["entropy"] = entropy
        if length is not None:
            contexts[doc_id]["length"] = length

        # Update progress description
        postfix = {"rate": selected_rate, "entropy": entropy}
        if length is not None:
            postfix["length"] = length
        pbar.set_postfix(postfix)

    return contexts


def _estimate_context_memory_mb(contexts: Dict[int, Dict[str, Any]]) -> float:
    total_bytes = 0
    for item in contexts.values():
        context = item.get("context")
        if isinstance(context, torch.Tensor):
            total_bytes += context.element_size() * context.nelement()
        elif context is not None:
            tensor = torch.as_tensor(context)
            total_bytes += tensor.element_size() * tensor.nelement()
    return total_bytes / (1024 * 1024)


def save_contexts(
    contexts: Dict[int, Dict[str, Any]], directory: str, filename: str = "contexts.pt"
) -> Tuple[str, Dict[str, Any]]:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    serialisable_contexts: Dict[int, Dict[str, Any]] = {}
    for doc_id, payload in contexts.items():
        context = payload.get("context")
        if context is None:
            raise ValueError(f"Missing context tensor for document id {doc_id}")
        tensor_context = context.cpu() if isinstance(context, torch.Tensor) else torch.as_tensor(context)
        serialisable_contexts[doc_id] = {
            **payload,
            "context": tensor_context,
        }

    torch.save(serialisable_contexts, path)

    memory_stats: Dict[str, Any] = {
        "approx_memory_mb": _estimate_context_memory_mb(serialisable_contexts),
        "pt_disk_mb": os.path.getsize(path) / (1024 * 1024),
    }
    return path, memory_stats


def generate_embeddings(
    docs: Dict[int, Dict[str, str]],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    embedder = TextEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
    )

    doc_ids: List[int] = sorted(docs.keys())
    texts: List[str] = [docs[doc_id]["text"] for doc_id in doc_ids]
    query_ids: List[Any] = [docs[doc_id]["query_id"] for doc_id in doc_ids]

    embeddings_array = embedder.encode(texts)

    return (
        np.asarray(doc_ids, dtype=np.int64),
        np.asarray([str(qid) for qid in query_ids], dtype=np.str_),
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

    query_ids = {
        payload["query_id"]
        for payload in docs.values()
        if payload.get("query_id") is not None
    }
    queries_path, queries_mem = save_queries_json(
        args.dataset,
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

    answers_path, answers_mem = save_answers_json(
        args.dataset,
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

    contexts = generate_contexts(docs, model, args.compression_rate)
    contexts_path, ctx_mem = save_contexts(contexts, args.contexts_out, "contexts.pt")
    print(
        "Generated contexts for {count} MS MARCO passages using model {model_src} -> {path} "
        "(approx memory: {mem:.2f} MB, disk: {disk:.2f} MB)".format(
            count=len(contexts),
            model_src=model_source,
            path=contexts_path,
            mem=ctx_mem.get("approx_memory_mb", 0.0),
            disk=ctx_mem.get("pt_disk_mb", 0.0),
        )
    )

    doc_ids, query_ids, embeddings = generate_embeddings(
        docs,
        args.embedder_model,
        args.embedder_batch_size,
        args.embedder_device,
        not args.no_embedder_normalize,
    )
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


if __name__ == "__main__":
    main()
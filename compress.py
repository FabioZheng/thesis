import argparse
import os
import pickle
from typing import Any, Dict, List, Optional

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
        contexts[doc_id] = {
            "query_id": item["query_id"],
            "context": emb.cpu().tolist(),
            "compression_rate": selected_rate,
        }
        if entropy is not None:
            contexts[doc_id]["entropy"] = entropy

        # Update progress description
        pbar.set_postfix({"rate": selected_rate, "entropy": entropy})

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

    contexts = generate_contexts(docs, model, args.compression_rate)
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
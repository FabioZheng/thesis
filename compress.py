import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np

from analyse.retrieval import TextEmbedder
from modeling_cocom import COCOM
from train_cmab import load_model_safely
from cmab_agent import CompressionBanditAgent
from metrics import batch_entropy
from utils import pad_tokens_to_rate
from save_json import load_and_flatten, save_answers_json, save_json, save_queries_json


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

def load_bandit_agent(path: str, rates: Iterable[int]) -> CompressionBanditAgent:
    with open(path, "rb") as f:
        agent_data = pickle.load(f)

    agent = CompressionBanditAgent(
        list(agent_data.get("rates", list(rates))),
        alpha=agent_data.get("alpha", 1.0),
        use_length_feature=agent_data.get("use_length_feature", False),
    )

    agent.A = agent_data.get("A", agent.A)
    agent.b = agent_data.get("b", agent.b)
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
    docs: Dict[int, Dict[str, str]], model: COCOM, fallback_rate: int
) -> Dict[int, Dict[str, Any]]:
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = model.compr.tokenizer
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id or 0
    )

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

    contexts: Dict[int, Dict[str, Any]] = {}
    pbar = tqdm(docs.items(), total=len(docs), desc="Generating contexts")

    with torch.no_grad():
        for doc_id, item in pbar:
            tokens = tokenizer(
                item["text"],
                return_tensors="pt",
                truncation=True,
            )

            selected_rate = fallback_rate
            entropy: Optional[float] = None
            length: Optional[float] = None

            if agent is not None:
                attention_mask = tokens["attention_mask"]
                entropy = float(
                    batch_entropy(tokens["input_ids"], attention_mask)[0]
                )
                if agent.use_length_feature:
                    length = float(attention_mask.sum().item())
                try:
                    selected_rate = agent.select_rate(entropy, length)
                except Exception:
                    selected_rate = fallback_rate

            tokens = pad_tokens_to_rate(tokens, selected_rate, pad_token_id)
            tokens = {k: v.to(device) for k, v in tokens.items()}

            emb = model.compr(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                rate=selected_rate,
            )
            context_tensor = emb.detach().cpu()

            record: Dict[str, Any] = {
                "query_id": item["query_id"],
                "context": context_tensor,
                "compression_rate": selected_rate,
            }
            if entropy is not None:
                record["entropy"] = entropy
            if length is not None:
                record["length"] = length
            contexts[doc_id] = record

            postfix = {"rate": selected_rate, "agent": agent_name}
            if entropy is not None:
                postfix["entropy"] = entropy
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
    path = Path(directory, filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialisable_contexts: Dict[int, Dict[str, Any]] = {}
    for doc_id, payload in contexts.items():
        context = payload.get("context")
        if context is None:
            raise ValueError(f"Missing context tensor for document id {doc_id}")
        tensor_context = (
            context.detach().cpu()
            if isinstance(context, torch.Tensor)
            else torch.as_tensor(context)
        )
        serialisable_contexts[doc_id] = {**payload, "context": tensor_context}

    torch.save(serialisable_contexts, path)

    memory_stats: Dict[str, Any] = {
        "approx_memory_mb": _estimate_context_memory_mb(serialisable_contexts),
        "pt_disk_mb": path.stat().st_size / (1024 * 1024),
    }
    return str(path), memory_stats


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
    output_path = Path(output_dir, filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        doc_ids=doc_ids,
        query_ids=query_ids,
        embeddings=embeddings,
    )

    memory_stats: Dict[str, Any] = {
        "embeddings_shape": tuple(int(dim) for dim in embeddings.shape),
        "approx_disk_mb": output_path.stat().st_size / (1024 * 1024),
    }
    return str(output_path), memory_stats


def _print_export_stats(
    label: str,
    count: int,
    path: str,
    stats: Dict[str, Any],
) -> None:
    print(
        "Extracted {count} {label} -> {path} (approx memory: {mem:.2f} MB, "
        "pickle: {pickle:.2f} MB, json: {json:.2f} MB)".format(
            count=count,
            label=label,
            path=path,
            mem=stats.get("approx_memory_mb", 0.0),
            pickle=stats.get("pickle_disk_mb", 0.0),
            json=stats.get("json_disk_mb", 0.0),
        )
    )


def prepare_documents(
    dataset_path: str, output_dir: str, limit: Optional[int]
) -> Dict[int, Dict[str, str]]:
    docs = load_and_flatten(dataset_path, limit=limit)
    docs = {
        doc_id: {k: v for k, v in payload.items() if k in ("query_id", "text")}
        for doc_id, payload in docs.items()
    }

    docs_path, docs_mem = save_json(docs, output_dir, "docs.json")
    _print_export_stats("MS MARCO passages", len(docs), docs_path, docs_mem)

    query_ids = {
        payload["query_id"]
        for payload in docs.values()
        if payload.get("query_id") is not None
    }

    queries_path, queries_mem = save_queries_json(
        dataset_path,
        output_dir,
        "queries.json",
        query_ids=query_ids or None,
    )
    _print_export_stats(
        "queries", queries_mem.get("count", 0), queries_path, queries_mem
    )

    answers_path, answers_mem = save_answers_json(
        dataset_path,
        output_dir,
        "answers.json",
        query_ids=query_ids or None,
    )
    _print_export_stats(
        "answers", answers_mem.get("count", 0), answers_path, answers_mem
    )

    return docs


def configure_model_and_agent(args: argparse.Namespace) -> Tuple[COCOM, str, int]:
    if args.checkpoint and args.hf_model_name:
        raise ValueError("Specify either --checkpoint or --hf_model_name, not both")

    model_source = args.checkpoint or args.hf_model_name
    if model_source is None:
        raise ValueError("You must provide either --checkpoint or --hf_model_name")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model = load_model_safely(model_source)
    model.to(device)
    model.eval()

    agent_name = "None"
    if args.use_bandit and args.bandit_agent:
        bandit_path = Path(args.bandit_agent)
        if bandit_path.is_dir():
            bandit_path = bandit_path / "bandit_agent.pkl"
        if bandit_path.exists():
            agent = load_bandit_agent(
                bandit_path.as_posix(), getattr(model, "compr_rates", [])
            )
            if hasattr(model, "set_bandit_agent"):
                model.set_bandit_agent(agent)
            else:
                setattr(model, "bandit_agent", agent)
            agent_name = agent.__class__.__name__
            print(f"Loaded bandit agent from {bandit_path}")
        else:
            print(
                f"Bandit agent not found at {bandit_path}; proceeding with fixed compression rate."
            )
    elif not args.use_bandit:
        print("Bandit agent disabled via --no-use-bandit flag; using fixed compression rate.")

    if getattr(model, "bandit_agent", None) is None:
        print("No bandit agent active; using a single compression rate during generation.")

    base_rate = resolve_base_compression_rate(model, args.compression_rate)
    print(f"Compression configuration -> agent={agent_name}, base_rate={base_rate}")

    setattr(model, "current_rate", base_rate)
    return model, model_source, base_rate


def main() -> None:
    args = parse_args()

    docs = prepare_documents(args.dataset, args.docs_out, args.limit)

    model, model_source, base_rate = configure_model_and_agent(args)

    contexts = generate_contexts(docs, model, base_rate)
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

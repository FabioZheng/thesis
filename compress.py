import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from modeling_cocom import COCOM
from analyse.retrieval import TextEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compressed contexts and embeddings for MS MARCO passages"
    )
    parser.add_argument("dataset", help="Path to MS MARCO dataset file (JSON/JSONL)")
    parser.add_argument("model_name", help="Pretrained COCOM model name or path")
    parser.add_argument("compression_rate", type=int, help="Compression rate")
    parser.add_argument("docs_out", help="Directory to save flattened documents")
    parser.add_argument("contexts_out", help="Directory to save compressed contexts")
    parser.add_argument("embeddings_out", help="Directory to save document embeddings")
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
        default=None,
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
        passage_texts = row.get("passages", {}).get("passage_text", [])
        for passage in passage_texts:
            docs[doc_id] = {"query_id": query_id, "text": passage}
            doc_id += 1
    return docs


def save_json(data: Dict, directory: str, filename: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def generate_contexts(
    docs: Dict[int, Dict[str, str]], model_name: str, rate: int
) -> Dict[int, Dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COCOM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    contexts: Dict[int, Dict[str, Any]] = {}
    for doc_id, item in docs.items():
        tokens = model.compr.tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            emb = model.compr(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                rate=rate,
            )
        contexts[doc_id] = {
            "query_id": item["query_id"],
            "context": emb.cpu().tolist(),
        }
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
    docs_path = save_json(docs, args.docs_out, "docs.json")
    print(f"Extracted {len(docs)} MS MARCO passages -> {docs_path}")

    contexts = generate_contexts(docs, args.model_name, args.compression_rate)
    contexts_path = save_json(contexts, args.contexts_out, "contexts.json")
    print(
        f"Generated contexts for {len(contexts)} MS MARCO passages -> {contexts_path}"
    )

    embeddings = generate_embeddings(
        docs,
        args.embedder_model,
        args.embedder_batch_size,
        args.embedder_device,
        not args.no_embedder_normalize,
    )
    emb_path = save_json(embeddings, args.embeddings_out, "embeddings.json")
    print(
        f"Generated embeddings for {len(embeddings)} MS MARCO passages -> {emb_path}"
    )


if __name__ == "__main__":
    main()

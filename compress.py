import argparse
import json
import os
from typing import Dict

import pandas as pd
import torch

from analyse.run import DatasetAnalyzer
from modeling_cocom import COCOM, COCOMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compressed embeddings using COCOM")
    parser.add_argument("dataset", help="Path to dataset file (CSV/JSON/JSONL/XLSX)")
    parser.add_argument("model_name", help="Pretrained COCOM model name or path")
    parser.add_argument("compression_rate", type=int, help="Compression rate")
    parser.add_argument("docs_out", help="Directory to save flattened documents")
    parser.add_argument("embeddings_out", help="Directory to save embeddings")
    return parser.parse_args()


def load_and_flatten(dataset_path: str) -> Dict[int, str]:
    analyzer = DatasetAnalyzer()
    df = analyzer.load_custom_dataset(dataset_path)
    if df is None:
        raise ValueError(f"Failed to load dataset at {dataset_path}")

    docs: Dict[int, str] = {}
    for doc_id, (_, row) in enumerate(df.iterrows()):
        texts = []
        for col in analyzer.text_columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            texts.append(analyzer._coerce_nested_text(val))
        docs[doc_id] = " ".join(texts).strip()
    return docs


def save_json(data: Dict, directory: str, filename: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return path


def generate_embeddings(docs: Dict[int, str], model_name: str, rate: int) -> Dict[int, list]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COCOM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings: Dict[int, list] = {}
    for doc_id, text in docs.items():
        tokens = model.compr.tokenizer(
            text,
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
        embeddings[doc_id] = emb.cpu().tolist()
    return embeddings


def main() -> None:
    args = parse_args()

    docs = load_and_flatten(args.dataset)
    docs_path = save_json(docs, args.docs_out, "docs.json")
    print(f"Extracted {len(docs)} documents -> {docs_path}")

    embeddings = generate_embeddings(docs, args.model_name, args.compression_rate)
    emb_path = save_json(embeddings, args.embeddings_out, "embeddings.json")
    print(f"Generated embeddings for {len(embeddings)} documents -> {emb_path}")


if __name__ == "__main__":
    main()

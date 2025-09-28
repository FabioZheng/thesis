#!/usr/bin/env python3
"""
Download any Hugging Face dataset and save it as JSON.

Usage:
    python save_hf_dataset_as_json.py <dataset_name_or_path> [--split SPLIT] [--out OUTFILE]

Examples:
    python save_hf_dataset_as_json.py squad --split train --out squad_train.json
    python save_hf_dataset_as_json.py dmrau/multi_qa --split validation
"""

import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Save a Hugging Face dataset split as JSON")
    parser.add_argument("dataset", type=str, help="Dataset name or path (e.g. 'squad', 'dmrau/multi_qa')")
    parser.add_argument("--split", type=str, default="train", help="Split to save (default: train)")
    parser.add_argument("--name", type=str, default=None, help="Split to save (default: train)")
    parser.add_argument("--jsonl", action="store_true", help="Save as JSONL (one object per line) instead of array")

    args = parser.parse_args()

    # Load dataset
    print(f"🔄 Loading dataset: {args.dataset}, split: {args.split}")
    dataset = load_dataset(args.dataset, split=args.split, name=args.name)

    # Default output file
    out_file = f"{args.dataset.replace('/', '_')}_{args.split}.json"

    # Save dataset
    print(f"💾 Saving to {out_file}")
    if args.jsonl:
        dataset.to_json(out_file, orient="records", lines=True)
    else:
        dataset.to_json(out_file)

    print("✅ Done!")


if __name__ == "__main__":
    main()

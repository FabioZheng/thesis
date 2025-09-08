import argparse
import os
import math
import pickle
import json
import matplotlib.pyplot as plt
from rouge import Rouge
import datasets
import torch
from torch.utils.data import DataLoader

from modeling_cocom import COCOM
from cmab_agent import CompressionBanditAgent, batch_entropy
from metrics import exact_match_score, compute_rouge_scores
from utils import prepare_auto_encoding


def collate_batch(batch):
    ret = {}
    for key in batch[0]:
        if 'text' not in key:
            ret[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            ret[key] = [item[key] for item in batch]
    return ret


def load_model_safely(checkpoint_path):
    """
    Load COCOM model with enhanced error handling and compatibility checks
    """
    print(f"Loading model from: {checkpoint_path}")

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Check for metadata file
    metadata_path = os.path.join(checkpoint_path, 'cmab_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Found CMAB metadata: {metadata}")

    try:
        # Load the model
        model = COCOM.from_pretrained(checkpoint_path, trust_remote_code=True)

        # Ensure current_rate is set for compatibility
        if not hasattr(model, 'current_rate') or model.current_rate is None:
            model.current_rate = model.compr_rates[0] if hasattr(model, 'compr_rates') else 64

        print(f"Model loaded successfully!")
        print(f"Compression rates: {model.compr_rates}")
        print(f"Current rate: {model.current_rate}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load with alternative method...")

        # Alternative loading method
        from modeling_cocom import COCOMConfig
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            config = COCOMConfig.from_pretrained(checkpoint_path)
            model = COCOM(config)
            model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'pytorch_model.bin')))
            model.current_rate = model.compr_rates[0]
            return model
        else:
            raise Exception("Could not load model with any method")


def get_args():
    parser = argparse.ArgumentParser(description="Train contextual bandit for compression rate selection")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained COCOM checkpoint")
    parser.add_argument("--dataset_name_or_dir", type=str, default="ms_marco", help="HF dataset or local path")
    parser.add_argument("--doc_max_length", type=int, default=128, help="Maximum document length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_examples", type=int, default=1024, help="Number of training examples")
    parser.add_argument("--output_dir", type=str, default="bandit_ckpt", help="Where to store the trained agent")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length during evaluation")
    parser.add_argument("--alpha", type=float, default=1.0, help="UCB exploration parameter")
    return parser.parse_args()


def prepare_dataset(dataset, dataset_name):
    """
    Prepare dataset with consistent text field extraction
    """
    print(f"Preparing dataset: {dataset_name}")
    print(f"Original columns: {dataset.column_names}")
    print(f"Sample data: {dataset[:2]}")

    # Handle different dataset formats
    if "passages" in dataset.column_names:
        print("Detected MS MARCO format with passages")

        def extract_passages(example):
            if isinstance(example["passages"]["passage_text"], list):
                return {"text": example["passages"]["passage_text"][0]}
            return {"text": str(example["passages"]["passage_text"])}

        dataset = dataset.map(extract_passages)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    elif "context" in dataset.column_names:
        print("Detected context column, renaming to text")
        dataset = dataset.rename_column("context", "text")
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    elif "text" not in dataset.column_names:
        # Try to find a suitable text column
        text_candidates = [col for col in dataset.column_names
                           if any(keyword in col.lower() for keyword in ['text', 'content', 'document', 'passage'])]
        if text_candidates:
            print(f"Using column '{text_candidates[0]}' as text")
            dataset = dataset.rename_column(text_candidates[0], "text")
            dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
        else:
            raise ValueError(f"No suitable text column found. Available columns: {dataset.column_names}")

    # Filter out empty or very short texts
    def filter_valid_text(example):
        return len(str(example["text"]).strip()) > 20

    dataset = dataset.filter(filter_valid_text)

    print(f"Final dataset size: {len(dataset)}")
    print(f"Sample text: {dataset[0]['text'][:100]}...")

    return dataset


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with enhanced error handling
    model = load_model_safely(args.checkpoint)
    model = model.to(device)
    model.eval()

    print(f"Model compression rates: {model.compr_rates}")

    # Initialize bandit agent
    agent = CompressionBanditAgent(model.compr_rates, alpha=args.alpha)
    model.set_bandit_agent(agent)

    # Load and prepare dataset
    if os.path.exists(args.dataset_name_or_dir):
        dataset = datasets.load_from_disk(args.dataset_name_or_dir)
        if isinstance(dataset, datasets.DatasetDict):
            dataset = dataset["train"]
    else:
        try:
            dataset = datasets.load_dataset(args.dataset_name_or_dir, "v2.1")["train"]
        except:
            try:
                dataset = datasets.load_dataset(args.dataset_name_or_dir)["train"]
            except:
                dataset = datasets.load_dataset(args.dataset_name_or_dir, split="train")

    # Select subset of examples
    dataset = dataset.select(range(min(args.num_examples, len(dataset))))

    # Prepare dataset for consistent format
    dataset = prepare_dataset(dataset, args.dataset_name_or_dir)

    # Training data for bandit
    rewards_history = []
    rouge = Rouge()

    print(f"\n🎯 Training bandit agent on {len(dataset)} examples")
    print(f"Compression rates: {model.compr_rates}")

    for rate in model.compr_rates:
        print(f"\n🔵 Testing Compression Rate: {rate}")

        # Prepare dataset for this compression rate
        prepped = dataset.map(
            prepare_auto_encoding,
            batched=True,
            load_from_cache_file=False,
            fn_kwargs={
                "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
                "decoder_tokenizer": model.decoder_tokenizer,
                "compression_rate": rate,
                "enc_max_len": args.doc_max_length,
                "train": False,
            },
        )

        loader = DataLoader(prepped, batch_size=args.batch_size, collate_fn=collate_batch)
        model.current_rate = rate

        batch_rewards = []
        total_batches = len(loader)

        for idx, batch in enumerate(loader):
            texts = batch.pop("text")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                preds = model.generate(batch, max_new_tokens=args.max_new_tokens)

            # Calculate rewards and update bandit
            entropies = batch_entropy(batch["enc_input_ids"].cpu(), batch["enc_attention_mask"].cpu())

            for ent, pred, gold in zip(entropies, preds, texts):
                # Calculate ROUGE-1 F1 as reward
                rouge_scores = compute_rouge_scores(rouge, [pred], [gold])
                reward = rouge_scores['Rouge-1']

                # Add compression efficiency bonus
                compression_bonus = 1.0 / math.sqrt(rate)  # Higher reward for higher compression
                final_reward = reward + 0.1 * compression_bonus

                # Update bandit
                agent.update(ent, rate, final_reward)
                batch_rewards.append(final_reward)

            # Progress reporting
            if (idx + 1) % max(1, total_batches // 10) == 0:
                recent_reward = sum(batch_rewards[-len(texts):]) / len(texts)
                print(f"  Batch {idx + 1}/{total_batches}: Recent Reward={recent_reward:.4f}")

        avg_rate_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
        rewards_history.append((rate, avg_rate_reward))
        print(f"✅ Avg Reward for Rate {rate}: {avg_rate_reward:.4f}")

    # Save trained agent
    os.makedirs(args.output_dir, exist_ok=True)

    # Save agent state
    agent_data = {
        "A": agent.A,
        "b": agent.b,
        "rates": agent.rates,
        "alpha": agent.alpha,
        "training_info": {
            "num_examples": args.num_examples,
            "doc_max_length": args.doc_max_length,
            "dataset": args.dataset_name_or_dir,
            "model_checkpoint": args.checkpoint
        }
    }

    with open(os.path.join(args.output_dir, "bandit_agent.pkl"), "wb") as f:
        pickle.dump(agent_data, f)

    # Save rewards history
    with open(os.path.join(args.output_dir, "rewards_history.json"), "w") as f:
        json.dump(rewards_history, f, indent=2)

    # Plot and save results
    if rewards_history:
        rates, avg_rewards = zip(*rewards_history)
        plt.figure(figsize=(10, 6))
        plt.plot(rates, avg_rewards, marker='o', linewidth=2, markersize=8)
        plt.title("Average Reward per Compression Rate", fontsize=14)
        plt.xlabel("Compression Rate", fontsize=12)
        plt.ylabel("Average Reward (ROUGE-1 + Compression Bonus)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "reward_progress.png"), dpi=300)
        plt.close()

        print(f"\n📊 Results saved to {args.output_dir}/")
        print(f"📈 Best compression rate: {rates[np.argmax(avg_rewards)]} (reward: {max(avg_rewards):.4f})")
    else:
        print("⚠️  No rewards collected - check your data and model compatibility")

    print(f"\n✅ Bandit training completed!")
    print(f"📁 Agent saved to: {args.output_dir}/bandit_agent.pkl")


if __name__ == "__main__":
    main()
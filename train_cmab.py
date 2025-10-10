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
import numpy as np
import wandb
from modeling_cocom import COCOM
from cmab_agent import CompressionBanditAgent
from metrics import batch_entropy
from metrics import exact_match_score, compute_rouge_scores
from utils import prepare_auto_encoding
from bert_score import BERTScorer


def collate_batch(batch):
    ret = {}
    for key in batch[0]:
        if 'text' not in key:
            ret[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            ret[key] = [item[key] for item in batch]
    return ret


def load_model_safely(model_source):
    """
    Load COCOM model with enhanced error handling and compatibility checks
    """
    print(f"Loading model from: {model_source}")

    is_local_path = os.path.exists(model_source)
    if not is_local_path:
        print("Checkpoint not found locally. Attempting to load from the Hugging Face Hub...")

    # Check for metadata file
    if is_local_path:
        metadata_path = os.path.join(model_source, 'cmab_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Found CMAB metadata: {metadata}")

    try:
        # Load the model
        model = COCOM.from_pretrained(model_source, trust_remote_code=True)

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

        # Alternative loading method (local checkpoints only)
        if not is_local_path:
            raise Exception("Could not load model from Hugging Face Hub") from e

        from modeling_cocom import COCOMConfig
        config_path = os.path.join(model_source, 'config.json')
        if os.path.exists(config_path):
            config = COCOMConfig.from_pretrained(model_source)
            model = COCOM(config)
            model.load_state_dict(torch.load(os.path.join(model_source, 'pytorch_model.bin')))
            model.current_rate = model.compr_rates[0]
            return model
        else:
            raise Exception("Could not load model with any method")


def get_args():
    parser = argparse.ArgumentParser(description="Train contextual bandit for compression rate selection")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained COCOM checkpoint")
    parser.add_argument("--hf_model_name", type=str, default=None, help="Hugging Face model id to load instead of a local checkpoint")
    parser.add_argument("--dataset_name_or_dir", type=str, default="ms_marco", help="HF dataset or local path")
    parser.add_argument("--doc_max_length", type=int, default=128, help="Maximum document length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_examples", type=int, default=1024, help="Number of training examples")
    parser.add_argument("--output_dir", type=str, default="bandit_ckpt", help="Where to store the trained agent")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length during evaluation")
    parser.add_argument("--alpha", type=float, default=1.0, help="UCB exploration parameter")
    parser.add_argument("--r_alpha", type=float, default=1.0, help="α weight for BERTScore_F1")
    parser.add_argument("--r_beta", type=float, default=1.0, help="β weight for ROUGE-L")
    parser.add_argument("--r_gamma", type=float, default=0.1, help="γ weight for 1/compression_rate penalty")
    parser.add_argument("--bertscore_lang", type=str, default="en", help="Language or model for BERTScore")
    parser.add_argument("--bertscore_rescale", action="store_true", help="Rescale BERTScore with baseline")
    parser.add_argument("--display_every", type=int, default=100, help="Frequency (in examples) to display reconstruction samples")
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
    wandb.init(project="COCOM CMAB", config=vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with enhanced error handling
    model_source = args.hf_model_name if args.hf_model_name else args.checkpoint
    model = load_model_safely(model_source)
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
    bert_scorer = BERTScorer(
        lang = args.bertscore_lang,
        rescale_with_baseline = args.bertscore_rescale
    )

    print(f"\n🎯 Training bandit agent (online UCB) on {len(dataset)} examples")
    print(f"Compression rates: {model.compr_rates}")

    # DO NOT attach the bandit to the model during training (to avoid double selection inside COCOM)
    # We will attach it after training for inference-time selection.
    # model.set_bandit_agent(agent)  # <- keep this commented during training

    # Prepare dataset once (no per-rate remapping)
    prepped = dataset.map(
        prepare_auto_encoding,
        batched=True,
        load_from_cache_file=False,
        fn_kwargs={
            "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
            "decoder_tokenizer": model.decoder_tokenizer,
            # compression_rate is NOT fixed here; we supply the chosen rate per example at runtime
            "compression_rate": model.current_rate,  # placeholder; the model’s current_rate will be set per-example
            "enc_max_len": args.doc_max_length,
            "train": False,
        },
    )

    loader = DataLoader(prepped, batch_size=args.batch_size, collate_fn=collate_batch)

    # Track rewards per rate for reporting
    rate_reward_sum = {r: 0.0 for r in model.compr_rates}
    rate_reward_cnt = {r: 0 for r in model.compr_rates}

    total_examples = 0
    for batch_idx, batch in enumerate(loader):
        texts = batch.pop("text")
        # Compute contexts (entropy) on CPU tensors expected by batch_entropy
        entropies = batch_entropy(batch["enc_input_ids"], batch["enc_attention_mask"])

        ent_arr = np.asarray(entropies, dtype=float)
        ent_min = float(ent_arr.min()) if ent_arr.size else float('nan')
        ent_max = float(ent_arr.max()) if ent_arr.size else float('nan')
        ent_mean = float(ent_arr.mean()) if ent_arr.size else float('nan')
        ent_std = float(ent_arr.std(ddof=0)) if ent_arr.size else float('nan')
        print(
            f"Batch {batch_idx + 1}/{len(loader)} | "
            f"entropy min={ent_min:.4f}, max={ent_max:.4f}, "
            f"avg={ent_mean:.4f}, std={ent_std:.4f}"
        )

        B = len(texts)

        # Process each example independently: select → play → update
        for i in range(B):
            x = float(entropies[i])  # context feature (entropy); if you use d>1 features, pass a vector
            # UCB arm selection (uses A_a, b_a, alpha inside the agent)
            chosen_rate = agent.select_rate(x)

            # Build a single-example sub-batch
            ex = {
                "enc_input_ids": batch["enc_input_ids"][i:i + 1].to(device),
                "enc_attention_mask": batch["enc_attention_mask"][i:i + 1].to(device),
                "dec_input_ids": batch["dec_input_ids"][i:i + 1].to(device),
                "dec_attention_mask": batch["dec_attention_mask"][i:i + 1].to(device),
            }

            # Force the model to use the chosen rate (no bandit attached during training)
            model.current_rate = chosen_rate

            with torch.no_grad():
                pred = model.generate(ex, max_new_tokens=args.max_new_tokens)[0]

            gold = texts[i]

            # Reward = autoencoding fidelity + compression bonus
            rouge_scores = compute_rouge_scores(rouge, [pred], [gold])
            rouge_l = rouge_scores['Rouge-L']
            _, _, f1 = bert_scorer.score([pred], [gold])  # returns tensors
            bertscore_f1 = float(f1.mean().item())
            compression_penalty = 1.0 / float(chosen_rate)
            r = args.r_beta * rouge_l - args.r_gamma * compression_penalty

            # Online update ONLY the chosen arm with (x, r)
            agent.update(x, chosen_rate, r)

            # Accounting
            rate_reward_sum[chosen_rate] += r
            rate_reward_cnt[chosen_rate] += 1
            total_examples += 1

            if args.display_every > 0 and total_examples % args.display_every == 0:
                print("\n🔍 Autoencoding quality check")
                print(f"Example #{total_examples} | Compression rate: {chosen_rate}")
                print(f"Original    : {gold}")
                print(f"Reconstructed: {pred}\n")

        # Progress log: average reward and selection counts
        if (batch_idx + 1) % max(1, len(loader) // 10) == 0:
            filled = {
                r: (rate_reward_sum[r] / max(1, rate_reward_cnt[r]))
                for r in model.compr_rates if rate_reward_cnt[r] > 0
            }
            avg_recent = np.mean(list(filled.values())) if filled else 0.0
            print(f"  Batch {batch_idx + 1}/{len(loader)}: avg reward = {avg_recent:.4f}")

            # Print selection counts for each rate
            counts_str = " | ".join([f"rate {r}: {rate_reward_cnt[r]}" for r in model.compr_rates])
            print(f"    Selections so far → {counts_str}")

            # Log metrics to Weights & Biases
            log_data = {"avg_reward": avg_recent}
            for r in model.compr_rates:
                log_data[f"selections_rate_{r}"] = rate_reward_cnt[r]
            wandb.log(log_data)


    # Summarize per-rate averages
    rewards_history = []
    for r in model.compr_rates:
        avg_r = (rate_reward_sum[r] / rate_reward_cnt[r]) if rate_reward_cnt[r] > 0 else 0.0
        rewards_history.append((r, avg_r))
        print(f"✅ Avg Reward (played) for Rate {r}: {avg_r:.4f}")

    # Now attach the trained bandit for inference-time selection inside COCOM.generate()
    model.set_bandit_agent(
        agent)  # COCOM will call agent.select_rate(...) based on entropy at generation time

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
            "model_checkpoint": args.checkpoint,
            "model_source": model_source
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
    wandb.finish()


if __name__ == "__main__":
    main()

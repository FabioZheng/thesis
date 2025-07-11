import argparse
import os
import math
import pickle
import matplotlib.pyplot as plt
from rouge import Rouge
import datasets
import torch
from torch.utils.data import DataLoader

from modeling_cocom import COCOM
from cmab_agent import CompressionBanditAgent, batch_entropy
from metrics import exact_match_score,compute_rouge_scores
from utils import prepare_auto_encoding


def collate_batch(batch):
    ret = {}
    for key in batch[0]:
        if 'text' not in key:
            ret[key] = torch.stack([torch.tensor(item[key]) for item in batch])
        else:
            ret[key] = [item[key] for item in batch]
    return ret


def get_args():
    parser = argparse.ArgumentParser(description="Train contextual bandit for compression rate selection")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained COCOM checkpoint")
    parser.add_argument("--dataset_name_or_dir", type=str, default="squad", help="HF dataset or local path")
    parser.add_argument("--doc_max_length", type=int, default=128, help="Maximum document length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_examples", type=int, default=1024, help="Number of training examples")
    parser.add_argument("--output_dir", type=str, default="bandit_ckpt", help="Where to store the trained agent")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Generation length during evaluation")
    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = COCOM.from_pretrained(args.checkpoint)
    model = model.to(device)

    print(model.compr_rates)

    agent = CompressionBanditAgent(model.compr_rates)
    model.set_bandit_agent(agent)

    if os.path.exists(args.dataset_name_or_dir):
        dataset = datasets.load_from_disk(args.dataset_name_or_dir)
    else:
        dataset = datasets.load_dataset(args.dataset_name_or_dir)
    dataset = dataset["train"].select(range(args.num_examples))
    if "context" in dataset.column_names:
        dataset = dataset.rename_column("context", "text")
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "text"])

    # 📈 To store rewards for plotting
    rewards_history = []
    rouge=Rouge()

    for rate in model.compr_rates:
        print(f"\n🔵 Compression Rate: {rate}")
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
        model.eval()

        batch_rewards = []
        for idx, batch in enumerate(loader):
            texts = batch.pop("text")
            batch = {k: v.to(device) for k, v in batch.items()}
            print(batch.keys())
            print(batch["labels"])
            with torch.no_grad():
                preds = model.generate(batch, max_new_tokens=args.max_new_tokens)
                print(len(preds))
                print(preds[0][:100])
                print(texts[0][:100])

            entropies = batch_entropy(batch["enc_input_ids"].cpu(), batch["enc_attention_mask"].cpu())
            for ent, pred, gold in zip(entropies, preds, texts):
                acc = compute_rouge_scores(rouge,[pred], [gold])
                acc=acc['Rouge-1']
                reward = acc * math.sqrt(rate)
                agent.update(ent, rate, reward)
                batch_rewards.append(reward)

            # Print progress every 50 steps
            if (idx + 1) % 2 == 0:
                avg_reward = sum(batch_rewards[-2:]) / 2
                print(f"Batch {idx+1}: Avg Reward={avg_reward:.4f}")

        avg_rate_reward = sum(batch_rewards) / len(batch_rewards)
        rewards_history.append((rate, avg_rate_reward))
        print(f"✅ Avg Reward for Rate {rate}: {avg_rate_reward:.4f}")

    # Save agent
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "bandit_agent.pkl"), "wb") as f:
        pickle.dump({"A": agent.A, "b": agent.b, "rates": agent.rates, "alpha": agent.alpha}, f)

    # 📊 Plot rewards per compression rate
    rates, avg_rewards = zip(*rewards_history)
    plt.figure(figsize=(8, 5))
    plt.plot(rates, avg_rewards, marker='o')
    plt.title("Average Reward per Compression Rate")
    plt.xlabel("Compression Rate")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, "reward_progress.png"))
    print(f"📊 Reward plot saved to {args.output_dir}/reward_progress.png")


if __name__ == "__main__":
    main()


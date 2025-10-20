# run_evaluation.py

import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import csv
import numpy as np
import argparse  # New import for command-line arguments

# --- New Evaluation Imports ---
# Make sure to install these: pip install bert-score sentence-transformers
try:
    from bert_score import score as bert_scorer
    from sentence_transformers.cross_encoder import CrossEncoder
except ImportError:
    print("=" * 50)
    print("ERROR: Missing required libraries.")
    print("Please install them by running: pip install bert-score sentence-transformers")
    print("=" * 50)
    exit(1)

# --- Configuration ---
MODEL_REPO = "ielabgroup/tinyllama-compression-multi-rate-4-16-128"
JUDGE_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

DATASET_NAME = "ms_marco"
DATASET_CONFIG = "v1.1"
DATASET_SPLIT = "train"
# N_EXAMPLES is now set by command-line argument
MAX_NEW_TOKENS = 128
CONTEXT_PREFIX = "use the following document to help answer: "
TOKENIZER_MAX_LENGTH = 2048 - MAX_NEW_TOKENS

# --- Output File ---
OUTPUT_CSV_FILE = "baseline_evaluation_results.csv"


# --- Helper Functions ---

def load_model_and_tokenizer(model_repo):
    """Loads the CoCoM model and returns its decoder and tokenizer."""
    print(f"Loading CoCoM model: {model_repo}")
    cocom_model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if not hasattr(cocom_model, "decoder") or not hasattr(cocom_model, "decoder_tokenizer"):
        raise ValueError("Loaded CoCoM model is missing the decoder components")

    decoder = cocom_model.decoder
    tokenizer = cocom_model.decoder_tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    decoder.eval()

    device = None
    try:
        device = next(decoder.parameters()).device
    except StopIteration:
        device = None

    if (device is None or device.type == 'meta') and hasattr(decoder, "hf_device_map"):
        first_device = next(iter(decoder.hf_device_map.values()))
        if isinstance(first_device, str):
            device = torch.device(first_device)
        elif isinstance(first_device, int):
            device = torch.device(f"cuda:{first_device}") if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(first_device, torch.device):
            device = first_device

    if device is None or device.type == 'meta':
        device = torch.device('cpu')

    print(f"Decoder loaded on device: {device}")

    # Keep a reference to the parent CoCoM model to avoid it being garbage collected.
    decoder._cocom_parent = cocom_model

    return decoder, tokenizer, device


def load_judge_model(model_name, device):
    """Loads the Cross-Encoder model to act as a judge."""
    print(f"Loading judge model: {model_name}")
    judge_model = CrossEncoder(model_name, max_length=512)
    print(f"Judge model loaded.")
    return judge_model


def process_example(example):
    """Extracts query, answer, and context from an ms_marco example."""
    query = example.get('query')
    answers = example.get('answers', [])
    real_answer = answers[0] if answers else ""

    passages = example.get('passages', {})
    documents = passages.get('passage_text', [])
    context = " ".join(documents)

    return query, real_answer, context


def generate_answer(model, tokenizer, prompt_text, device):
    """Generates an answer from the model given a prompt."""
    if not prompt_text:
        return "[Prompt Error]"
    try:
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH,
            padding=True
        ).to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )

        full_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]
        prompt_text_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        answer = full_text[len(prompt_text_decoded):].strip()

        return answer

    except Exception as e:
        print(f"Error during generation: {e}")
        return "[Generation Error]"


def format_chat_prompt(tokenizer, messages):
    """Applies the model's chat template to a list of messages."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        print(f"Error formatting chat prompt: {e}")
        return ""


def get_bert_score(candidate, reference, device):
    """Calculates BERTScore F1."""
    if not candidate or not reference:
        return 0.0
    try:
        # Suppress BERTScore warnings
        _, _, f1 = bert_scorer([candidate], [reference], lang='en', device=device, verbose=False)
        return f1.item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0


def get_judge_score(judge_model, pair_list):
    """Gets a relevance/faithfulness score from the judge model."""
    valid_pairs = [(str(a), str(b)) for a, b in pair_list if a and b]
    if not valid_pairs:
        return [0.0] * len(pair_list)

    try:
        return judge_model.predict(valid_pairs, show_progress_bar=False)
    except Exception as e:
        print(f"Error getting judge score: {e}")
        return [0.0] * len(pair_list)


def save_results_csv(results, filename):
    """Saves all results to a single CSV file."""
    if not results:
        print("No results to save to CSV.")
        return

    print(f"\nSaving results to {filename}...")
    # These headers include the text and all scores for a complete record
    headers = [
        "query_id", "bert_f1_query_only", "bert_f1_rag", "relevancy_rag", "faithfulness_rag"
    ]

    try:
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print("CSV results saved successfully.")
    except Exception as e:
        print(f"Error saving CSV file: {e}")


# --- Main Execution ---

def main():
    # --- NEW: Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Run RAG baseline evaluation on ms_marco.")
    parser.add_argument(
        "-n", "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate from the dataset."
    )
    args = parser.parse_args()
    N_EXAMPLES = args.num_samples
    # --- End of new argument parsing ---

    model, tokenizer, device = load_model_and_tokenizer(MODEL_REPO)
    judge_model = load_judge_model(JUDGE_MODEL_NAME, device)

    print(f"\nLoading and streaming dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        streaming=True,
        split=DATASET_SPLIT
    )

    limited_dataset = dataset.take(N_EXAMPLES)
    all_results = []
    print(f"Running baseline evaluation on {N_EXAMPLES} samples...")

    for i, example in enumerate(limited_dataset):
        print("\n" + "=" * 80)
        print(f"Processing Example {i + 1} / {N_EXAMPLES}")
        print("=" * 80)

        query_id = example.get('query_id', i)

        try:
            query, real_answer, context = process_example(example)
            if not query or not real_answer:
                print(f"Skipping example {query_id}: missing query or real answer.")
                continue

            # --- 1. Query-Only Baseline ---
            messages_query_only = [
                {"role": "system", "content": "You are a helpful assistant that answers questions."},
                {"role": "user", "content": query}
            ]
            prompt_query_only = format_chat_prompt(tokenizer, messages_query_only)
            answer_query_only = generate_answer(model, tokenizer, prompt_query_only, device)

            # --- 2. RAG Baseline ---
            rag_user_prompt = (
                f"{CONTEXT_PREFIX}{context}\n\n"
                f"Based on the document, please answer the following question:\n{query}"
            )
            messages_rag = [
                {"role": "system",
                 "content": "You are a helpful assistant that answers questions based on the provided document."},
                {"role": "user", "content": rag_user_prompt}
            ]
            prompt_rag = format_chat_prompt(tokenizer, messages_rag)
            answer_rag = generate_answer(model, tokenizer, prompt_rag, device)

            # --- 3. Evaluation ---
            bert_f1_query_only = get_bert_score(answer_query_only, real_answer, device)
            bert_f1_rag = get_bert_score(answer_rag, real_answer, device)

            relevancy_query_only = get_judge_score(judge_model, [(query, answer_query_only)])[0]
            relevancy_rag = get_judge_score(judge_model, [(query, answer_rag)])[0]

            faithfulness_query_only = 0.0  # Not applicable
            faithfulness_rag = get_judge_score(judge_model, [(context, answer_rag)])[0]

            # --- 4. Store results ---
            result_item = {
                "query_id": query_id,
                "query": query,
                "real_answer": real_answer,
                "generated_answer_query_only": answer_query_only,
                "bert_f1_query_only": bert_f1_query_only,
                "relevancy_query_only": relevancy_query_only,
                "generated_answer_rag": answer_rag,
                "bert_f1_rag": bert_f1_rag,
                "relevancy_rag": relevancy_rag,
                "faithfulness_rag": faithfulness_rag
            }
            all_results.append(result_item)

            # --- 5. Print detailed results for this item ---
            print(f"QUERY: {query}")
            print(f"GOLDEN ANSWER: {real_answer}\n")

            print("--- (1) QUERY-ONLY RESULT ---")
            print(f"Prediction: {answer_query_only}")
            print(f" -> BERTScore (Correctness): {bert_f1_query_only:.4f}")
            print(f" -> Relevancy (to Query):    {relevancy_query_only:.4f}\n")

            print("--- (2) RAG RESULT ---")
            print(f"Prediction: {answer_rag}")
            print(f" -> BERTScore (Correctness): {bert_f1_rag:.4f}")
            print(f" -> Relevancy (to Query):    {relevancy_rag:.4f}")
            print(f" -> Faithfulness (to Doc): {faithfulness_rag:.4f}")


        except Exception as e:
            print(f"CRITICAL Error processing example {query_id}: {e}")
            continue

    # --- Save CSV Results ---
    print("\n" + "=" * 80)

    csv_results=[]
    for item in all_results:
        csv_results.append({
            "query_id": item["query_id"],
            "bert_f1_query_only": item["bert_f1_query_only"],
            "bert_f1_rag": item["bert_f1_rag"],
            "relevancy_rag": item["relevancy_rag"],
            "faithfulness_rag": item["faithfulness_rag"]
        })
    save_results_csv(csv_results, OUTPUT_CSV_FILE)

    # --- Print Average Scores ---
    if all_results:
        print("\n--- Average Scores ---")
        avg_bert_f1_query = np.mean([r['bert_f1_query_only'] for r in all_results])
        avg_bert_f1_rag = np.mean([r['bert_f1_rag'] for r in all_results])
        avg_relevancy_query = np.mean([r['relevancy_query_only'] for r in all_results])
        avg_relevancy_rag = np.mean([r['relevancy_rag'] for r in all_results])
        avg_faithfulness_rag = np.mean([r['faithfulness_rag'] for r in all_results])

        print(f"Query-Only | BERTScore F1: {avg_bert_f1_query:.4f} | Relevancy: {avg_relevancy_query:.4f}")
        print(
            f"RAG        | BERTScore F1: {avg_bert_f1_rag:.4f} | Relevancy: {avg_relevancy_rag:.4f} | Faithfulness: {avg_faithfulness_rag:.4f}")
        print("------------------------")

    print("Done.")


if __name__ == "__main__":
    main()
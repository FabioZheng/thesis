# baseline.py

import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Import the custom Cocom model definition from your repository
try:
    from modeling_cocom import CocomForCausalLM
except ImportError:
    print("Error: 'modeling_cocom.py' not found.")
    print("Please ensure 'baseline.py' is in the same directory as 'modeling_cocom.py'.")
    exit(1)

# --- Configuration ---
MODEL_NAME = "ielabgroup/tinyllama-compression-multi-rate-4-16-128"
DATASET_NAME = "ms_marco"
DATASET_CONFIG = "v1.1"
DATASET_SPLIT = "train"  # 'v1.1' only has 'train' and 'test' splits
N_EXAMPLES = 100  # Number of examples to process from the stream
MAX_NEW_TOKENS = 128  # Max tokens to generate for an answer
OUTPUT_FILE = "baseline_results.json"
CONTEXT_PREFIX = "use the following document to help answer: "
# Set a max length for the input to avoid OOM errors, leaving space for generation
# Cocom's base (GPT-2) is 1024, but let's be safer with RAG prompts.
# Adjust this based on your available VRAM.
TOKENIZER_MAX_LENGTH = 1024 - MAX_NEW_TOKENS


# --- Helper Functions ---

def load_model_and_tokenizer(model_name):
    """Loads the Cocom model and tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = CocomForCausalLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded on device: {device}")

    return model, tokenizer, device


def process_example(example):
    """
    Extracts query, answer, and context from an ms_marco example,
    similar to the logic described for save_json.py.
    """
    query = example.get('query')
    answers = example.get('answers', [])

    # Use the first answer as the "real answer"
    real_answer = answers[0] if answers else "No Answer Provided"

    # Concatenate all passages to form the document context
    passages = example.get('passages', {})
    documents = passages.get('passage_text', [])
    context = " ".join(documents)

    return query, real_answer, context


def generate_answer(model, tokenizer, prompt_text, device):
    """
    Generates an answer from the model given a prompt.
    """
    try:
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=TOKENIZER_MAX_LENGTH
        ).to(device)

        input_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=5,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode only the newly generated tokens
        generated_tokens = output_sequences[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return answer

    except Exception as e:
        print(f"Error during generation: {e}")
        return "[Generation Error]"


# --- Main Execution ---

def main():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME)

    print(f"\nLoading and streaming dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        streaming=True,
        split=DATASET_SPLIT
    )

    # Take a subset of the stream
    limited_dataset = dataset.take(N_EXAMPLES)

    results = []
    print(f"Running baseline evaluation on {N_EXAMPLES} examples...")

    for i, example in enumerate(tqdm(limited_dataset, total=N_EXAMPLES)):
        query_id = example.get('query_id', i)
        try:
            query, real_answer, context = process_example(example)

            if not query:
                print(f"Skipping example {query_id}: missing query.")
                continue

            # --- 1. Query-Only Baseline ---
            # Format prompt for better QA
            prompt_query_only = f"Question: {query}\nAnswer:"
            answer_query_only = generate_answer(model, tokenizer, prompt_query_only, device)

            # --- 2. RAG Baseline ---
            # Use the exact prefix requested
            prompt_rag = (
                f"{CONTEXT_PREFIX}{context}\n\n"
                f"Question: {query}\nAnswer:"
            )
            answer_rag = generate_answer(model, tokenizer, prompt_rag, device)

            # Store results
            results.append({
                "query_id": query_id,
                "query": query,
                "real_answer": real_answer,
                "generated_answer_query_only": answer_query_only,
                "generated_answer_rag": answer_rag
            })

            # Optional: Print intermediate results for debugging
            if i % 20 == 0 or i == N_EXAMPLES - 1:
                print("\n---")
                print(f"Example {i} (ID: {query_id})")
                print(f"Query: {query}")
                print(f"Real Answer: {real_answer}")
                print(f"Answer (Query Only): {answer_query_only}")
                print(f"Answer (RAG): {answer_rag}")
                print("---")

        except Exception as e:
            print(f"Error processing example {query_id}: {e}")
            continue

    # --- Save Results ---
    print(f"\nEvaluation complete. Saving {len(results)} results to {OUTPUT_FILE}")
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Done.")
    except Exception as e:
        print(f"Error saving results to file: {e}")


if __name__ == "__main__":
    main()
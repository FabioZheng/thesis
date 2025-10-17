import argparse
import torch


def get_fine_tuning_args():
    """Argument parser dedicated to the QA RAG fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description="RAG QA Fine-tuning Configuration")
    parser.add_argument("--doc_max_length", type=int, default=128)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic",
        help="Hugging Face model identifier that provides the fine-tuning configuration.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional local checkpoint to load the COCOM model weights from.",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of times to iterate over the local QA dataset during fine-tuning.",
    )
    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=50,
        help="Run evaluation and display metrics every N training steps. Set to 0 to disable.",
    )
    parser.add_argument(
        "--save_every_steps",
        type=int,
        default=200,
        help="Overwrite the fine-tuned model checkpoint every N training steps. Set to 0 to disable.",
    )
    parser.add_argument(
        "--compression_rates",
        type=int,
        nargs="+",
        default=None,
        help="Optional override for compression rates when not using the pretrained configuration values.",
    )
    parser.add_argument("--lora", type=str, default="False")
    parser.add_argument("--experiment_folder", type=str, default="experiments_compress_real")
    parser.add_argument("--dataset_name_or_dir", type=str, default="openwebtext")
    parser.add_argument(
        "--compression_linear_type",
        type=str,
        default=None,
        help="Optional override for the compression projection layout when not using the pretrained value.",
    )
    parser.add_argument("--rag_contexts_path", type=str, default="data/contexts/contexts.h5")
    parser.add_argument("--rag_embeddings_path", type=str, default="data/embeddings/embeddings.npz")
    parser.add_argument("--rag_docs_path", type=str, default="data/docs.json")
    parser.add_argument("--rag_queries_path", type=str, default="data/queries.json")
    parser.add_argument("--rag_query_embeddings_path", type=str, default="data/query_embeddings.npz")
    parser.add_argument(
        "--rag_answers_path",
        type=str,
        default="data/answers.json",
        help="Optional path to a JSON file containing answers for each query id.",
    )
    parser.add_argument(
        "--retriever_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--retriever_batch_size", type=int, default=64)
    parser.add_argument(
        "--retriever_device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--retriever_top_k", type=int, default=5)
    parser.add_argument("--decoder_max_length", type=int, default=256)
    parser.add_argument(
        "--show_retrieval_preview",
        action="store_true",
        help=(
            "Enable verbose FAISS retrieval previews during evaluation checkpoints. When set, "
            "the script prints a sample query from the evaluation split along with the "
            "retrieved document ids and text snippets at every evaluation step."
        ),
    )
    return parser.parse_args()

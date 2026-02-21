import argparse
import torch


def get_fine_tuning_args():
    """Argument parser dedicated to the QA RAG fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description="RAG QA Fine-tuning Configuration")
    parser.add_argument("--doc_max_length", type=int, default=256)
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
    parser.add_argument(
        "--rag_use_doc_text_context",
        action="store_true",
        help=(
            "Use retrieved document text from --rag_docs_path as the RAG context in prompts "
            "instead of context embeddings from --rag_contexts_path."
        ),
    )
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
    parser.add_argument("--faiss_batch_size", type=int, default=64)
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
    parser.add_argument(
        "--auto_continue_training",
        action="store_true",
        help=(
            "Automatically rerun compress.py between training cycles to stream additional "
            "chunks from the dataset and resume fine-tuning from the latest checkpoint."
        ),
    )
    parser.add_argument(
        "--auto_continue_cycles",
        type=int,
        default=1,
        help=(
            "Total number of consecutive training cycles to run when auto continuation is "
            "enabled. Each cycle consumes a new chunk of data produced by compress.py."
        ),
    )
    parser.add_argument(
        "--compress_script_path",
        type=str,
        default="compress.py",
        help="Path to the compress.py entrypoint that should be invoked between cycles.",
    )
    parser.add_argument(
        "--compress_hf_model_name",
        type=str,
        default="ielabgroup/tinyllama-compression-multi-rate-4-16-128",
        help="Model identifier passed to compress.py via --hf_model_name.",
    )
    parser.add_argument(
        "--compress_limit",
        type=int,
        default=10000,
        help="Limit argument forwarded to compress.py (default mirrors manual command).",
    )
    parser.add_argument(
        "--compress_context_batch_size",
        type=int,
        default=64,
        help="Context batch size forwarded to compress.py.",
    )
    parser.add_argument(
        "--compress_initial_offset",
        type=int,
        default=0,
        help="Initial --hf-offset value supplied to compress.py when auto continuation runs.",
    )
    parser.add_argument(
        "--compress_offset_increment",
        type=int,
        default=10000,
        help="Offset increment applied between consecutive compress.py invocations.",
    )
    parser.add_argument(
        "--compress_cuda_devices",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value to set when invoking compress.py.",
    )
    parser.add_argument(
        "--compress_hf_dataset_name",
        type=str,
        default="ms_marco",
        help="Optional dataset name forwarded to compress.py via --hf-dataset-name.",
    )
    parser.add_argument(
        "--compress_hf_dataset_config",
        type=str,
        default="v2.1",
        help="Optional dataset config forwarded to compress.py via --hf-dataset-config.",
    )
    parser.add_argument(
        "--compress_hf_dataset_split",
        type=str,
        default="train",
        help="Optional dataset split forwarded to compress.py via --hf-dataset-split.",
    )
    parser.add_argument(
        "--compress_additional_args",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Additional raw arguments appended to the compress.py invocation. Provide each "
            "flag or value as a separate token."
        ),
    )
    parser.add_argument(
        "--compress_use_hf_dataset",
        dest="compress_use_hf_dataset",
        action="store_true",
        help="Pass --use-hf-dataset to compress.py when auto continuation is active.",
    )
    parser.add_argument(
        "--compress_no_hf_dataset",
        dest="compress_use_hf_dataset",
        action="store_false",
        help="Omit --use-hf-dataset when invoking compress.py.",
    )
    parser.set_defaults(compress_use_hf_dataset=True)
    return parser.parse_args()

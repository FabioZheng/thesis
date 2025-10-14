import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    parser.add_argument("--doc_max_length", type=int, default=128)
    parser.add_argument("--compressor_model_name", type=str, default=None)
    parser.add_argument("--decoder_model_name", type=str,
                        default="neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_save_steps", type=int, default=1)
    parser.add_argument("--tc_ratio", type=float, default=0.5)
    parser.add_argument("--compression_rates", type=int, nargs='+', default=[32, 64, 128])  # Changed to accept list
    parser.add_argument("--lora", type=str, default='False')
    parser.add_argument("--experiment_folder", type=str, default='experiments_compress_real')
    parser.add_argument("--dataset_name_or_dir", type=str, default='openwebtext')
    parser.add_argument("--compression_linear_type", type=str, default='concat')
    parser.add_argument("--model_path", type=str,
                        default="neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic")
    parser.add_argument("--dataset_RAG", type=str,
                        default="squad")
    return parser.parse_args()
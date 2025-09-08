from transformers import Trainer, TrainingArguments
import datasets
import os
from custom_parser import get_args
from rouge import Rouge
from metrics import compute_rouge_scores, exact_match_score
import numpy as np
from utils import *
import random
from modeling_cocom import COCOM, COCOMConfig
from transformers.trainer_utils import get_last_checkpoint
import json
from datasets.fingerprint import Hasher
import shutil
import wandb
from accelerate import Accelerator
import torch
from cmab_agent import CompressionBanditAgent

random.seed(42)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_only_final = True  # Flag to control saving behavior

    def training_step(self, model, *args):
        inputs = args[0] if len(args) > 0 else None
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        self.accelerator.backward(loss)

        nan_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_gradients = True
                param.grad = None

        if nan_gradients:
            print("NaN gradient detected, skipping optimizer step.")
            return torch.tensor(0.0, requires_grad=False).to(loss.device)

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        """Override to save model in a cleaner format"""
        # Only save if it's the final save or explicitly requested
        if not self.save_only_final or self.state.global_step >= self.state.max_steps:
            super()._save_checkpoint(model, trial)


def compute_metrics(eval_pred, model, rouge):
    # Handle multiple logits from different compression rates
    logits_list, labels = eval_pred
    if isinstance(logits_list, tuple):
        logits_list = logits_list[0]

    # We'll use the first compression rate's logits for evaluation
    logits = logits_list[0] if isinstance(logits_list, list) else logits_list

    preds = np.argmax(logits, axis=-1)
    original_model = model.module if hasattr(model, 'module') else model
    ignore_positions = labels == -100

    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Print a short preview of prediction vs. reference during evaluation
    for pred_str, label_str in zip(preds_str[:2], labels_str[:2]):
        print('pred:', pred_str[:100])
        print('text:', label_str[:100])
        print()
    metrics = {}
    rouge_scores = compute_rouge_scores(rouge, preds_str, labels_str)
    em = exact_match_score(preds_str, labels_str)
    metrics.update(rouge_scores)
    metrics.update({'EM': em})
    return metrics


def pretrain_tokenize_function(examples,
                               compressor_tokenizer,
                               decoder_tokenizer,
                               tc_ratio=0.0,
                               compression_rates=[],  # Now accepts list
                               max_len=512):
    ae = random.random() >= tc_ratio
    # For multiple compression rates, we'll use the first one for tokenization
    # since the actual compression happens in the model
    compression_rate = compression_rates[0] if isinstance(compression_rates, list) else compression_rates

    if ae:
        return prepare_auto_encoding(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, max_len,
                                     train=True)
    else:
        return prepare_text_continuation(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, max_len,
                                         train=True)


def save_model_for_cmab(model, output_path):
    """
    Save model in a format compatible with train_cmab.py
    This saves the model state without optimizer/scheduler state to reduce size
    """
    print(f"Saving CMAB-compatible model to {output_path}")

    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Save just the model state without training artifacts
    if hasattr(model, 'module'):
        # Handle DataParallel/DistributedDataParallel
        model_to_save = model.module
    else:
        model_to_save = model

    # Save model using the standard save_pretrained method
    model_to_save.save_pretrained(output_path, safe_serialization=True)

    # Save additional metadata for CMAB compatibility
    cmab_metadata = {
        'compression_rates': model_to_save.compr_rates,
        'model_type': 'COCOM',
        'decoder_model_name': model_to_save.config.decoder_model_name,
        'compressor_model_name': getattr(model_to_save.config, 'compr_model_name', None),
        'generation_top_k': model_to_save.generation_top_k,
        'current_rate': getattr(model_to_save, 'current_rate', model_to_save.compr_rates[0])
    }

    with open(os.path.join(output_path, 'cmab_metadata.json'), 'w') as f:
        json.dump(cmab_metadata, f, indent=2)

    print(f"Model saved successfully. Size: {get_folder_size(output_path):.2f} GB")


def get_folder_size(folder_path):
    """Calculate folder size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 ** 3)  # Convert to GB


def main():
    accelerator = Accelerator()
    args = get_args()
    rouge = Rouge()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Use timestamp for unique folder names
    import time
    timestamp = int(time.time())
    folder_name = f'cocom_training_{timestamp}'
    output_dir = f"{args.experiment_folder}/{folder_name}"
    model_output_dir = output_dir + '/checkpoints/'
    final_model_dir = output_dir + '/final_model/'

    lora = args.lora.lower() == 'true'

    if accelerator.is_main_process:
        run_name = f'COCOM_{args.compression_rates}_{args.tc_ratio}_{lora}_{timestamp}'
        wandb.init(project="COCOM Pretrain", name=run_name)
        print(f"Experiment folder: {output_dir}")

    from itertools import islice

    # Load streaming dataset
    if os.path.exists(args.dataset_name_or_dir):
        dataset = datasets.load_from_disk(args.dataset_name_or_dir)
    else:
        dataset = datasets.load_dataset(args.dataset_name_or_dir)
    '''
    dataset_stream = datasets.load_dataset("openwebtext", split="train", streaming=True)


    def split_text_row(example):
        """
        Split the 'text' field into two halves:
        - First half → 'text'
        - Second half → 'next_text'
        """
        original_text = example["text"]
        midpoint = len(original_text) // 2
        return {
            "text": original_text[:midpoint],
            "next_text": original_text[midpoint:]
        }

    # Take subsets from the stream
    train_stream = islice(dataset_stream, 3000)
    test_stream = islice(dataset_stream, 32)

    # Materialize and split text fields
    train_data = [split_text_row(row) for row in train_stream]
    test_data = [split_text_row(row) for row in test_stream]

    # Turn into DatasetDict
    dataset = datasets.DatasetDict({
        'train': datasets.Dataset.from_list(train_data),
        'test': datasets.Dataset.from_list(test_data)
    })
    '''

    dataset['train'] = dataset['train'].select(range(min(10000, len(dataset['train']))))
    dataset['test'] = dataset['test'].select(range(min(32, len(dataset['test']))))

    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model_name,
        quantization='no',
        generation_top_k=1,
        sep=False,
        compr_model_name=args.compressor_model_name,
        compr_rates=args.compression_rates,  # Pass list of rates
        compr_linear_type=args.compression_linear_type,
        lora=lora,
    )

    model = COCOM(cfg)
    if accelerator.is_main_process:
        print(f"Model compression rates: {model.compr_rates}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Initialize current rate for compatibility
    model.current_rate = model.compr_rates[0]

    dataset = dataset.map(
        pretrain_tokenize_function,
        batched=True,
        fn_kwargs={
            "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
            "decoder_tokenizer": model.decoder_tokenizer,
            "tc_ratio": args.tc_ratio,
            "max_len": args.doc_max_length,
            "compression_rates": args.compression_rates
        }
    )

    dataset['train'] = dataset['train'].shuffle(seed=42)

    # Streamlined training arguments - minimal checkpointing
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=args.lr,
        eval_accumulation_steps=args.gradient_accumulation,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy='steps',
        save_total_limit=1,  # Keep only 1 checkpoint
        report_to=None,
        num_train_epochs=1,
        save_strategy="epoch",  # Save only at epoch end
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=4,
        do_eval=True,
        max_grad_norm=1.0,
        logging_steps=5,
        eval_steps=5,
        # Disable intermediate saves to reduce disk usage
        save_on_each_node=False,
        load_best_model_at_end=False,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=lambda e: compute_metrics(e, model=model, rouge=rouge)
    )

    # Set flag for final save only
    trainer.save_only_final = (args.num_save_steps == 1)

    trainer.create_optimizer_and_scheduler(num_training_steps=len(dataset['train']) //
                                                              (
                                                                          args.per_device_batch_size * torch.cuda.device_count() * args.gradient_accumulation))

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
    )

    # Save configuration
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/training_config.json', 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)

    # Training
    print("Starting training...")
    trainer.train()
    accelerator.wait_for_everyone()

    # Save final model in CMAB-compatible format
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)

        # Save the final model for CMAB training
        save_model_for_cmab(unwrapped_model, final_model_dir)

        # Clean up intermediate checkpoints to save space
        if os.path.exists(model_output_dir):
            shutil.rmtree(model_output_dir)
            print("Cleaned up intermediate checkpoints")

        # Print final folder size
        total_size = get_folder_size(output_dir)
        print(f"Final experiment folder size: {total_size:.2f} GB")
        print(f"Model saved to: {final_model_dir}")
        print(f"Use this path for train_cmab.py: --checkpoint {final_model_dir}")

    if accelerator.is_main_process and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
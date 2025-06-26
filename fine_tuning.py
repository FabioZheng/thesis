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
import shutil
import wandb
from accelerate import Accelerator
import torch
from datasets.fingerprint import Hasher

random.seed(42)

class FineTuningTrainer(Trainer):
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

def compute_metrics(eval_pred, model, rouge):
    logits_list, labels = eval_pred
    if isinstance(logits_list, tuple):
        logits_list = logits_list[0]

    logits = logits_list[0] if isinstance(logits_list, list) else logits_list

    preds = np.argmax(logits, axis=-1)
    original_model = model.module if hasattr(model, 'module') else model
    ignore_positions = labels == -100

    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    metrics = {}
    rouge_scores = compute_rouge_scores(rouge, preds_str, labels_str)
    em = exact_match_score(preds_str, labels_str)
    metrics.update(rouge_scores)
    metrics.update({'EM': em})
    return metrics

def prepare_squad_finetune_inputs(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, enc_max_len, dec_max_len):
    input_texts = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
    target_texts = [ans["text"][0] if len(ans["text"]) > 0 else "" for ans in examples["answers"]]

    num_mem_tokens = math.ceil(enc_max_len / compression_rate)

    enc = compressor_tokenizer(
        input_texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=enc_max_len
    )

    mem_prefix = decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_mem_tokens
    dec_inputs = [mem_prefix for _ in target_texts]

    dec = decoder_tokenizer(
        dec_inputs,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=dec_max_len,
        add_special_tokens=False
    )

    labels = decoder_tokenizer(
        target_texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=dec_max_len,
        add_special_tokens=False
    )['input_ids']

    labels[labels == decoder_tokenizer.pad_token_id] = -100

    return {
        "enc_input_ids": enc["input_ids"],
        "enc_attention_mask": enc["attention_mask"],
        "dec_input_ids": dec["input_ids"],
        "dec_attention_mask": dec["attention_mask"],
        "labels": labels
    }


def qa_tokenize_function(examples,
                          compressor_tokenizer,
                          decoder_tokenizer,
                          compression_rates=[64],
                          max_len=512,
                          tc_ratio=0.0):
    compression_rate = compression_rates[0] if isinstance(compression_rates, list) else compression_rates
    return prepare_squad_finetune_inputs(
        examples,
        compressor_tokenizer,
        decoder_tokenizer,
        compression_rate,
        enc_max_len=max_len,
        dec_max_len=128  # ← you can also parametrize this via args if needed
    )


def main():
    accelerator = Accelerator()
    args = get_args()
    rouge = Rouge()

    folder_name = f'{Hasher.hash(str(args))}'
    output_dir = f"{args.experiment_folder}/tmp_{folder_name}"
    model_output_dir = output_dir + '/train/'
    lora = args.lora.lower() == 'true'

    if accelerator.is_main_process:
        run_name = f'{args.compressor_model_name}_{args.decoder_model_name}_{args.compression_rates}_QA_{lora}_{args.lr}_{folder_name}'
        wandb.init(project="COCOM QA Finetune", name=run_name)

    dataset = datasets.load_dataset("squad")

    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model_name,
        quantization='no',
        generation_top_k=1,
        sep=False,
        compr_model_name=args.compressor_model_name,
        compr_rates=args.compression_rates,
        compr_linear_type=args.compression_linear_type,
        lora=lora,
    )

    model = COCOM(cfg)

    if accelerator.is_main_process:
        print(model)

    # Map the tokenization cleanly
    dataset["train"] = dataset["train"].map(
        qa_tokenize_function,
        batched=True,
        fn_kwargs={
            "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
            "decoder_tokenizer": model.decoder_tokenizer,
            "compression_rates": args.compression_rates,
            "max_len": args.doc_max_length,
            "tc_ratio": 0.0,
        }
    )

    dataset["validation"] = dataset["validation"].map(
        qa_tokenize_function,
        batched=True,
        fn_kwargs={
            "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
            "decoder_tokenizer": model.decoder_tokenizer,
            "compression_rates": args.compression_rates,
            "max_len": args.doc_max_length,
            "tc_ratio": 0.0,
        }
    )


    dataset["train"] = dataset["train"].shuffle(seed=42)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=args.lr,
        eval_accumulation_steps=args.gradient_accumulation,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(args.per_device_batch_size // 4, 1),
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps",
        save_total_limit=10,
        report_to=None,
        save_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=4,
        do_eval=True,
        max_grad_norm=1.0,
    )

    total_batch_size = args.per_device_batch_size * torch.cuda.device_count() * args.gradient_accumulation
    total_steps = len(dataset["train"]) // total_batch_size
    save_steps = max(total_steps // args.num_save_steps, 1)
    training_args.save_steps = save_steps
    training_args.logging_steps = 10
    training_args.eval_steps = 10

    trainer = FineTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=lambda e: compute_metrics(e, model=model, rouge=rouge)
    )

    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
    )

    checkpoint = None
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    if accelerator.is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f"{output_dir}/config.json", "w") as json_file:
            json.dump(vars(args), json_file, indent=4)

    trainer.train(resume_from_checkpoint=checkpoint)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(f"{output_dir}/last_model/")
        final_output_dir = f"{args.experiment_folder}/{folder_name}"
        shutil.move(output_dir, final_output_dir)

if __name__ == "__main__":
    main()

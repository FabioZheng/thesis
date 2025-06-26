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

random.seed(42)


class CustomTrainer(Trainer):
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

    '''
    for pre, lab in zip(preds_str[:2], labels_str[:2]):
        print('label: ', lab)
        print('pred: ', pre)
        print()
    print('_' * 15)
    '''
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
                               compression_rates=[64],  # Now accepts list
                               max_len=512):
    texts = examples["text"]
    # Fake continuation = repeat same text
    examples = {
        "text": texts,
        "next_text": texts
    }

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


def main():
    accelerator = Accelerator()
    args = get_args()
    rouge = Rouge()

    folder_name = f'{Hasher.hash(str(args))}'
    output_dir = f"{args.experiment_folder}/tmp_{folder_name}"
    model_output_dir = output_dir + '/train/'
    lora = args.lora.lower() == 'true'

    if accelerator.is_main_process:
        run_name = f'{args.compressor_model_name}_{args.decoder_model_name}_{args.compression_rates}_{args.tc_ratio}_{lora}_{args.lr}_{folder_name}'
        wandb.init(project="COCOM Pretrain", name=run_name)

    dataset = datasets.load_dataset("openwebtext", split="train", streaming=True)

    from itertools import islice

    # Split manually
    train_stream = islice(dataset, 10000)
    test_stream = islice(dataset, 1000)

    # Turn streams into datasets
    dataset = {
        'train': datasets.Dataset.from_list(list(train_stream)),
        'test': datasets.Dataset.from_list(list(test_stream))
    }

    dataset['train'] = dataset['train'].select(range(min(100000, len(dataset['train']))))
    dataset['test'] = dataset['test'].select(range(min(32, len(dataset['test']))))

    num_proc = 2
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
        print(model)

    dataset = dataset.map(pretrain_tokenize_function, batched=True,
                          fn_kwargs={
                              "compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
                              "decoder_tokenizer": model.decoder_tokenizer,
                              "tc_ratio": args.tc_ratio,
                              "max_len": args.doc_max_length,
                              "compression_rates": args.compression_rates})

    dataset['train'] = dataset['train'].shuffle(seed=42)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        learning_rate=args.lr,
        eval_accumulation_steps=args.gradient_accumulation,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=max(args.per_device_batch_size // 4, 1),
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy='steps',
        save_total_limit=10,
        report_to=None,
        num_train_epochs=1,
        save_strategy="steps",
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=4,
        do_eval=True,
        max_grad_norm=1.0,
    )

    total_batch_size = args.per_device_batch_size * torch.cuda.device_count() * args.gradient_accumulation
    total_steps = len(dataset['train']) // total_batch_size
    save_steps = max(total_steps // args.num_save_steps, 1)
    training_args.save_steps = save_steps
    training_args.logging_steps = 10
    training_args.eval_steps = 10


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
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

    print(f"Loaded from the checkpoint: {checkpoint}")

    if accelerator.is_main_process:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(f'{output_dir}/config.json', 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)

    trainer.train(resume_from_checkpoint=checkpoint)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(f'{output_dir}/last_model/')
        final_output_dir = f"{args.experiment_folder}/{folder_name}"
        shutil.move(output_dir, final_output_dir)


if __name__ == "__main__":
    main()
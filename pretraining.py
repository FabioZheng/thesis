from transformers import Trainer, TrainingArguments
import datasets
import os
from rouge import Rouge
from metrics import compute_rouge_scores, exact_match_score
import numpy as np
import argparse
from utils import *
import random
from modeling_cocom import COCOM, COCOMConfig
from transformers.trainer_utils import get_last_checkpoint
import json
from datasets.fingerprint import Hasher
import shutil
import wandb
random.seed(42)
from accelerate import Accelerator
import torch

class CustomTrainer(Trainer):
    class CustomTrainer(Trainer):
        def training_step(self, model, *args):
            # We only need the first two arguments (self, model) and inputs (args[0])
            inputs = args[0] if len(args) > 0 else None

            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()

            self.accelerator.backward(loss)

            # NaN gradient checking
            nan_gradients = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    nan_gradients = True
                    param.grad = None

            if nan_gradients:
                print("NaN gradient detected, skipping optimizer step.")
                return torch.tensor(0.0, requires_grad=False).to(loss.device)

            return loss.detach() / self.args.gradient_accumulation_steps

# compute metrics do not work properly at the moment, because the first n tokens, i.e memory tokens are decoded and taken into account in evaluation.
def compute_metrics(eval_pred, model, rouge):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Check if logits are wrapped in a tuple
        logits = logits[0]  # Adjust this to access the correct tuple element

    preds = np.argmax(logits, axis=-1)
    original_model = model.module if hasattr(model, 'module') else model
    ignore_positions = labels == -100

    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print examples
    for pre, lab in zip(preds_str[:2], labels_str[:2]):
        print('label: ', lab)
        print('pred: ', pre)
        print()
    print('_'*15)
    metrics = {}
    rouge_scores = compute_rouge_scores(rouge, preds_str, labels_str)
    em = exact_match_score(preds_str, labels_str)
    metrics.update(rouge_scores)
    metrics.update({'EM': em})
    return metrics

def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    # Model Configuration
    parser.add_argument("--doc_max_length", type=int, default=128, help="Maximum document length")
    parser.add_argument("--compressor_model_name", type=str, default=None)
    parser.add_argument("--decoder_model_name", type=str, default="meta-llama/Llama-2-7b-hf")

    # Training Configuration
    parser.add_argument("--per_device_batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="warmup ratio")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_save_steps", type=int, default=30, help="How Many Models to Save")

    # Compression Configuration
    parser.add_argument("--tc_ratio", type=float, default=0.5)
    parser.add_argument("--compression_rate", type=int, default=16)
    parser.add_argument("--lora", type=str, default='False')
    parser.add_argument("--experiment_folder", type=str, default='experiments_compress_real')
    parser.add_argument("--dataset_name_or_dir", type=str, default='wshuai190/fineweb-sample-10BT-128')
    parser.add_argument("--compression_linear_type", type=str, default='concat')

    args = parser.parse_args()
    return args

def pretrain_tokenize_function(examples,
                                compressor_tokenizer,
                                decoder_tokenizer,
                                tc_ratio=0.0,
                                compression_rate=1,
                                max_len=512):

        ae = random.random() >= tc_ratio
        if ae:
            training_input = prepare_auto_encoding(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, max_len, train=True)
        else:
            training_input  = prepare_text_continuation(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, max_len, train=True)
        return training_input




def main():
    accelerator = Accelerator()

    args = get_args()
    rouge = Rouge()

    folder_name = f'{Hasher.hash(str(args))}'
    output_dir = f"{args.experiment_folder}/tmp_{folder_name}"

    model_output_dir = output_dir + '/train/'
    lora = args.lora.lower() == 'true'
    if accelerator.is_main_process:
        run_name = f'{args.compressor_model_name}_{args.decoder_model_name}_{args.compression_rate}_{args.tc_ratio}_{lora}_{args.lr}_{folder_name}'
        wandb.init(project="COCOM Pretrain", name=run_name)

    if os.path.exists(args.dataset_name_or_dir):
        dataset = datasets.load_from_disk(args.dataset_name_or_dir)
    else:
        dataset = datasets.load_dataset(args.dataset_name_or_dir)


    num_proc = 24
    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model_name,
        max_new_tokens=128,
        quantization='no',
        generation_top_k=1,
        sep=False,
        compr_model_name=args.compressor_model_name,
        compr_rate=args.compression_rate,
        compr_linear_type=args.compression_linear_type,
        lora=lora,
    )


    model = COCOM(cfg)


    if accelerator.is_main_process:
        print(model)

    dataset['train'] = dataset['train'].select(range(10000000))
    dataset['test'] = dataset['test'].select(range(64))


    dataset = dataset.map(pretrain_tokenize_function, num_proc=num_proc, batched=True,
                                            fn_kwargs={"compressor_tokenizer": model.compr.tokenizer if model.compr else model.decoder_tokenizer,
                                                        "decoder_tokenizer": model.decoder_tokenizer,
                                                        "tc_ratio": args.tc_ratio,
                                                        "max_len": args.doc_max_length,
                                                        "compression_rate": args.compression_rate}
                                                        )

    dataset['train'] = dataset['train'].shuffle(seed=42)



    # for index, item in enumerate(dataset['train']):
    #     validate_data_item(item)
        # if index > 100:  # Limit to first 100 items for quick checking
        #     break


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
    training_args.logging_steps = save_steps // 5
    training_args.eval_steps = save_steps // 5


    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=lambda e: compute_metrics(e, model=model, rouge=rouge)
    )

    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, trainer.optimizer,
                                                                              trainer.get_train_dataloader(),
                                                                              trainer.get_eval_dataloader())


    checkpoint = None

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    # save config
    # wait all process to finish

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






# class MyTrainer(Trainer):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         # def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
#         #         with torch.no_grad():
#         #             return model.prediction_step(**inputs)

#         def save_model(self, output_dir=None, _internal_call=False):
#             # Save the model
#             super().save_model(output_dir=output_dir, _internal_call=_internal_call)
#             if self.is_world_p

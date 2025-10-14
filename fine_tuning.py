import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import datasets
import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments
from analyse.retrieval import CosineRetriever, TextEmbedder
from fine_tuning_parser import get_fine_tuning_args
from datasets.fingerprint import Hasher
from metrics import compute_rouge_scores, exact_match_score, f1_score
from modeling_cocom import COCOM, COCOMConfig
from rouge import Rouge
from transformers.trainer_utils import get_last_checkpoint

import wandb

random.seed(42)


def _to_str_doc_id(doc_id: object) -> str:
    try:
        if isinstance(doc_id, bytes):
            return doc_id.decode("utf-8")
        if isinstance(doc_id, (np.integer, np.int64, np.int32)):
            return str(int(doc_id))
        return str(doc_id)
    except Exception:
        return str(doc_id)


def load_context_store(path: str) -> Dict[str, torch.Tensor]:
    raw_contexts = torch.load(path, map_location="cpu")
    context_store: Dict[str, torch.Tensor] = {}
    for doc_id, payload in raw_contexts.items():
        key = _to_str_doc_id(doc_id)
        context_tensor = payload.get("context") if isinstance(payload, dict) else payload
        if not isinstance(context_tensor, torch.Tensor):
            context_tensor = torch.as_tensor(context_tensor)
        # Compressors output tensors of shape (1, mem_tokens, hidden)
        if context_tensor.dim() == 3 and context_tensor.size(0) == 1:
            context_tensor = context_tensor.squeeze(0)
        context_store[key] = context_tensor.to(torch.float32)
    return context_store


def load_retriever(embeddings_path: str, docs_path: Optional[str]) -> CosineRetriever:
    with np.load(embeddings_path) as embeddings_file:
        doc_ids = [_to_str_doc_id(doc_id) for doc_id in embeddings_file["doc_ids"]]
        embeddings = embeddings_file["embeddings"]

    docs: Dict[str, Dict[str, str]] = {}
    if docs_path and os.path.exists(docs_path):
        with open(docs_path, "r", encoding="utf-8") as handle:
            docs = json.load(handle)

    store: Dict[str, Dict[str, str]] = {}
    for doc_id in doc_ids:
        doc_payload = docs.get(doc_id)
        if doc_payload is None:
            try:
                numeric_key = str(int(float(doc_id)))
            except Exception:
                numeric_key = None
            if numeric_key is not None:
                doc_payload = docs.get(numeric_key)
        text = ""
        if isinstance(doc_payload, dict):
            text = doc_payload.get("text", "")
        store[doc_id] = {"text": text}

    return CosineRetriever(embeddings=embeddings, doc_ids=doc_ids, store=store)


def _extract_answer(example: Dict) -> str:
    answers = example.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text") or answers.get("answers")
        if texts and isinstance(texts, (list, tuple)):
            first = texts[0]
            if isinstance(first, dict):
                return first.get("text", "")
            return first
    if isinstance(answers, (list, tuple)) and answers:
        first = answers[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return first
    return example.get("answer", "") or ""


class RAGFineTuneDataset(Dataset):
    def __init__(
        self,
        qa_split: datasets.arrow_dataset.Dataset,
        retriever: CosineRetriever,
        query_embedder: TextEmbedder,
        context_store: Dict[str, torch.Tensor],
        top_k: int,
    ) -> None:
        self.context_store = context_store
        self.top_k = top_k
        self.samples: List[Dict[str, object]] = []

        skipped_due_to_contexts = 0
        for example in tqdm(qa_split, desc="Preparing RAG QA split"):
            question = example.get("question") or example.get("query") or ""
            if not question:
                continue
            answer = _extract_answer(example)
            hits = retriever.search(question, query_embedder, k=top_k)
            doc_ids: List[str] = []
            for doc_id, _score, _text in hits:
                key = _to_str_doc_id(doc_id)
                if key in context_store:
                    doc_ids.append(key)
            if len(doc_ids) < top_k:
                skipped_due_to_contexts += 1
                continue
            self.samples.append({
                "question": question,
                "answer": answer,
                "doc_ids": doc_ids[:top_k],
            })

        retained = len(self.samples)
        if retained == 0:
            raise ValueError("No training examples available after retrieval. Check retrieval resources.")
        print(
            f"Prepared {retained} examples for RAG fine-tuning (skipped {skipped_due_to_contexts} without sufficient contexts)."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]
        doc_ids: Iterable[str] = sample["doc_ids"]

        context_tensors: List[torch.Tensor] = []
        for doc_id in doc_ids:
            context_tensor = self.context_store[doc_id]
            tensor = context_tensor
            if tensor.dim() == 3 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            tensor = tensor.to(dtype=torch.float32)
            context_tensors.append(tensor)

        if not context_tensors:
            raise ValueError("Missing context tensors for retrieved document ids.")

        max_mem = max(tensor.size(0) for tensor in context_tensors)
        hidden = context_tensors[0].size(1)
        padded_contexts: List[torch.Tensor] = []
        for tensor in context_tensors:
            if tensor.size(1) != hidden:
                raise ValueError("Inconsistent hidden size across context tensors.")
            if tensor.size(0) < max_mem:
                pad = torch.zeros((max_mem - tensor.size(0), hidden), dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            padded_contexts.append(tensor)
        stacked = torch.stack(padded_contexts, dim=0)  # (top_k, mem, hidden)

        return {
            "question": question,
            "answer": answer,
            "context_embeddings": stacked,
        }


@dataclass
class RAGDataCollator:
    tokenizer: any
    top_k: int
    decoder_max_length: int

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        context_embeddings = [feature["context_embeddings"] for feature in features]
        questions = [feature["question"] for feature in features]
        answers = [feature["answer"] for feature in features]

        top_k = context_embeddings[0].size(0)
        if top_k != self.top_k:
            raise ValueError(
                f"Batch contexts use {top_k} documents but collator was initialised for {self.top_k}."
            )
        if any(tensor.size(0) != top_k for tensor in context_embeddings):
            raise ValueError("All samples must have the same number of retrieved contexts.")

        max_mem = max(tensor.size(1) for tensor in context_embeddings)
        hidden = context_embeddings[0].size(2)

        padded_contexts: List[torch.Tensor] = []
        for tensor in context_embeddings:
            if tensor.size(1) < max_mem:
                pad = torch.zeros((top_k, max_mem - tensor.size(1), hidden), dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=1)
            padded_contexts.append(tensor)
        context_batch = torch.stack(padded_contexts, dim=0)

        mem_tokens = self.tokenizer.mem_token * max_mem if max_mem > 0 else ""
        mem_block = mem_tokens * top_k
        prompts = [
            f"{self.tokenizer.bos_token}{mem_block}[INST]{question}\n[/INST]\n"
            for question in questions
        ]

        dec_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            max_length=self.decoder_max_length,
            truncation=True,
            add_special_tokens=False,
        )

        labels = self.tokenizer(
            answers,
            return_tensors="pt",
            padding="max_length",
            max_length=self.decoder_max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch = {
            "dec_input_ids": dec_inputs["input_ids"],
            "dec_attention_mask": dec_inputs["attention_mask"],
            "labels": labels,
            "context_embeddings": context_batch,
        }
        return batch


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
        context_embeddings = inputs.get("context_embeddings")
        if context_embeddings is not None:
            outputs = model.forward_with_context_embeddings(
                context_embeddings=context_embeddings,
                dec_input_ids=inputs["dec_input_ids"],
                dec_attention_mask=inputs["dec_attention_mask"],
                labels=inputs.get("labels"),
            )
        else:
            outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred, model, rouge):
    logits_list, labels = eval_pred
    if isinstance(logits_list, tuple):
        logits_list = logits_list[0]

    logits = logits_list[0] if isinstance(logits_list, list) else logits_list

    preds = np.argmax(logits, axis=-1)
    original_model = model.module if hasattr(model, "module") else model
    ignore_positions = labels == -100

    labels = labels.copy()
    preds = preds.copy()
    labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
    preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

    preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    metrics = {}
    rouge_scores = compute_rouge_scores(rouge, preds_str, labels_str)
    em = exact_match_score(preds_str, labels_str)
    f1 = f1_score(preds_str, labels_str)
    metrics.update(rouge_scores)
    metrics.update({"EM": em, "F1": f1})
    return metrics


def main():
    accelerator = Accelerator()
    args = get_fine_tuning_args()
    rouge = Rouge()

    folder_name = f"{Hasher.hash(str(args))}"
    output_dir = f"{args.experiment_folder}/tmp_{folder_name}"
    model_output_dir = output_dir + "/train/"
    lora = args.lora.lower() == "true"

    if accelerator.is_main_process:
        run_name = f"{args.compressor_model_name}_{args.decoder_model_name}_{args.compression_rates}_QA_{lora}_{args.lr}_{folder_name}"
        wandb.init(project="COCOM QA Finetune", name=run_name)

    dataset = datasets.load_dataset(args.dataset_RAG)

    context_store = load_context_store(args.rag_contexts_path)
    retriever = load_retriever(args.rag_embeddings_path, args.rag_docs_path)
    query_embedder = TextEmbedder(
        model_name=args.retriever_model_name,
        batch_size=args.retriever_batch_size,
        device=args.retriever_device,
        normalize=True,
    )

    train_dataset = RAGFineTuneDataset(
        dataset["train"],
        retriever,
        query_embedder,
        context_store,
        top_k=args.retriever_top_k,
    )
    eval_dataset = RAGFineTuneDataset(
        dataset["validation"],
        retriever,
        query_embedder,
        context_store,
        top_k=args.retriever_top_k,
    )

    del query_embedder

    cfg = COCOMConfig(
        decoder_model_name=args.decoder_model_name,
        quantization="no",
        generation_top_k=args.retriever_top_k,
        sep=False,
        compr_model_name=args.compressor_model_name,
        compr_rates=args.compression_rates,
        compr_linear_type=args.compression_linear_type,
        lora=lora,
    )

    model = COCOM(cfg)
    model.generation_top_k = args.retriever_top_k

    if accelerator.is_main_process:
        print(model)

    data_collator = RAGDataCollator(
        tokenizer=model.decoder_tokenizer,
        top_k=args.retriever_top_k,
        decoder_max_length=args.decoder_max_length,
    )

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
        remove_unused_columns=False,
    )

    world_size = max(accelerator.num_processes, 1)
    total_batch_size = args.per_device_batch_size * world_size * args.gradient_accumulation
    steps_per_epoch = max(math.ceil(len(train_dataset) / total_batch_size), 1)
    total_steps = steps_per_epoch
    save_steps = max(total_steps // args.num_save_steps, 1)
    training_args.save_steps = save_steps
    training_args.logging_steps = 10
    training_args.eval_steps = 10

    trainer = FineTuningTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=lambda e: compute_metrics(e, model=model, rouge=rouge),
    )

    trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        trainer.optimizer,
        trainer.get_train_dataloader(),
        trainer.get_eval_dataloader(),
    )

    trainer.model = model
    trainer.optimizer = optimizer
    trainer._train_dataloader = train_dataloader
    trainer._eval_dataloader = eval_dataloader

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
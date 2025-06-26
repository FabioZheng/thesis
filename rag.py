import os
import shutil
import json
import time
from tqdm import tqdm

import torch

from modules.retrieve import Retrieve
from modules.rerank import Rerank
from modules.dataset_processor import ProcessDatasets
from modules.metrics import RAGMetrics
from modeling_cocom import COCOM, COCOMConfig

class COCOMRAG:
    def __init__(self,
                 retriever=None,
                 reranker=None,
                 runs_folder='runs/',
                 run_name=None,
                 dataset=None,
                 processing_num_proc=1,
                 dataset_folder='datasets/',
                 index_folder='indexes/',
                 experiments_folder='experiments/',
                 retrieve_top_k=1,
                 rerank_top_k=1,
                 generation_top_k=1,
                 config=None,
                 debug=False,
                 **kwargs):

        self.dataset_folder = dataset_folder
        self.experiments_folder = experiments_folder
        self.runs_folder = runs_folder
        self.index_folder = index_folder
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.generation_top_k = generation_top_k
        self.config = config
        self.debug = debug

        os.makedirs(self.experiments_folder, exist_ok=True)
        self.run_name = run_name or f"run_{int(time.time())}"
        self.experiment_folder = os.path.join(self.experiments_folder, self.run_name)
        os.makedirs(self.experiment_folder, exist_ok=True)

        self.datasets = ProcessDatasets.process(dataset, out_folder=self.dataset_folder, num_proc=processing_num_proc, overwrite=True, debug=debug)
        self.metrics = {"train": RAGMetrics, "dev": RAGMetrics, "test": None}

        self.retriever = Retrieve(**retriever) if retriever else None
        self.reranker = Rerank(**reranker) if reranker else None

        cocom_cfg = COCOMConfig(**config.generator.init_args)
        self.generator_model = COCOM(cocom_cfg)

        print("COCOMRAG initialized. Retriever:", retriever is not None, ", Reranker:", reranker is not None)

    def save_json(self, data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def write_trec(self, path, query_ids, doc_ids, scores):
        with open(path, 'w') as f:
            for qid, dids, scs in zip(query_ids, doc_ids, scores):
                for did, score in zip(dids, scs):
                    f.write(f"{qid} Q0 {did} 0 {score} run\n")

    def load_trec(self, path):
        query_ids, doc_ids, scores = [], [], []
        with open(path, 'r') as f:
            lines = f.readlines()
            current_qid, current_dids, current_scores = None, [], []
            for line in lines:
                parts = line.strip().split()
                qid, did, score = parts[0], parts[2], float(parts[4])
                if qid != current_qid:
                    if current_qid is not None:
                        query_ids.append(current_qid)
                        doc_ids.append(current_dids)
                        scores.append(current_scores)
                    current_qid, current_dids, current_scores = qid, [], []
                current_dids.append(did)
                current_scores.append(score)
            if current_qid is not None:
                query_ids.append(current_qid)
                doc_ids.append(current_dids)
                scores.append(current_scores)
        return query_ids, doc_ids, scores

    def prepare_dataset_from_ids(self, dataset, query_ids, doc_ids):
        queries = dataset['query']
        docs = dataset['doc']
        id_to_doc = {doc['id']: doc for doc in docs}
        prepared = []
        for qid, dids in zip(query_ids, doc_ids):
            context = [id_to_doc[did]['text'] for did in dids if did in id_to_doc]
            question = queries[qid]['text']
            answer = queries[qid]['answers'][0] if queries[qid]['answers'] else ""
            prepared.append({'ctxs': context, 'query': question, 'answers': [answer]})
        return prepared


    def retrieve(self, dataset, dataset_split):
        ranking_file = os.path.join(self.runs_folder, f"{dataset_split}_retrieve.trec")
        if not os.path.exists(ranking_file):
            print("Running retrieval...")
            out = self.retriever.retrieve(dataset, None, None, self.retrieve_top_k)
            self.write_trec(ranking_file, out['q_id'], out['doc_id'], out['score'])
        return self.load_trec(ranking_file)

    def rerank(self, dataset, dataset_split, query_ids, doc_ids):
        reranking_file = os.path.join(self.runs_folder, f"{dataset_split}_rerank.trec")
        if not os.path.exists(reranking_file):
            print("Running reranking...")
            rerank_dataset = self.prepare_dataset_from_ids(dataset, query_ids, doc_ids)
            out = self.reranker.eval(rerank_dataset)
            self.write_trec(reranking_file, out['q_id'], out['doc_id'], out['score'])
        return self.load_trec(reranking_file)

    def prepare_dataset_from_ids(self, dataset, query_ids, doc_ids):
        queries = dataset['query']
        docs = dataset['doc']
        id_to_doc = {doc['id']: doc for doc in docs}
        prepared = []
        for qid, dids in zip(query_ids, doc_ids):
            context = [id_to_doc[did]['text'] for did in dids if did in id_to_doc]
            question = queries[qid]['text']
            answer = queries[qid]['answers'][0] if queries[qid]['answers'] else ""
            prepared.append({'ctxs': context, 'query': question, 'answers': [answer]})
        return prepared

    def generate(self, dataset_split):
        dataset = self.datasets[dataset_split]
        query_ids, doc_ids, _ = self.retrieve(dataset, dataset_split)
        if self.reranker:
            query_ids, doc_ids, _ = self.rerank(dataset, dataset_split, query_ids, doc_ids)
        doc_ids = [d[:self.generation_top_k] for d in doc_ids]

        prepared_dataset = self.prepare_dataset_from_ids(dataset, query_ids, doc_ids)

        questions, predictions, references = [], [], []

        for sample in tqdm(prepared_dataset, desc="Generating"):
            contexts = sample['ctxs']
            question = sample['query']
            answer = sample['answers'][0]
            generated = self.generator_model.generate_from_text([contexts], [question])[0]

            questions.append(question)
            predictions.append(generated)
            references.append(answer)

        output = {
            'questions': questions,
            'predictions': predictions,
            'references': references
        }

        self.save_json(output, os.path.join(self.experiment_folder, f"{dataset_split}_generation.json"))
        print("Saved generations to", os.path.join(self.experiment_folder, f"{dataset_split}_generation.json"))

    def eval_metrics(self, dataset_split):
        with open(os.path.join(self.experiment_folder, f"{dataset_split}_generation.json"), 'r') as f:
            data = json.load(f)

        metrics = self.metrics[dataset_split].compute(
            predictions=data['predictions'],
            references=data['references'],
            questions=data['questions']
        )

        self.save_json(metrics, os.path.join(self.experiment_folder, f"{dataset_split}_metrics.json"))
        print("Saved evaluation metrics.")


    def finetune(self, dataset_split='train', output_dir=None, num_train_epochs=3, batch_size=2, lr=2e-5):
        from transformers import TrainingArguments, Trainer
        from torch.utils.data import DataLoader

        dataset = self.datasets[dataset_split]
        query_dataset_name = dataset['query'].name
        doc_dataset_name = dataset['doc'].name

        if self.retriever:
            query_ids, doc_ids, _ = self.retrieve(dataset, query_dataset_name, doc_dataset_name, dataset_split,
                                                  self.retrieve_top_k)
        else:
            query_ids, doc_ids = None, None

        if self.reranker:
            query_ids, doc_ids, _ = self.rerank(dataset, query_dataset_name, doc_dataset_name, dataset_split, query_ids,
                                                doc_ids, self.rerank_top_k)

        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids is not None else doc_ids

        finetune_dataset = self.prepare_dataset_from_ids(dataset, query_ids, doc_ids, multi_doc=True)

        class FineTuneDataset(torch.utils.data.Dataset):
            def __init__(self, data, tokenizer, generation_top_k):
                self.data = data
                self.tokenizer = tokenizer
                self.generation_top_k = generation_top_k

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                sample = self.data[idx]
                contexts = sample['ctxs']
                question = sample['query']
                answer = sample['answers'][0] if 'answers' in sample and sample['answers'] else ''

                input_texts = [ctx for ctx in contexts]
                target_text = answer

                inputs = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt',
                                        pad_to_multiple_of=self.generation_top_k)
                outputs = self.tokenizer(target_text, truncation=True, return_tensors='pt')

                return {
                    'enc_input_ids': inputs['input_ids'].squeeze(0),
                    'enc_attention_mask': inputs['attention_mask'].squeeze(0),
                    'dec_input_ids': outputs['input_ids'].squeeze(0)[:-1],
                    'labels': outputs['input_ids'].squeeze(0)[1:],
                    'dec_attention_mask': outputs['attention_mask'].squeeze(0)[1:]
                }

        train_dataset = FineTuneDataset(finetune_dataset, self.generator_model.decoder_tokenizer, self.generation_top_k)

        def collate_fn(batch):
            enc_input_ids = torch.nn.utils.rnn.pad_sequence([item['enc_input_ids'] for item in batch], batch_first=True,
                                                            padding_value=self.generator_model.decoder_tokenizer.pad_token_id)
            enc_attention_mask = torch.nn.utils.rnn.pad_sequence([item['enc_attention_mask'] for item in batch],
                                                                 batch_first=True, padding_value=0)
            dec_input_ids = torch.nn.utils.rnn.pad_sequence([item['dec_input_ids'] for item in batch], batch_first=True,
                                                            padding_value=self.generator_model.decoder_tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence([item['labels'] for item in batch], batch_first=True,
                                                     padding_value=-100)
            dec_attention_mask = torch.nn.utils.rnn.pad_sequence([item['dec_attention_mask'] for item in batch],
                                                                 batch_first=True, padding_value=0)

            return {
                'enc_input_ids': enc_input_ids,
                'enc_attention_mask': enc_attention_mask,
                'dec_input_ids': dec_input_ids,
                'dec_attention_mask': dec_attention_mask,
                'labels': labels,
            }

        args = TrainingArguments(
            output_dir=output_dir or f'{self.experiment_folder}/cocom_finetune/',
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            evaluation_strategy="no",
            save_strategy="epoch",
            logging_steps=10,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = Trainer(
            model=self.generator_model,
            args=args,
            train_dataset=train_dataset,
            tokenizer=self.generator_model.decoder_tokenizer,
            data_collator=collate_fn,
        )

        trainer.train()

        # Save the final model
        trainer.save_model(args.output_dir)

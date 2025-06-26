
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Rerank:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def eval(self, dataset, return_embeddings=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        qids, docids, scores = [], [], []
        for sample in tqdm(dataset, desc='Reranking'):
            question = sample['query']
            ctxs = sample['ctxs']
            inputs = self.tokenizer([ (question, ctx) for ctx in ctxs ], return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(-1)
            sorted_indices = torch.argsort(logits, descending=True)
            qids.append(sample['id'])
            docids.append([ctxs[i] for i in sorted_indices])
            scores.append([logits[i].item() for i in sorted_indices])
        return {'q_id': qids, 'doc_id': docids, 'score': scores}
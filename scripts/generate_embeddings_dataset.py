from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from torch.utils.data import random_split
import torch
import datasets
import os
from tqdm import tqdm
from peft import PeftModel, PeftConfig



def load_embeddings(input_path):
    embds = list()
    for i, emb_file in tqdm(enumerate(os.listdir(input_path)), desc='Loading embeddings...'):
        if i > 1:
            break
        if emb_file.endswith('.pt'):
            emb = torch.load(f'{input_path}/{emb_file}')
            embds.append(emb)

    embds = torch.concat(embds)
    return embds

embeddings_path = '/beegfs/scratch/user/drau/research/reml/indexes/kilt-100w_doc_castorini_repllama-v1-7b-lora-passage'
dataset_path = '/beegfs/scratch/user/drau/research/reml/datasets/kilt-100w_full/'

model_name = 'castorini/repllama-v1-7b-lora-passage'
input_embeddings = load_embeddings(embeddings_path).numpy()
docs_dataset = datasets.load_from_disk(dataset_path).select(range(len(input_embeddings)))
docs = docs_dataset['content']

# Initialize the model and tokenizer


dataset = datasets.Dataset.from_dict({'labels': docs, 'inputs_embeds': input_embeddings})

# Add the new feature column to the dataset
dataset.save_to_disk('kilt-100w_emb_auto_reg_pretrain_task.hf')

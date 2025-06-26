import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

class DotProduct:

    @staticmethod
    def sim(query_embds, doc_embds):
        return torch.mm(query_embds, doc_embds.t())

class CosineSim:

    @staticmethod
    def sim(query_embds, doc_embds):
        query_embds = query_embds / (torch.norm(query_embds, dim=-1, keepdim=True) + 1e-9)
        doc_embds = doc_embds / (torch.norm(doc_embds, dim=-1, keepdim=True) + 1e-9)
        return torch.mm(query_embds, doc_embds.t())
    
retrieval_file_path = 'run.msmarco-passage.bm25.topics.dl20_54.txt_top_100'
reranked_file_path = f'{retrieval_file_path}.reranked.cse.trec'
model_name = '/projects/0/gusr0546/research/dec_from_emb/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_100_emb_1/checkpoint-107000/'
model_name = '/projects/0/gusr0546/research/dec_from_emb/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_100_emb_1_contrastive_True_scalar_1.0/checkpoint-32000'
tokenizer_name = 'tinyllama/TinyLlama-1.1B-intermediate-step-1431k-3T'
query_dataset_path = 'msmarco-test2020-queries_54.tsv'
doc_dataset_path = 'collection_dl20_54.tsv'
sim_fn = CosineSim.sim

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='right')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModel.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16, quantization_config=bnb_config)
model.eval()


tokenizer.pad_token = tokenizer.bos_token

# add special tokens
# token indicates encoding / compression task 
tokenizer.enc_token = '<enc>'
tokenizer.add_tokens([tokenizer.enc_token], special_tokens=True) 
tokenizer.enc_token_id = tokenizer.encode(tokenizer.enc_token, add_special_tokens=False)[0]

# emb token
tokenizer.emb_token0 = '<emb0>'
tokenizer.add_tokens([tokenizer.emb_token0], special_tokens=True)
tokenizer.emb_token_id0 = tokenizer.encode(tokenizer.emb_token0, add_special_tokens=False)[0]
print(tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset(path):
    data = {}
    for l in open(path):
        id_, content = l.strip().split('\t')
        data[id_] = content
    return data

query_dataset = load_dataset(query_dataset_path)
doc_dataset = load_dataset(doc_dataset_path)



def get_emb_repr(model, inp, token_id):
    emb = model(**inp, output_hidden_states=True).hidden_states[-1]
    mask = inp['input_ids'] == token_id
    return emb[mask]


# Rerank
def rerank(query, passages):
    passages = [tokenizer.enc_token + passage + tokenizer.eos_token + tokenizer.emb_token0 for passage in passages]
    passage_encoding = tokenizer(passages, padding=True, add_special_tokens=False, return_tensors='pt')
    queries = [tokenizer.enc_token + query + tokenizer.eos_token + tokenizer.emb_token0]
    query_encoding = tokenizer(queries, padding=True, add_special_tokens=False, return_tensors='pt')
    
    query_encoding = {key: query_encoding[key].to(device) for key in query_encoding}
    passage_encoding = {key: passage_encoding[key].to(device) for key in passage_encoding}
    
    with torch.no_grad():
        query_embedding = get_emb_repr(model, query_encoding, tokenizer.emb_token_id0)
        passage_embedding = get_emb_repr(model, passage_encoding, tokenizer.emb_token_id0)
        scores = sim_fn(query_embedding, passage_embedding).squeeze()
    return scores.float().cpu().numpy()

def rerank_queries(queries):
    reranked_queries = {}
    for qid, passage_scores in tqdm(queries.items()):
        query = query_dataset[qid]
        passages = [doc_dataset[doc_id] for doc_id, _ in passage_scores]
        scores = rerank(query, passages)
        for (doc_id, bm25_score), score in zip(passage_scores, scores):
            print(qid, doc_id, 'bm25', bm25_score, 'emb',score)
        reranked_passages = [(doc_id, score) for (doc_id, _), score in zip(passage_scores, scores)]
        reranked_passages = sorted(reranked_passages, key=lambda x: x[1], reverse=True)
        reranked_queries[qid] = reranked_passages
    return reranked_queries

# Load retrieval file (initial ranking)
def load_trec_file(file_path):
    queries = {}
    with open(file_path, 'r') as f:
        for line in f:
            qid, _, doc_id, rank, score, _ = line.strip().split()
            if qid not in queries:
                queries[qid] = []
            queries[qid].append((doc_id, float(score)))
    return queries

retrieval_data = load_trec_file(retrieval_file_path)

reranked_data = rerank_queries(retrieval_data)

# Save reranked data to a new TREC file
def save_to_trec_file(reranked_data, file_path):
    with open(file_path, 'w') as f:
        for qid, passage_scores in reranked_data.items():
            for rank, (doc_id, score) in enumerate(passage_scores, 1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score} reranked\n")

save_to_trec_file(reranked_data, reranked_file_path)





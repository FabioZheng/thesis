import datasets 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.utils.data import DataLoader
import torch
from vllm import LLM as vllm
from vllm import  SamplingParams
import random
from tqdm import tqdm
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
batch_size = 1024
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.bos_token
max_new_tokens = 24
# quant_config = BitsAndBytesConfig(
#                     load_in_4bit=True,
#                     bnb_4bit_quant_type='nf4',
#                     bnb_4bit_compute_dtype='bfloat16',
#                 )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quant_config, 
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
#     device_map='auto',
# )

model = vllm(model=model_name,tensor_parallel_size=torch.cuda.device_count(), dtype=torch.float16,gpu_memory_utilization=0.9, max_model_len=4096, enforce_eager=False, kv_cache_dtype="fp8_e5m2")  
sampling_params =  SamplingParams(temperature=1,max_tokens=max_new_tokens,best_of=1, top_p=1, top_k=-1)



dataset_path = '/projects/0/gusr0546/research/RAG/datasets/kilt-100w_full/'
dataset = datasets.load_from_disk(dataset_path)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def collate_fn(batch):

    ids = list()
    prompts = list()
    for example in batch:
        length = random.randint(4, 15)
        messages = [
            {"role": "system", "content": "You are an AI that models diverse user queries (not only wh-questions) based on a text passage. "},
            {"role": "user", "content": f"Generate a short, random user search query (length={length})based on this passage: {example['content']}"},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=
            False,
        )  + 'Question: '
        prompts.append(prompt)
        ids.append(example['id'])

    #inp = tokenizer(prompts, padding=True, return_tensors='pt')
    return  prompts, ids
# Create a custom DataLoader using the custom collate function
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=3)
with open('queries.txt', 'w') as fout:
    # Iterate over the dataloader
    for inp, ids in tqdm(dataloader):
        # outputs = model.generate(
        # **inp.to(model.device),
        # max_new_tokens=32,
        # eos_token_id=terminators,
        # do_sample=False,
        # )
        with torch.no_grad():
            outputs = model.generate(inp, sampling_params)
            decoded = [output.outputs[0].text for output in outputs]
            for id_, gen in zip(ids, decoded):
                # print('doc:', doc)
                gen_clean = gen.strip().split('<|eot_id|>')[0]
                gen_clean = gen_clean.replace('\n' , ' ').replace('\n', ' ')
                gen_clean = gen_clean.strip('"')
                print(gen_clean)
                fout.write(f'{id_}\t{gen_clean}\n')
                fout.flush()


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import datasets
from torch.utils.data import DataLoader
import datasets
import os 
from poc_decode_emb import compress, clean_generated
from torch.utils.data import DataLoader
from modeling_llama import LlamaForCausalLM



model_name = 'emb_TinyLlama_TinyLlama-1.1B-Chat-v1.0/checkpoint-100/'
model_name = '/gpfs/scratch1/shared/drau/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_32_emb_4/checkpoint-39000'
model_name = '/gpfs/scratch1/shared/drau/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_32/checkpoint-68000/'

model_name = '/projects/0/gusr0546/research/dec_from_emb/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_100_emb_1/checkpoint-100000'
#model_name = '/projects/0/gusr0546/research/dec_from_emb/emb_TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T_lora_False_len_300_emb_1/checkpoint-44000'
tokenizer_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
doc_max_length = 100
num_emb_tokens = 1
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side='right')
tokenizer.pad_token = tokenizer.bos_token

dataset_path = 'BeIR/msmarco'
textfield = 'text'
dataset = datasets.load_dataset(dataset_path, 'corpus')['corpus']
# dataset_path = '/projects/0/gusr0546/research/RAG/datasets/kilt-100w_full/'
# textfield = 'content'
# dataset = datasets.load_from_disk(dataset_path).select(range(100))



# emb_tokens, emb_token_ids = list(), list()
# # add num_emb_tokens special tokens to tokenizer
# for i in range(num_emb_tokens):
# # the representations of those tokens are used to get representations for, additionally they are useed as placeholders to be replaced with the compressed representations
#     emb_tokens.append(getattr(tokenizer, f'emb_token{i}'))
#     emb_token_ids.append(getattr(tokenizer, f'emb_token_id{i}'))
# print(tokenizer)


 # add special tokens
# token indicates encoding / compression task
tokenizer.enc_token = '<enc>'
tokenizer.add_tokens([tokenizer.enc_token], special_tokens=True)
tokenizer.enc_token_id = tokenizer.encode(tokenizer.enc_token, add_special_tokens=False)[0]

# token indicates decoding / decompression task
tokenizer.dec_emb_token = '<dec_emb>'
tokenizer.add_tokens([tokenizer.dec_emb_token], special_tokens=True)
tokenizer.dec_emb_token_id = tokenizer.encode(tokenizer.dec_emb_token, add_special_tokens=False)[0]

# token indicates decoding / decompression task
tokenizer.start_dec = '<start_dec>'
tokenizer.add_tokens([tokenizer.start_dec], special_tokens=True)
tokenizer.start_dec_id = tokenizer.encode(tokenizer.start_dec, add_special_tokens=False)[0]


emb_tokens, emb_token_ids = list(), list()
# add num_emb_tokens special tokens to tokenizer
for i in range(num_emb_tokens):
# the representations of those tokens are used to get representations for, additionally they are useed as placeholders to be replaced with the compressed representations
    setattr(tokenizer, f'emb_token{i}', f'<emb{i}>')
    tokenizer.add_tokens([getattr(tokenizer, f'emb_token{i}')], special_tokens=True)
    setattr(tokenizer, f'emb_token_id{i}', tokenizer.encode(getattr(tokenizer, f'emb_token{i}'), add_special_tokens=False)[0])
    emb_tokens.append(getattr(tokenizer, f'emb_token{i}'))
    emb_token_ids.append(getattr(tokenizer, f'emb_token_id{i}'))
print(tokenizer)


quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )

                
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quant_config, 
#     attn_implementation="flash_attention_2",
#     device_map='auto',
# )

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map='auto',
    vocab_size=len(tokenizer)    
)

def collate_fn(examples, tokenizer, doc_max_length, textfield, emb_tokens):
    # input for compression
    truncated_doc = [tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e[textfield], add_special_tokens=False)[:doc_max_length])) for e in examples]
    inp_enc = [tokenizer.enc_token + truncated_doc[i] + tokenizer.eos_token + ''.join(emb_tokens) for i in range(len(examples))]
    inp_enc = tokenizer(inp_enc, return_tensors='pt', padding=True, add_special_tokens=False)
    # input for generation / decompression
    inp_dec = [tokenizer.dec_emb_token + ''.join(emb_tokens) + tokenizer.start_dec for i in range(len(examples))]
    inp_dec = tokenizer(inp_dec, return_tensors='pt', padding=True, add_special_tokens=False)

    labels = inp_dec['input_ids'].clone()
    # for emb_token_id in emb_token_ids:
    #     labels[labels == emb_token_id] = -100
    # labels[labels == tokenizer.dec_emb_token_id] = -100
    # labels[labels == tokenizer.start_dec_id] = -100
    # labels[labels == tokenizer.bos_token_id] = -100
    labels[labels == tokenizer.pad_token_id] = -100
    inp_dec.update({
        'inp_enc': inp_enc,
        'labels': labels
        })
    return inp_dec

dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda l: collate_fn(l, tokenizer, doc_max_length, textfield, emb_tokens))
for inputs in dataloader:
    with torch.no_grad():
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs.pop('labels')

        inputs_embeds, scaled_aux_loss = compress(model, inputs, emb_token_ids)
        out = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, do_sample=False, max_new_tokens=64)

        labels[labels == -100] = tokenizer.pad_token_id
        preds_str = tokenizer.batch_decode(out, skip_special_tokens=False)
        print(preds_str)
        preds_str = [ clean_generated(p) for p in preds_str]
        labels_str = tokenizer.batch_decode(labels, skip_special_tokens=False)
        print(tokenizer.batch_decode(inputs['inp_enc']['input_ids']))
        for l, gen in zip(labels_str, preds_str):
            print('label:', l)
            print('gen:', gen)
            print()
    


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
from transformers import Trainer, TrainingArguments
import torch
import datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets
import os 
from rouge import Rouge
from metrics import compute_rouge_scores, exact_match_score
import numpy as np
import argparse
import torch.nn.functional as F
import random 
from utils import *
random.seed(42)
# from modeling_llama import LlamaForCausalLM


# input scheme 


# content: t1, ..., tn
# compression
# inp: <enc>t1, .... tn <eos><emb1><emb2>... -> we obain compressed representations <emb1><emb2>...
# we then use those <emb1><emb2>... for auto-encoding / generation from compressed representations
# inp training: <dec><emb1><emb2>...t1, .... tn <eos>
# labels:        -100 ...  -100     t1, .... tn <eos>
# inference 
# <dec><emb1><emb2>...

# config 
# ------------------------------------------------

#model_name = 'google/gemma-2b-it'

def get_args():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # Model Configuration
    parser.add_argument("--num_emb_tokens", type=int, default=1, help="Number of embedding tokens")
    parser.add_argument("--doc_max_length", type=int, default=500, help="Maximum document length")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", help="Model name")

    # Training Configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=5000, help="Number of warmup steps")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=10000, help="Save steps")
    parser.add_argument("--lora_training", type=bool, default=False, help="Lora training flag")
    parser.add_argument("--scalar", type=float, default=1, help="contrastive loss scalar")
    parser.add_argument("--do_contrastive", type=bool, default=False, help="add contrastive sim-cse loss") 
    parser.add_argument("--tc", action='store_true', default=False, help="") 
    parser.add_argument("--me", action='store_true', default=False, help="") 
    parser.add_argument("--ae", action='store_true', default=False, help="") 
    parser.add_argument("--nte", action='store_true', default=False, help="") 
    parser.add_argument("--run", type=str, default='') 

    args = parser.parse_args()
    return args

args = get_args()

# this script expects each row in the hf dataset to contain a field 'content' which will be the subject to auto-encoding
dataset_path = '/projects/0/gusr0546/research/RAG/datasets/kilt-100w_full/'
model_name = args.model_name
train_folder = model_name.replace('/', '_')
num_emb_tokens = args.num_emb_tokens
batch_size = args.batch_size
gradient_accumulation = args.gradient_accumulation
save_steps = args.save_steps
warmup_steps = args.warmup_steps
lora_training = args.lora_training
doc_max_length = args.doc_max_length
do_contrastive = args.do_contrastive
scalar = args.scalar
epochs = args.epochs
lr = args.lr
logging_steps = args.logging_steps
pre_training_tasks = list()
if args.ae:
    pre_training_tasks.append(prepare_auto_encoding)
if args.tc:
    pre_training_tasks.append(prepare_text_continuation)
if args.me:
    pre_training_tasks.append(prepare_masked_extraction)
if args.nte:
    pre_training_tasks.append(prepare_next_token_emb)

output_dir = f"/scratch-shared/drau/dec_from_emb/emb_{train_folder}_lora_{lora_training}_len_{doc_max_length}_emb_{num_emb_tokens}_contrastive_{do_contrastive}_scalar_{scalar}_warmup_{args.warmup_steps}_lr_{args.lr}_ae{args.ae}_tc_{args.tc}_me_{args.me}_nte_{args.nte}_{args.run}"

textfield = 'content'
rouge = Rouge()

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')
tokenizer.pad_token = tokenizer.bos_token

 # add special tokens
# token indicates encoding / compression task 
tokenizer.enc_token = '<enc>'
tokenizer.add_tokens([tokenizer.enc_token], special_tokens=True) 
tokenizer.enc_token_id = tokenizer.encode(tokenizer.enc_token, add_special_tokens=False)[0]

if args.tc or args.ae or args.me:
    # token indicates decoding / decompression task 
    tokenizer.dec_emb_token = '<dec_emb>'
    tokenizer.add_tokens([tokenizer.dec_emb_token], special_tokens=True) 
    tokenizer.dec_emb_token_id = tokenizer.encode(tokenizer.dec_emb_token, add_special_tokens=False)[0]

# token indicates decoding / decompression task 
tokenizer.start_dec = '<start_dec>'
tokenizer.add_tokens([tokenizer.start_dec], special_tokens=True) 
tokenizer.start_dec_id = tokenizer.encode(tokenizer.start_dec, add_special_tokens=False)[0]

if args.tc:
    # token indicates text-continuation task
    tokenizer.tc_token = '<tc>'
    tokenizer.add_tokens([tokenizer.tc_token], special_tokens=True) 
    tokenizer.tc_token_id = tokenizer.encode(tokenizer.tc_token, add_special_tokens=False)[0]

if args.me:
    # token indicates masked-extraction task
    tokenizer.me_token = '<me>'
    tokenizer.add_tokens([tokenizer.me_token], special_tokens=True) 
    tokenizer.me_token_id = tokenizer.encode(tokenizer.me_token, add_special_tokens=False)[0]

    # token indicates masked-extraction task
    tokenizer.mask_token = '<mask>'
    tokenizer.add_tokens([tokenizer.mask_token], special_tokens=True) 
    tokenizer.mask_token_id = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)[0]

# special_tokens_dict = {'additional_special_tokens': ['[C1]','[C2]','[C3]','[C4]']}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
#Otherwise use tokenizer.add_special_tokens({'unk_token': '<unk>'})

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


dataset = datasets.load_from_disk(dataset_path).shuffle(seed=42)
dataset = dataset.train_test_split(test_size=200)

# def truncate(example):
#     example['content'] = ' '.join(example['content'].split(' ')[:doc_max_length])
#     return example

# dataset = dataset.map(truncate, num_proc=10)


quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype='bfloat16',
                )
quant_config = None
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# use for bi-directional attention 

# model = LlamaForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config=quant_config, 
#     device_map='auto',
# )
# model.merge_and_unload()


model.config.pretraining_tp = 1
model.resize_token_embeddings(len(tokenizer))

if lora_training:
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        target_modules=['q_proj', 'down_proj', 'gate_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj'],
        )
    print(lora_config)
    # get adapter
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


def collate_fn(examples, tokenizer, doc_max_length, textfield, emb_tokens, train=True):
    docs = [tokenizer.decode(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(e[textfield], add_special_tokens=False)[:doc_max_length])) for e in examples]
    prepare_input = random.choice(pre_training_tasks)
    inp_enc, inp_dec, labels = prepare_input(tokenizer, emb_tokens, docs, train=train)
    labels[labels == tokenizer.pad_token_id] = -100
    for emb_token_id in emb_token_ids:
        labels[labels == emb_token_id] = -100
    inp_dec.update({
        'inp_enc': inp_enc,
        'labels': labels
        })
    return inp_dec

os.environ["WANDB_PROJECT"] = 'dec_from_emb'




# # get a representation given a token_id
def get_emb_repr(model, inp, token_id):
    emb = model(**inp, output_hidden_states=True).hidden_states[-1]
    # for bi-attention use this line
    #emb = model(**inp, output_hidden_states=True, is_causal=False).hidden_states[-1]
    mask = inp['input_ids'] == token_id
    return emb[mask].unsqueeze(1)

# replace representation in inputs_embeds where input_ids == place_folder_id with insert_embeds
def replace_embeddings(input_ids, inputs_embeds, insert_embeds, emb_token_id):
    insert_embeds = insert_embeds.to(inputs_embeds.device)
    # get indexes of placeholder ids
    repl_indexes = (input_ids == emb_token_id).nonzero(as_tuple=True)[1]
    for i, (repl_index, insert_embed) in enumerate(zip(repl_indexes,insert_embeds )):
        inputs_embeds[i] = torch.cat((inputs_embeds[i, :repl_index], insert_embed, inputs_embeds[i, repl_index+1:] ), 0)
 
    # replace_index = (input_ids == placeholder_id).nonzero(as_tuple=True)[1][0]
    # inputs_embeds_ = torch.cat((inputs_embeds[:, :replace_index], insert_embeds, inputs_embeds[:, replace_index+1:] ), 1)
    return inputs_embeds


# replace all representations
def compress_and_contruct_decoder_input(model, inputs, emb_token_ids, do_contrastive=False):
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])

    for emb_token_id in emb_token_ids:
        emb = get_emb_repr(model, inputs['inp_enc'], emb_token_id)
        inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb, emb_token_id)

    return inputs_embeds 




class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        attention_mask = inputs['attention_mask']
        labels = inputs.pop('labels')
        inputs_embeds = compress_and_contruct_decoder_input(model, inputs, emb_token_ids, do_contrastive=do_contrastive)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return out.loss

    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
            with torch.no_grad():
                attention_mask = inputs['attention_mask']
                labels = inputs.pop('labels')
                inputs_embeds = compress_and_contruct_decoder_input(model, inputs, emb_token_ids, do_contrastive=do_contrastive)
                out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                return out.loss, out.logits, labels
            
    def save_model(self, output_dir=None, _internal_call=False):
        # Save the model
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)
        if self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

    def get_eval_dataloader(self, eval_dataset):
        eval_dataloader = super().get_eval_dataloader(eval_dataset)
        eval_dataloader.collate_fn = lambda l: collate_fn(l, tokenizer, doc_max_length, textfield, emb_tokens, train=True)
        return eval_dataloader
    
def clean_generated(text):
    if hasattr(tokenizer, 'start_dec'):
        text = text.split(tokenizer.start_dec, 1)[1] if tokenizer.start_dec in text else text
    text = text.split(tokenizer.eos_token, 1)[0] if tokenizer.eos_token in text else text
    return text


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels[labels == -100] = tokenizer.pad_token_id
    preds_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
    preds_str = [ clean_generated(p) for p in preds_str]
    labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # print examples
    for pre, lab in zip(preds_str[:20], labels_str[:20]):
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


if __name__ == "__main__": 
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        remove_unused_columns=False,
        gradient_accumulation_steps=gradient_accumulation,
        evaluation_strategy='steps',
        report_to=None,
        save_strategy="steps",
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        dataloader_num_workers=4,
        do_eval=True,
    )
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=lambda l: collate_fn(l, tokenizer, doc_max_length, textfield, emb_tokens),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    # Training loop
    trainer.train()




# def cosine(query_embds, doc_embds):
#     query_embds = query_embds / (torch.norm(query_embds, dim=-1, keepdim=True) + 1e-9)
#     doc_embds = doc_embds / (torch.norm(doc_embds, dim=-1, keepdim=True) + 1e-9)
#     return torch.mm(query_embds, doc_embds.t())
# def contrastive_loss(emb_1, emb_2):
#     logits = cosine(emb_1.squeeze(1), emb_2.squeeze(1))
#     labels = torch.arange(emb_1.size(0)).to(emb_1.device)
#     loss = F.cross_entropy(logits, labels)
#     return loss * scalar



#     else:
#         emb1 = get_emb_repr(model, inputs['inp_enc'], emb_token_ids[0])
#         emb2 = get_emb_repr(model, inputs['inp_enc'], emb_token_ids[0])
#         aux_loss = contrastive_loss(emb1, emb2)
#         #inputs_embeds = replace_embeddings(inputs['input_ids'], inputs_embeds, emb1, emb_token_ids[0])


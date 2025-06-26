import torch
import random
import math

def prepare_auto_encoding(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, enc_max_len, train=True):
    # auto-encoding 
    # input for compression
    num_embeds = math.ceil(enc_max_len / compression_rate)
    if compressor_tokenizer ==  decoder_tokenizer:
        inp_enc_text = [ decoder_tokenizer.enc_token +  decoder_tokenizer.bos_token + text + decoder_tokenizer.eos_token for text in examples['text']]
        inp_enc = compressor_tokenizer(inp_enc_text, return_tensors='pt', padding='max_length', max_length=enc_max_len + 3, truncation=True, add_special_tokens=False)
        mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_embeds), decoder_tokenizer.mem_token_id, dtype=torch.long)
        inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
        inp_enc['attention_mask'] = torch.cat([inp_enc['attention_mask'], torch.ones(inp_enc['input_ids'].size(0), num_embeds)], dim=1)
        
    else:
        inp_enc = compressor_tokenizer(examples['text'], return_tensors='pt', padding='max_length', max_length=enc_max_len, truncation=True)

    dec_max_length = 3 + num_embeds + 128
    inp_dec = [decoder_tokenizer.ae_token + decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds + text + decoder_tokenizer.eos_token for text in examples['text']]
    inp_dec = decoder_tokenizer(inp_dec, return_tensors='pt', padding='max_length', add_special_tokens=False, max_length=dec_max_length, truncation=True)

    labels = inp_dec['input_ids'].clone()

    labels[labels == decoder_tokenizer.pad_token_id] = -100
    if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
        labels[labels == decoder_tokenizer.bos_token_id] = -100
    labels[labels == decoder_tokenizer.mem_token_id] = -100
    labels[labels == decoder_tokenizer.ae_token_id] = -100

    if not train:
        instr = [decoder_tokenizer.ae_token + decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds for i in range(len(examples['text']))]
        inp_dec = decoder_tokenizer(instr, return_tensors='pt', padding="longest",  max_length=dec_max_length, add_special_tokens=False,truncation=True)

    enc_attention_mask, enc_input_ids = inp_enc['attention_mask'], inp_enc['input_ids']
    dec_attention_mask, dec_input_ids = inp_dec['attention_mask'], inp_dec['input_ids']

    return {
        'enc_input_ids': enc_input_ids,
        'enc_attention_mask': enc_attention_mask,
        'dec_input_ids': dec_input_ids,
        'dec_attention_mask': dec_attention_mask,
        'labels': labels
    }

def prepare_text_continuation( examples, compressor_tokenizer, decoder_tokenizer, compression_rate, enc_max_len, train=True):
    # text continuation
    # input for text continuation
    num_embeds = math.ceil(enc_max_len / compression_rate)

    if compressor_tokenizer == decoder_tokenizer:
        inp_enc_text = [ decoder_tokenizer.enc_token +  decoder_tokenizer.bos_token + text + decoder_tokenizer.eos_token for text in examples['text']]
        inp_enc = compressor_tokenizer(inp_enc_text, return_tensors='pt', padding='max_length', max_length=enc_max_len + 3, truncation=True, add_special_tokens=False)
        mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_embeds), decoder_tokenizer.mem_token_id, dtype=torch.long)
        inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
        inp_enc['attention_mask'] = torch.cat([inp_enc['attention_mask'], torch.ones(inp_enc['input_ids'].size(0), num_embeds)], dim=1)
    else:
        inp_enc = compressor_tokenizer(examples['text'], return_tensors='pt', padding='max_length', max_length=enc_max_len, truncation=True)

    dec_max_length = 3 + num_embeds + 128
    inp_dec = [decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds + text + decoder_tokenizer.eos_token for text in examples['next_text']]
    inp_dec = decoder_tokenizer(inp_dec, return_tensors='pt', padding='max_length', truncation=True, max_length=dec_max_length)

    labels = inp_dec['input_ids'].clone()
    labels[labels == decoder_tokenizer.pad_token_id] = -100
    if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
        labels[labels == decoder_tokenizer.bos_token_id] = -100
    labels[labels == decoder_tokenizer.mem_token_id] = -100



    if not train:
        instr = [decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds for i in range(len(examples['text']))]
        inp_dec = decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,truncation=True)
        
    enc_attention_mask, enc_input_ids = inp_enc['attention_mask'], inp_enc['input_ids']
    dec_attention_mask, dec_input_ids = inp_dec['attention_mask'], inp_dec['input_ids']

    return {
        'enc_input_ids': enc_input_ids,
        'enc_attention_mask': enc_attention_mask,
        'dec_input_ids': dec_input_ids,
        'dec_attention_mask': dec_attention_mask,
        'labels': labels
    }


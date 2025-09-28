import torch
import random
import math
from typing import Dict


def pad_tokens_to_rate(
        tokens: Dict[str, torch.Tensor],
        rate: int,
        pad_token_id: int,
        pad_token_type_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Pad tokenized inputs so sequence length becomes a multiple of ``rate``.

    Parameters
    ----------
    tokens
        Dictionary returned by a tokenizer containing at least ``input_ids`` and
        optionally ``attention_mask`` and ``token_type_ids`` tensors.
    rate
        Desired compression rate. The resulting sequence length will be padded to
        the smallest multiple of ``rate`` that is greater than or equal to the
        current length.
    pad_token_id
        Token identifier used to pad ``input_ids``.
    pad_token_type_id
        Value used when extending ``token_type_ids``. Defaults to ``0`` which is
        suitable for single-sequence BERT-style models.

    Returns
    -------
    Dict[str, torch.Tensor]
        A dictionary with padded tensors. The original dictionary is not modified.
    """

    if rate <= 0:
        raise ValueError("rate must be positive")

    if "input_ids" not in tokens:
        raise KeyError("tokens dictionary must contain 'input_ids'")

    input_ids = tokens["input_ids"]
    seq_len = input_ids.size(1)
    remainder = seq_len % rate
    if remainder == 0:
        return tokens

    pad_len = rate - remainder
    pad_shape = (input_ids.size(0), pad_len)

    padded_tokens = dict(tokens)
    padded_tokens["input_ids"] = torch.cat(
        [input_ids, input_ids.new_full(pad_shape, pad_token_id)], dim=1
    )

    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        padded_tokens["attention_mask"] = torch.cat(
            [attention_mask, attention_mask.new_zeros(pad_shape)], dim=1
        )

    token_type_ids = tokens.get("token_type_ids")
    if token_type_ids is not None:
        padded_tokens["token_type_ids"] = torch.cat(
            [token_type_ids, token_type_ids.new_full(pad_shape, pad_token_type_id)],
            dim=1,
        )

    return padded_tokens


def prepare_auto_encoding(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, enc_max_len, train=True):
    # auto-encoding
    # input for compression
    num_embeds = math.ceil(enc_max_len / compression_rate)
    if compressor_tokenizer == decoder_tokenizer:
        inp_enc_text = [decoder_tokenizer.enc_token + decoder_tokenizer.bos_token + text + decoder_tokenizer.eos_token
                        for text in examples['text']]
        inp_enc = compressor_tokenizer(inp_enc_text, return_tensors='pt', padding='max_length',
                                       max_length=enc_max_len + 3, truncation=True, add_special_tokens=False)
        mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_embeds), decoder_tokenizer.mem_token_id,
                                dtype=torch.long)
        inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
        inp_enc['attention_mask'] = torch.cat(
            [inp_enc['attention_mask'], torch.ones(inp_enc['input_ids'].size(0), num_embeds)], dim=1)

    else:
        inp_enc = compressor_tokenizer(examples['text'], return_tensors='pt', padding='max_length',
                                       max_length=enc_max_len, truncation=True)

    dec_max_length = 3 + num_embeds + 128
    inp_dec = [
        decoder_tokenizer.ae_token + decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds + text + decoder_tokenizer.eos_token
        for text in examples['text']]
    inp_dec = decoder_tokenizer(inp_dec, return_tensors='pt', padding='max_length', add_special_tokens=False,
                                max_length=dec_max_length, truncation=True)

    labels = inp_dec['input_ids'].clone()

    labels[labels == decoder_tokenizer.pad_token_id] = -100
    if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
        labels[labels == decoder_tokenizer.bos_token_id] = -100
    labels[labels == decoder_tokenizer.mem_token_id] = -100
    labels[labels == decoder_tokenizer.ae_token_id] = -100

    if not train:
        instr = [decoder_tokenizer.ae_token + decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds for
                 i in range(len(examples['text']))]
        inp_dec = decoder_tokenizer(instr, return_tensors='pt', padding="longest", max_length=dec_max_length,
                                    add_special_tokens=False, truncation=True)

    enc_attention_mask, enc_input_ids = inp_enc['attention_mask'], inp_enc['input_ids']
    dec_attention_mask, dec_input_ids = inp_dec['attention_mask'], inp_dec['input_ids']

    return {
        'enc_input_ids': enc_input_ids,
        'enc_attention_mask': enc_attention_mask,
        'dec_input_ids': dec_input_ids,
        'dec_attention_mask': dec_attention_mask,
        'labels': labels
    }


def prepare_text_continuation(examples, compressor_tokenizer, decoder_tokenizer, compression_rate, enc_max_len,
                              train=True):
    # text continuation
    # input for text continuation
    num_embeds = math.ceil(enc_max_len / compression_rate)

    if compressor_tokenizer == decoder_tokenizer:
        inp_enc_text = [decoder_tokenizer.enc_token + decoder_tokenizer.bos_token + text + decoder_tokenizer.eos_token
                        for text in examples['text']]
        inp_enc = compressor_tokenizer(inp_enc_text, return_tensors='pt', padding='max_length',
                                       max_length=enc_max_len + 3, truncation=True, add_special_tokens=False)
        mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_embeds), decoder_tokenizer.mem_token_id,
                                dtype=torch.long)
        inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
        inp_enc['attention_mask'] = torch.cat(
            [inp_enc['attention_mask'], torch.ones(inp_enc['input_ids'].size(0), num_embeds)], dim=1)
    else:
        inp_enc = compressor_tokenizer(examples['text'], return_tensors='pt', padding='max_length',
                                       max_length=enc_max_len, truncation=True)

    dec_max_length = 3 + num_embeds + 128
    inp_dec = [
        decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds + text + decoder_tokenizer.eos_token for
        text in examples['next_text']]
    inp_dec = decoder_tokenizer(inp_dec, return_tensors='pt', padding='max_length', truncation=True,
                                max_length=dec_max_length)

    labels = inp_dec['input_ids'].clone()
    labels[labels == decoder_tokenizer.pad_token_id] = -100
    if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
        labels[labels == decoder_tokenizer.bos_token_id] = -100
    labels[labels == decoder_tokenizer.mem_token_id] = -100

    if not train:
        instr = [decoder_tokenizer.bos_token + decoder_tokenizer.mem_token * num_embeds for i in
                 range(len(examples['text']))]
        inp_dec = decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                    truncation=True)

    enc_attention_mask, enc_input_ids = inp_enc['attention_mask'], inp_enc['input_ids']
    dec_attention_mask, dec_input_ids = inp_dec['attention_mask'], inp_dec['input_ids']

    return {
        'enc_input_ids': enc_input_ids,
        'enc_attention_mask': enc_attention_mask,
        'dec_input_ids': dec_input_ids,
        'dec_attention_mask': dec_attention_mask,
        'labels': labels
    }

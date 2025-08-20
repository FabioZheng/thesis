import torch
import random
import math

def _prepare_decoder_inputs(examples, decoder_tokenizer, num_embeds, task="ae", train=True):
    """Helper to create decoder inputs/labels for a given number of <MEM> tokens."""

    if task == "ae":
        prefix = decoder_tokenizer.ae_token + decoder_tokenizer.bos_token
        texts = examples["text"]
    else:
        prefix = decoder_tokenizer.bos_token
        texts = examples["next_text"]

    instr = [
        prefix + decoder_tokenizer.mem_token * num_embeds + text + decoder_tokenizer.eos_token
        for text in texts
    ]

    dec_max_length = 3 + num_embeds + 128
    inp_dec = decoder_tokenizer(
        instr,
        return_tensors="pt",
        padding="max_length",
        add_special_tokens=False,
        max_length=dec_max_length,
        truncation=True,
    )

    labels = inp_dec["input_ids"].clone()
    labels[labels == decoder_tokenizer.pad_token_id] = -100
    if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
        labels[labels == decoder_tokenizer.bos_token_id] = -100
    labels[labels == decoder_tokenizer.mem_token_id] = -100
    if task == "ae":
        labels[labels == decoder_tokenizer.ae_token_id] = -100

    if not train:
        if task == "ae":
            prompt = [
                decoder_tokenizer.ae_token
                + decoder_tokenizer.bos_token
                + decoder_tokenizer.mem_token * num_embeds
                for _ in range(len(texts))
            ]
        else:
            prompt = [
                decoder_tokenizer.bos_token
                + decoder_tokenizer.mem_token * num_embeds
                for _ in range(len(texts))
            ]
        inp_dec = decoder_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=True,
        )

    return inp_dec["input_ids"], inp_dec["attention_mask"], labels


def prepare_auto_encoding(
    examples,
    compressor_tokenizer,
    decoder_tokenizer,
    compression_rates,
    enc_max_len,
    train=True,
):
    """Prepare inputs for auto-encoding across multiple compression rates."""

    if not isinstance(compression_rates, (list, tuple)):
        compression_rates = [compression_rates]

    num_embeds = {r: math.ceil(enc_max_len / r) for r in compression_rates}

    if compressor_tokenizer == decoder_tokenizer:
        max_mem = max(num_embeds.values())
        inp_enc_text = [
            decoder_tokenizer.enc_token
            + decoder_tokenizer.bos_token
            + text
            + decoder_tokenizer.eos_token
            for text in examples["text"]
        ]
        inp_enc = compressor_tokenizer(
            inp_enc_text,
            return_tensors="pt",
            padding="max_length",
            max_length=enc_max_len + 3,
            truncation=True,
            add_special_tokens=False,
        )
        mem_tokens = torch.full(
            (inp_enc["input_ids"].size(0), max_mem),
            decoder_tokenizer.mem_token_id,
            dtype=torch.long,
        )
        inp_enc["input_ids"] = torch.cat([inp_enc["input_ids"], mem_tokens], dim=1)
        inp_enc["attention_mask"] = torch.cat(
            [inp_enc["attention_mask"], torch.ones(inp_enc["input_ids"].size(0), max_mem)],
            dim=1,
        )
    else:
        inp_enc = compressor_tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=enc_max_len,
            truncation=True,
        )

    dec_input_ids = {}
    dec_attention_mask = {}
    labels = {}
    for r, n_mem in num_embeds.items():
        ids, mask, lab = _prepare_decoder_inputs(
            examples, decoder_tokenizer, n_mem, task="ae", train=train
        )
        dec_input_ids[r] = ids
        dec_attention_mask[r] = mask
        labels[r] = lab

    return {
        "enc_input_ids": inp_enc["input_ids"],
        "enc_attention_mask": inp_enc["attention_mask"],
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "labels": labels,
    }


def prepare_text_continuation(
    examples,
    compressor_tokenizer,
    decoder_tokenizer,
    compression_rates,
    enc_max_len,
    train=True,
):
    """Prepare inputs for text-continuation across multiple compression rates."""

    if not isinstance(compression_rates, (list, tuple)):
        compression_rates = [compression_rates]

    num_embeds = {r: math.ceil(enc_max_len / r) for r in compression_rates}

    if compressor_tokenizer == decoder_tokenizer:
        max_mem = max(num_embeds.values())
        inp_enc_text = [
            decoder_tokenizer.enc_token
            + decoder_tokenizer.bos_token
            + text
            + decoder_tokenizer.eos_token
            for text in examples["text"]
        ]
        inp_enc = compressor_tokenizer(
            inp_enc_text,
            return_tensors="pt",
            padding="max_length",
            max_length=enc_max_len + 3,
            truncation=True,
            add_special_tokens=False,
        )
        mem_tokens = torch.full(
            (inp_enc["input_ids"].size(0), max_mem),
            decoder_tokenizer.mem_token_id,
            dtype=torch.long,
        )
        inp_enc["input_ids"] = torch.cat([inp_enc["input_ids"], mem_tokens], dim=1)
        inp_enc["attention_mask"] = torch.cat(
            [inp_enc["attention_mask"], torch.ones(inp_enc["input_ids"].size(0), max_mem)],
            dim=1,
        )
    else:
        inp_enc = compressor_tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=enc_max_len,
            truncation=True,
        )

    dec_input_ids = {}
    dec_attention_mask = {}
    labels = {}
    for r, n_mem in num_embeds.items():
        ids, mask, lab = _prepare_decoder_inputs(
            examples, decoder_tokenizer, n_mem, task="tc", train=train
        )
        dec_input_ids[r] = ids
        dec_attention_mask[r] = mask
        labels[r] = lab

    return {
        "enc_input_ids": inp_enc["input_ids"],
        "enc_attention_mask": inp_enc["attention_mask"],
        "dec_input_ids": dec_input_ids,
        "dec_attention_mask": dec_attention_mask,
        "labels": labels,
    }


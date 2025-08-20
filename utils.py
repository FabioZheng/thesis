import torch
import random
import math


def prepare_auto_encoding(examples, compressor_tokenizer, decoder_tokenizer, compression_rates, enc_max_len, train=True):
    """Prepare inputs for the auto-encoding task.

    Instead of returning nested dictionaries keyed by the compression rate, this
    function now flattens the keys so that each rate-specific tensor is stored
    under a distinct key, e.g. ``dec_input_ids_{rate}``.
    """
    if not isinstance(compression_rates, (list, tuple)):
        compression_rates = [compression_rates]

    # Use the smallest rate to determine the maximum number of <MEM> tokens the
    # encoder needs to allocate.
    min_rate = min(compression_rates)
    num_enc_mem = math.ceil(enc_max_len / min_rate)

    if compressor_tokenizer == decoder_tokenizer:
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
            (inp_enc["input_ids"].size(0), num_enc_mem),
            decoder_tokenizer.mem_token_id,
            dtype=torch.long,
        )
        inp_enc["input_ids"] = torch.cat([inp_enc["input_ids"], mem_tokens], dim=1)
        inp_enc["attention_mask"] = torch.cat(
            [
                inp_enc["attention_mask"],
                torch.ones(inp_enc["input_ids"].size(0), num_enc_mem),
            ],
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

    result = {
        "enc_input_ids": inp_enc["input_ids"],
        "enc_attention_mask": inp_enc["attention_mask"],
    }

    for rate in compression_rates:
        num_embeds = math.ceil(enc_max_len / rate)
        dec_max_length = 3 + num_embeds + 128

        inp_dec_text = [
            decoder_tokenizer.ae_token
            + decoder_tokenizer.bos_token
            + decoder_tokenizer.mem_token * num_embeds
            + text
            + decoder_tokenizer.eos_token
            for text in examples["text"]
        ]
        inp_dec = decoder_tokenizer(
            inp_dec_text,
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
        labels[labels == decoder_tokenizer.ae_token_id] = -100

        if not train:
            instr = [
                decoder_tokenizer.ae_token
                + decoder_tokenizer.bos_token
                + decoder_tokenizer.mem_token * num_embeds
                for _ in range(len(examples["text"]))
            ]
            inp_dec = decoder_tokenizer(
                instr,
                return_tensors="pt",
                padding="longest",
                max_length=dec_max_length,
                add_special_tokens=False,
                truncation=True,
            )

        result[f"dec_input_ids_{rate}"] = inp_dec["input_ids"]
        result[f"dec_attention_mask_{rate}"] = inp_dec["attention_mask"]
        result[f"labels_{rate}"] = labels

    return result


def prepare_text_continuation(examples, compressor_tokenizer, decoder_tokenizer, compression_rates, enc_max_len, train=True):
    """Prepare inputs for the text-continuation task with flattened keys."""
    if not isinstance(compression_rates, (list, tuple)):
        compression_rates = [compression_rates]

    min_rate = min(compression_rates)
    num_enc_mem = math.ceil(enc_max_len / min_rate)

    if compressor_tokenizer == decoder_tokenizer:
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
            (inp_enc["input_ids"].size(0), num_enc_mem),
            decoder_tokenizer.mem_token_id,
            dtype=torch.long,
        )
        inp_enc["input_ids"] = torch.cat([inp_enc["input_ids"], mem_tokens], dim=1)
        inp_enc["attention_mask"] = torch.cat(
            [
                inp_enc["attention_mask"],
                torch.ones(inp_enc["input_ids"].size(0), num_enc_mem),
            ],
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

    result = {
        "enc_input_ids": inp_enc["input_ids"],
        "enc_attention_mask": inp_enc["attention_mask"],
    }

    for rate in compression_rates:
        num_embeds = math.ceil(enc_max_len / rate)
        dec_max_length = 3 + num_embeds + 128

        inp_dec_text = [
            decoder_tokenizer.bos_token
            + decoder_tokenizer.mem_token * num_embeds
            + text
            + decoder_tokenizer.eos_token
            for text in examples["next_text"]
        ]
        inp_dec = decoder_tokenizer(
            inp_dec_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=dec_max_length,
        )

        labels = inp_dec["input_ids"].clone()
        labels[labels == decoder_tokenizer.pad_token_id] = -100
        if decoder_tokenizer.bos_token_id != decoder_tokenizer.pad_token_id:
            labels[labels == decoder_tokenizer.bos_token_id] = -100
        labels[labels == decoder_tokenizer.mem_token_id] = -100

        if not train:
            instr = [
                decoder_tokenizer.bos_token
                + decoder_tokenizer.mem_token * num_embeds
                for _ in range(len(examples["text"]))
            ]
            inp_dec = decoder_tokenizer(
                instr,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False,
                truncation=True,
            )

        result[f"dec_input_ids_{rate}"] = inp_dec["input_ids"]
        result[f"dec_attention_mask_{rate}"] = inp_dec["attention_mask"]
        result[f"labels_{rate}"] = labels

    return result


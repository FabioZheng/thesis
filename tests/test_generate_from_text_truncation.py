import logging
import math

import pytest

torch = pytest.importorskip("torch")

from modeling_cocom import COCOM


class DummyTokenizer:
    def __init__(self, model_max_length, pad_token_id=0, with_special_tokens=True):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.with_special_tokens = with_special_tokens
        # decoder-specific attributes
        self.mem_token = "<MEM>"
        self.mem_token_id = 99
        self.sep_token = "<SEP>"
        self.bos_token = "<BOS>"
        self.enc_token = "<ENC>"

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        return_tensors=None,
        pad_to_multiple_of=None,
        return_overflowing_tokens=False,
        return_length=False,
        return_attention_mask=True,
        add_special_tokens=None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_masks = []
        lengths = []
        truncated_counts = []
        overflow_tokens = []
        overflow_mapping = []

        for idx, text in enumerate(texts):
            tokens = text.strip().split()
            if self.with_special_tokens and (add_special_tokens is None or add_special_tokens):
                tokens = ["<s>"] + tokens + ["</s>"]
            full_ids = [len(tok) % 7 + 1 for tok in tokens]
            truncated = full_ids[: self.model_max_length] if truncation else full_ids
            dropped = len(full_ids) - len(truncated)
            lengths.append(len(truncated))
            truncated_counts.append(dropped)
            input_ids.append(truncated)
            attention_masks.append([1] * len(truncated))
            if dropped > 0 and return_overflowing_tokens:
                overflow_tokens.append(full_ids[-dropped:])
                overflow_mapping.append(idx)

        max_length = max((len(ids) for ids in input_ids), default=0)
        if pad_to_multiple_of:
            max_length = int(math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of)

        if padding is True or padding == "longest":
            for ids, mask in zip(input_ids, attention_masks):
                pad_len = max_length - len(ids)
                if pad_len > 0:
                    ids.extend([self.pad_token_id] * pad_len)
                    mask.extend([0] * pad_len)

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long)
            if return_tensors == "pt"
            else input_ids,
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
            if return_tensors == "pt"
            else attention_masks,
        }

        if return_length:
            result["length"] = lengths
        if return_overflowing_tokens:
            result["overflowing_tokens"] = overflow_tokens
            result["overflow_to_sample_mapping"] = overflow_mapping
            result["num_truncated_tokens"] = truncated_counts

        return result


class DummyCompressor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class DummyModel:
    def __init__(self, use_compressor):
        self.current_rate = 2
        self.sep = False
        self.decoder_tokenizer = DummyTokenizer(model_max_length=6)
        self.compr = DummyCompressor(DummyTokenizer(model_max_length=4)) if use_compressor else None
        self._generate_called = False

    def generate(self, model_input, max_new_tokens=128):
        self._generate_called = True
        return {"output": "dummy"}


@pytest.mark.parametrize("use_compressor", [False, True])
def test_generate_from_text_emits_warning_for_truncation(caplog, use_compressor):
    dummy_model = DummyModel(use_compressor)
    contexts = [["one two three four five six seven eight"]]
    questions = ["What?"]

    with caplog.at_level(logging.WARNING):
        COCOM.generate_from_text(dummy_model, contexts, questions)

    assert dummy_model._generate_called, "generate should be invoked"
    warning_messages = [record.message for record in caplog.records if record.levelno == logging.WARNING]
    assert warning_messages, "Expected at least one warning message"
    assert any("Context" in message and "truncated" in message for message in warning_messages)

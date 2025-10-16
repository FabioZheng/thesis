import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from train_cmab import load_model_safely


def _resolve_context_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    alternatives = [
        Path("data") / "contexts.pt",
        Path("data") / "contexts" / "contexts.pt",
    ]
    for candidate in alternatives:
        if candidate.exists():
            print(f"Context file not found at {path}. Using {candidate} instead.")
            return candidate

    raise FileNotFoundError(f"Could not find context file at {path} or known alternatives.")


def _load_first_context(path: Path) -> Tuple[str, torch.Tensor, Dict[str, Any]]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or not data:
        raise ValueError("Context file must contain a non-empty dictionary of tensors.")

    first_key = next(iter(data))
    payload = data[first_key]

    if isinstance(payload, dict):
        context = payload.get("context")
        metadata = {k: v for k, v in payload.items() if k != "context"}
    else:
        context = payload
        metadata = {}

    if context is None:
        raise ValueError(f"Missing 'context' tensor for document id {first_key}.")

    if not isinstance(context, torch.Tensor):
        context = torch.as_tensor(context)

    # Stored contexts can have a batch dimension of size 1; remove it for convenience.
    if context.dim() == 3 and context.size(0) == 1:
        context = context.squeeze(0)

    if context.dim() != 2:
        raise ValueError(
            "Context tensor must have shape (mem_tokens, hidden_size) after squeezing the batch dimension."
        )

    return str(first_key), context.to(torch.float32), metadata


def _build_autoencoding_prompt(tokenizer, mem_tokens: int) -> Dict[str, torch.Tensor]:
    """Mimic the auto-encoding setup by feeding only memory tokens to the decoder."""

    ae_symbol = getattr(tokenizer, "ae_token", "") or ""
    bos = tokenizer.bos_token or ""
    mem_symbol = getattr(tokenizer, "mem_token", None)
    if not mem_symbol:
        raise ValueError("Decoder tokenizer does not define a memory token symbol.")

    prompt = f"{ae_symbol}{bos}{mem_symbol * mem_tokens}"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}


def decode_first_context(
    context_path: Path,
    model_source: str,
    max_new_tokens: int,
) -> None:
    doc_id, context, metadata = _load_first_context(context_path)
    print(f"Loaded context for document id {doc_id} (metadata: {metadata}).")

    model = load_model_safely(model_source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    mem_tokens = context.size(0)
    prompt = _build_autoencoding_prompt(model.decoder_tokenizer, mem_tokens)

    dec_input_ids = prompt["input_ids"].to(device)
    dec_attention_mask = prompt["attention_mask"].to(device)

    context = context.to(model.decoder.get_input_embeddings().weight.dtype)
    context = context.unsqueeze(0).unsqueeze(0)  # (batch=1, top_k=1, mem_tokens, hidden)

    with torch.no_grad():
        top_k = context.size(1)
        model.generation_top_k = top_k
        flattened = context.reshape(context.size(0) * top_k, context.size(2), context.size(3)).to(device)
        indices = range(0, flattened.size(0) + 1, top_k)
        inputs_embeds = model.replace_embeddings(flattened, dec_input_ids, indices)
        outputs = model.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens,
        )
        decoded = model.decoder_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    text = decoded[0] if decoded else ""
    print("Decoded text:\n")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode the first stored COCOM context.")
    parser.add_argument(
        "--contexts",
        default="data/context.pt",
        help="Path to the stored contexts file (default: data/context.pt)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local checkpoint directory containing a trained COCOM model.",
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help="Hugging Face model identifier to load the decoder from instead of a local checkpoint.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate during decoding.",
    )

    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.hf_model_name):
        raise SystemExit("Specify exactly one of --checkpoint or --hf-model-name.")

    context_path = _resolve_context_path(args.contexts)
    model_source = args.checkpoint or args.hf_model_name

    decode_first_context(
        context_path=context_path,
        model_source=model_source,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

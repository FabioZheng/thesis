
"""
info_density.py

Information Density evaluator for single documents.

Method:
1) Embed the original document with a chosen embedding model.
2) For T trials (default 10):
   - remove a *contiguous* block equal to 10% of the tokens (rounded up, at least 1 token),
   - re-embed the edited document,
   - compute deviation between embeddings.
3) Return the mean deviation across trials (and per-trial details).

This module depends on `sentence-transformers` and can reuse the TextEmbedder
from `retrieval.py` if available.

Usage (CLI):
    python info_density.py --text "your document here"
    # or
    python info_density.py --file /path/to/doc.txt --model sentence-transformers/all-MiniLM-L6-v2

Programmatic:
    from info_density import InformationDensityEvaluator
    evaluator = InformationDensityEvaluator(model_name="sentence-transformers/all-MiniLM-L6-v2")
    result = evaluator.evaluate("Some long document text ...", trials=10, remove_frac=0.10)
    print(result["avg_deviation"])
"""

from __future__ import annotations
import argparse
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Literal, Optional

import numpy as np

try:
    # Prefer reusing the same abstraction as in retrieval.py if present
    from retrieval import TextEmbedder  # type: ignore
except Exception:
    TextEmbedder = None  # will fallback below

# Fallback minimal embedder if retrieval.TextEmbedder is not available
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization; preserves order."""
    # Normalize whitespace while keeping token boundaries simple
    return str(text).strip().split()


def _detokenize(tokens: List[str]) -> str:
    """Join tokens back with single spaces."""
    return " ".join(tokens)


def _pick_contiguous_block(n_tokens: int, k: int, rng: random.Random) -> Tuple[int, int]:
    """Return (start_inclusive, end_exclusive) for a contiguous block of length k within [0, n_tokens)."""
    if n_tokens <= k:
        return 0, n_tokens
    start = rng.randint(0, n_tokens - k)
    end = start + k
    return start, end


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine distance = 1 - cosine similarity. Inputs may be unnormalized."""
    u = u.astype(np.float32, copy=False)
    v = v.astype(np.float32, copy=False)
    # Avoid division by zero
    nu = np.linalg.norm(u) + 1e-12
    nv = np.linalg.norm(v) + 1e-12
    return 1.0 - float((u @ v) / (nu * nv))


def _euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


@dataclass
class InformationDensityEvaluator:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    batch_size: int = 256
    seed: int = 42
    device: Optional[str] = None  # "cpu" | "cuda"
    distance: Literal["cosine", "euclidean"] = "cosine"

    def __post_init__(self):
        self.rng = random.Random(self.seed)

        if TextEmbedder is not None:
            self.embedder = TextEmbedder(
                model_name=self.model_name,
                normalize=self.normalize,
                batch_size=self.batch_size,
                device=self.device,
            )
        else:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required. Run: pip install sentence-transformers")
            self._st_model = SentenceTransformer(self.model_name, device=self.device)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if TextEmbedder is not None:
            return self.embedder.encode(texts)
        # Fallback
        embs = self._st_model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return embs

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.distance == "cosine":
            return _cosine_distance(a, b)
        return _euclidean_distance(a, b)

    def evaluate(
        self,
        text: str,
        trials: int = 10,
        remove_frac: float = 0.10,
        min_tokens: int = 20,
    ) -> Dict:
        """
        Compute information density by average embedding deviation after removing
        a contiguous block equal to `remove_frac` of the tokens, across `trials` random positions.

        Returns a dict with keys:
            - avg_deviation: float
            - std_deviation: float
            - trials: List[Dict] with per-trial details
            - tokens: int  (original token count)
            - removed_tokens_each: int
            - distance: str
            - model_name: str
        """
        tokens = _tokenize(text)
        n = len(tokens)
        if n == 0:
            return {
                "avg_deviation": 0.0,
                "std_deviation": 0.0,
                "trials": [],
                "tokens": 0,
                "removed_tokens_each": 0,
                "distance": self.distance,
                "model_name": self.model_name,
            }

        # Round up to ensure at least 1 token is removed
        k = max(1, int(math.ceil(remove_frac * n)))
        # Guard: if the doc is extremely short, still try removing at least one token
        k = min(k, n - 1) if n > 1 else 1

        # Compute original embedding once
        orig_emb = self._encode([text])[0].astype(np.float32)

        deviations: List[float] = []
        details: List[Dict] = []

        for t in range(trials):
            s, e = _pick_contiguous_block(n, k, self.rng)
            edited_tokens = tokens[:s] + tokens[e:]
            edited_text = _detokenize(edited_tokens)

            edited_emb = self._encode([edited_text])[0].astype(np.float32)
            dev = self._distance(orig_emb, edited_emb)
            deviations.append(dev)

            details.append({
                "trial": t + 1,
                "start_idx": s,
                "end_idx": e,
                "removed_span_tokens": k,
                "deviation": dev,
            })

        deviations_np = np.array(deviations, dtype=np.float32)
        result = {
            "avg_deviation": float(deviations_np.mean() if len(deviations_np) else 0.0),
            "std_deviation": float(deviations_np.std(ddof=0) if len(deviations_np) else 0.0),
            "trials": details,
            "tokens": n,
            "removed_tokens_each": k,
            "distance": self.distance,
            "model_name": self.model_name,
        }
        return result


def _cli():
    p = argparse.ArgumentParser(description="Information density by embedding deviation after 10% contiguous removal.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", type=str, help="Raw document text to evaluate")
    src.add_argument("--file", type=str, help="Path to a UTF-8 text file to evaluate")
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model name")
    p.add_argument("--trials", type=int, default=10, help="Number of random removals")
    p.add_argument("--remove-frac", type=float, default=0.10, help="Fraction (0-1) of tokens to remove each trial")
    p.add_argument("--distance", type=str, choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    text = args.text
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()

    evaluator = InformationDensityEvaluator(
        model_name=args.model,
        distance=args.distance,
        seed=args.seed,
    )
    result = evaluator.evaluate(text, trials=args.trials, remove_frac=args.remove_frac)
    # Pretty print JSON
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    _cli()

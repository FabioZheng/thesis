import math
from collections import Counter
from typing import Iterable, List
import numpy as np
import torch

class CompressionBanditAgent:
    """Simple contextual multi-armed bandit using linear UCB."""

    def __init__(self, rates: Iterable[int], alpha: float = 1.0):
        self.rates = list(rates)
        self.alpha = alpha
        self.A = {r: np.identity(1) for r in self.rates}
        self.b = {r: np.zeros((1, 1)) for r in self.rates}

    def _feat(self, perplexity: float) -> np.ndarray:
        return np.array([[perplexity]], dtype=float)

    def select_rate(self, perplexity: float) -> int:
        x = self._feat(perplexity)
        scores = {}
        for r in self.rates:
            A_inv = np.linalg.inv(self.A[r])
            theta = A_inv @ self.b[r]
            print(f"Rate {r} theta: {theta.ravel()[0]:.4f}")
            p = float(theta.T @ x + self.alpha * math.sqrt(x.T @ A_inv @ x))
            scores[r] = p
        return max(self.rates, key=lambda r: scores[r])

    def update(self, perplexity: float, rate: int, reward: float) -> None:
        x = self._feat(perplexity)
        self.A[rate] += x @ x.T
        self.b[rate] += reward * x


def batch_entropy(input_ids, attention_mask) -> List[float]:
    entropies = []
    for ids, mask in zip(input_ids, attention_mask):
        tokens = ids[mask.bool()].tolist()
        total = len(tokens)
        counts = Counter(tokens)
        probs = [c / total for c in counts.values() if c > 0]
        ent = -sum(p * math.log(p, 2) for p in probs)
        entropies.append(ent)
    return entropies


def batch_perplexity(model, input_ids, attention_mask) -> List[float]:
    """Compute the perplexity of each document in a batch.

    Parameters
    ----------
    model : transformers.PreTrainedModel or COCOM
        Language model used to compute the likelihood of the documents. If a
        ``COCOM`` instance is provided, its ``decoder`` will be used.
    input_ids : torch.LongTensor
        Token ids of shape ``(batch_size, seq_len)``.
    attention_mask : torch.LongTensor
        Attention mask with the same shape as ``input_ids``.

    Returns
    -------
    List[float]
        Perplexity score for every document.
    """

    # ``COCOM`` wraps a decoder model which exposes the standard causal LM
    # interface.  If present, compute perplexity with the decoder.
    lm = getattr(model, "decoder", model)

    device = next(lm.parameters()).device
    perplexities = []
    lm.eval()
    with torch.no_grad():
        for ids, mask in zip(input_ids, attention_mask):
            ids = ids.to(device)
            mask = mask.to(device)
            outputs = lm(
                input_ids=ids.unsqueeze(0),
                attention_mask=mask.unsqueeze(0),
                labels=ids.unsqueeze(0),
            )
            loss = outputs.loss
            perplexities.append(float(math.exp(loss.item())))
    return perplexities

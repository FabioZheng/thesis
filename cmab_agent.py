import math
from collections import Counter
from typing import Iterable, List
import numpy as np
import torch

class CompressionBanditAgent:
    """Simple contextual multi-armed bandit using linear UCB.

    The agent operates on a two-dimensional feature vector consisting of
    document entropy and normalized document length.  Both features are
    expected to be provided when selecting and updating an arm so the
    agent can jointly reason about uncertainty and size.
    """

    def __init__(self, rates: Iterable[int], alpha: float = 1.0, feature_dim: int = 2):
        self.rates = list(rates)
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.A = {r: np.identity(self.feature_dim) for r in self.rates}
        self.b = {r: np.zeros((self.feature_dim, 1)) for r in self.rates}

    def _feat(self, entropy: float, doc_length: float) -> np.ndarray:
        if self.feature_dim == 1:
            return np.array([[entropy]], dtype=float)
        if self.feature_dim == 2:
            return np.array([[entropy], [doc_length]], dtype=float)
        raise ValueError(f"Unsupported feature dimension: {self.feature_dim}")

    def select_rate(self, entropy: float, doc_length: float) -> int:
        x = self._feat(entropy, doc_length)
        scores = {}
        for r in self.rates:
            A_inv = np.linalg.inv(self.A[r])
            theta = A_inv @ self.b[r]
            p = float(theta.T @ x + self.alpha * math.sqrt(x.T @ A_inv @ x))
            scores[r] = p
        return max(self.rates, key=lambda r: scores[r])

    def update(self, entropy: float, doc_length: float, rate: int, reward: float) -> None:
        x = self._feat(entropy, doc_length)
        self.A[rate] += x @ x.T
        self.b[rate] += reward * x


def batch_perplexity(model, input_ids, attention_mask) -> List[float]:
    """Compute the perplexity of each document in a batch.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        Language model used to compute the likelihood of the documents.
    input_ids : torch.LongTensor
        Token ids of shape ``(batch_size, seq_len)``.
    attention_mask : torch.LongTensor
        Attention mask with the same shape as ``input_ids``.

    Returns
    -------
    List[float]
        Perplexity score for every document.
    """

    device = next(model.parameters()).device
    perplexities = []
    model.eval()
    with torch.no_grad():
        for ids, mask in zip(input_ids, attention_mask):
            ids = ids.to(device)
            mask = mask.to(device)
            outputs = model(input_ids=ids.unsqueeze(0), attention_mask=mask.unsqueeze(0), labels=ids.unsqueeze(0))
            loss = outputs.loss
            perplexities.append(float(math.exp(loss.item())))
    return perplexities
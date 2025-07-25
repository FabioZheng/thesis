from typing import List
from rouge import Rouge
import evaluate

from metrics import compute_rouge_scores

# Pre-load metrics
rouge = Rouge()
bertscore = evaluate.load("bertscore")


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    """Return average F1 BERTScore for predictions vs references."""
    scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    return sum(scores["f1"]) / len(scores["f1"])


def reward(predictions: List[str], references: List[str], compression_rate: int,
           alpha: float = 0.5, beta: float = 0.5, gamma: float = 0.1) -> float:
    """Compute combined reward used for bandit training."""
    bert_f1 = compute_bertscore(predictions, references)
    rouge_l = compute_rouge_scores(rouge, predictions, references)["Rouge-L"]
    compression_penalty = 1.0 / float(compression_rate)
    return alpha * bert_f1 + beta * rouge_l - gamma * compression_penalty

import numpy as np
from collections import Counter
from typing import List

import evaluate
import math
import numpy as np
import regex
import string
from rouge import Rouge


def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))



def em_single(prediction, ground_truth):
    return float(normalize(prediction) == normalize(ground_truth))


def exact_match_score(predictions, references):
    return np.mean([
        em_single(prediction, ground_truth)
        for ground_truth, prediction in zip(references, predictions)
    ])


def f1_single(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize(prediction).split()
    ground_truth_tokens = normalize(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_score(predictions: List[str], references: List[str]) -> float:
    return np.mean([
        f1_single(prediction, ground_truth)
        for prediction, ground_truth in zip(predictions, references)
    ])


def rouge_wrapper(rouge, prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def compute_rouge_scores(rouge, predictions, references):
    rouge1, rouge2, rougel = list(), list(), list()
    for ground_truths, predicition in zip(references, predictions):
        rouge1_, rouge2_, rougel_ =  rouge_wrapper(rouge, predicition, ground_truths)
        rouge1.append(rouge1_)
        rouge2.append(rouge2_)
        rougel.append(rougel_)
    return {'Rouge-1': np.mean(rouge1), 'Rouge-2': np.mean(rouge2), 'Rouge-L': np.mean(rougel)}


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



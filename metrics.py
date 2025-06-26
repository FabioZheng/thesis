import numpy as np 

import regex 
import string 

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
    return np.mean([em_single(prediction, ground_truth) for ground_truth, prediction in zip(references, predictions)])


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




from transformers import pipeline
from collections import defaultdict
from rouge import Rouge
from metrics import compute_rouge_scores  # Assumes this is your custom scorer returning a dict

# Initialize ROUGE and NER
rouge = Rouge()
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


def extract_entities(texts):
    """Extract grouped entities from each input text using BERT-based NER."""
    entities_per_text = []
    for text in texts:
        ner_results = ner_pipeline(text)
        entities = defaultdict(list)
        for ent in ner_results:
            entities[ent['entity_group']].append(ent['word'])
        entities_per_text.append(entities)
    return entities_per_text


def compute_plain_rouge(preds_str, labels_str):
    """Compute plain ROUGE on full text predictions vs. references."""
    return compute_rouge_scores(rouge, preds_str, labels_str)


def compute_ner_rouge(preds_str, labels_str):
    """Compute ROUGE over matched named entities between predictions and references."""
    pred_entities = extract_entities(preds_str)
    label_entities = extract_entities(labels_str)

    total_score = 0
    count = 0
    for pred_ents, label_ents in zip(pred_entities, label_entities):
        for ent_type in label_ents:
            label_text = " ".join(label_ents[ent_type])
            pred_text = " ".join(pred_ents.get(ent_type, []))
            if label_text:
                # Use compute_rouge_scores here as well
                scores = compute_rouge_scores(rouge, [pred_text], [label_text])
                total_score += scores["Rouge-L"]
                count += 1

    avg_ner_rouge_score = total_score / count if count > 0 else 0.0
    return {"ner_rougeL": avg_ner_rouge_score}


def main():
    # Sample predictions and references
    predictions = [
        "Barack Obama was born in Hawaii.",
        "Apple Inc. released the new iPhone in September.",
        "The Eiffel Tower is located in Paris."
    ]

    labels = [
        "Obama was born in Hawaii.",
        "The new iPhone was released by Apple in September.",
        "Paris is home to the Eiffel Tower."
    ]

    print("== Plain ROUGE Scores ==")
    plain_scores = compute_plain_rouge(predictions, labels)
    for k, v in plain_scores.items():
        print(f"{k}: {v:.4f}")

    print("\n== NER-based ROUGE Scores ==")
    ner_scores = compute_ner_rouge(predictions, labels)
    for k, v in ner_scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()

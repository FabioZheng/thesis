from sklearn.metrics import precision_recall_fscore_support

class RAGMetrics:
    @staticmethod
    def compute(predictions, references, questions=None):
        exact_match = sum([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)]) / len(predictions)
        f1 = precision_recall_fscore_support(
            [r.strip().lower() for r in references],
            [p.strip().lower() for p in predictions],
            average='micro'
        )[2]
        return {
            'exact_match': exact_match,
            'f1': f1,
        }
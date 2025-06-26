
from pyserini.index import IndexReader
from pyserini.search import SimpleSearcher


class Retrieve:
    def __init__(self, index_path=None, top_k=10):
        self.searcher = SimpleSearcher.from_prebuilt_index('robust04') if index_path is None else SimpleSearcher(index_path)
        self.searcher.set_bm25()
        self.top_k = top_k

    def retrieve(self, dataset, query_embeds_path=None, doc_embeds_path=None, top_k=10):
        queries = dataset['query']
        qids, docids, scores = [], [], []
        for idx, q in enumerate(queries):
            hits = self.searcher.search(q['text'], k=top_k)
            qids.append(str(idx))
            docids.append([h.docid for h in hits])
            scores.append([h.score for h in hits])
        return {'q_id': qids, 'doc_id': docids, 'score': scores}
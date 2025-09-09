# retrieval.py
from __future__ import annotations
import os
import sys
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

# Viz
from sklearn.decomposition import PCA
import plotly.express as px

# Optional Streamlit integration
try:
    import streamlit as st
except Exception:
    st = None


# -----------------------------
# Utilities
# -----------------------------
def _coerce_to_text(cell: Any) -> str:
    """
    Convert a dataframe cell into plain text.
    Handles dict-like MS MARCO 'passages' -> 'passage_text' list.
    """
    if cell is None:
        return ""

    # If already string, return stripped
    if isinstance(cell, str):
        # Try to parse JSON strings holding nested structures
        s = cell.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict) and "passage_text" in parsed:
                    return " ".join([str(x) for x in parsed.get("passage_text", [])])
                # If another text-like key is present, prefer it
                for k in ("text", "passage", "content", "body"):
                    if k in parsed and isinstance(parsed[k], (str, list)):
                        return " ".join(parsed[k]) if isinstance(parsed[k], list) else parsed[k]
            except Exception:
                pass
        return s

    # If dict-like (e.g., MS MARCO)
    if isinstance(cell, dict):
        if "passage_text" in cell:
            return " ".join([str(x) for x in cell.get("passage_text", [])])
        for k in ("text", "passage", "content", "body"):
            if k in cell:
                v = cell[k]
                return " ".join(v) if isinstance(v, list) else str(v)
        # Fallback to stringified dict
        return json.dumps(cell, ensure_ascii=False)

    # If list-like, join
    if isinstance(cell, (list, tuple)):
        return " ".join([_coerce_to_text(x) for x in cell])

    return str(cell)


def _sizeof_dict_bytes(d: Dict[str, Any], sample: int = 200) -> int:
    """
    Heuristic memory usage (bytes) of a dict {id: {embedding: np.ndarray, text: str}}.
    Samples up to `sample` items to estimate.
    """
    if not d:
        return 0
    keys = list(d.keys())
    n = len(keys)
    sample_n = min(sample, n)
    picked = keys[:sample_n]

    total = sys.getsizeof(d)
    for k in picked:
        v = d[k]
        total += sys.getsizeof(k)
        total += sys.getsizeof(v)
        if isinstance(v, dict):
            for kk, vv in v.items():
                total += sys.getsizeof(kk)
                if isinstance(vv, np.ndarray):
                    total += vv.nbytes
                else:
                    total += sys.getsizeof(vv)

    # scale up
    return int(total * (n / sample_n))


# -----------------------------
# Embedding model
# -----------------------------
@dataclass
class TextEmbedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    batch_size: int = 256
    device: Optional[str] = None  # e.g., "cpu", "cuda"

    def __post_init__(self):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return embs


# -----------------------------
# Corpus encoding & store
# -----------------------------
@dataclass
class CorpusEncoder:
    embedder: TextEmbedder

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        id_prefix: str = "doc",
    ) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, Any]]]:
        """
        Returns:
          - embeddings: (N, D) float32
          - doc_ids: list[str]
          - store: {doc_id: {"embedding": np.ndarray, "text": str}}
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in dataframe.")

        texts: List[str] = []
        doc_ids: List[str] = []

        for i, cell in enumerate(df[text_column].tolist()):
            text = _coerce_to_text(cell)
            texts.append(text)
            doc_ids.append(f"{id_prefix}-{i}")

        embeddings = self.embedder.encode(texts).astype(np.float32)

        store: Dict[str, Dict[str, Any]] = {}
        for i, did in enumerate(doc_ids):
            store[did] = {"embedding": embeddings[i], "text": texts[i]}

        return embeddings, doc_ids, store


# -----------------------------
# Retrieval (cosine similarity)
# -----------------------------
@dataclass
class CosineRetriever:
    embeddings: np.ndarray  # (N, D), L2-normalized if using cosine via dot
    doc_ids: List[str]
    store: Dict[str, Dict[str, Any]]

    def __post_init__(self):
        if self.embeddings.dtype != np.float32:
            self.embeddings = self.embeddings.astype(np.float32)

    def _cosine_topk(
        self, q: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        q should be L2-normalized. Returns list of (doc_id, score).
        """
        # dot product with normalized embeddings = cosine
        sims = self.embeddings @ q.astype(np.float32)
        topk_idx = np.argpartition(-sims, kth=min(k, len(sims)-1))[:k]
        topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
        return [(self.doc_ids[i], float(sims[i])) for i in topk_idx]

    def search(
        self, query: str, query_embedder: TextEmbedder, k: int = 5
    ) -> List[Tuple[str, float, str]]:
        q_emb = query_embedder.encode([query])[0].astype(np.float32)
        hits = self._cosine_topk(q_emb, k=k)
        # attach text
        return [(doc_id, score, self.store[doc_id]["text"]) for doc_id, score in hits]


# -----------------------------
# Visualization
# -----------------------------
def plot_embeddings_pca(
    embeddings: np.ndarray,
    doc_ids: List[str],
    sample: int = 1000,
    random_state: int = 42,
):
    """
    Reduces embeddings to 2D with PCA and returns a Plotly scatter figure.
    """
    n = len(doc_ids)
    idx = np.arange(n)
    if n > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(idx, size=sample, replace=False)

    sub_emb = embeddings[idx]
    sub_ids = [doc_ids[i] for i in idx]

    # PCA to 2D
    pca = PCA(n_components=2, random_state=random_state)
    xy = pca.fit_transform(sub_emb)

    df_plot = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "doc_id": sub_ids})
    fig = px.scatter(
        df_plot, x="x", y="y", hover_name="doc_id", title="Doc Embeddings (PCA sample)"
    )
    return fig


# -----------------------------
# Streamlit block (optional)
# -----------------------------
def render_embeddings_block(df: pd.DataFrame, candidate_text_cols: Optional[List[str]] = None):
    """
    Streamlit UI to:
      - pick a text column (defaults to 'text' or 'passage' if present)
      - build embeddings
      - show dict size + embeddings DF RAM size + embeddings Pickle disk size
      - display 2D scatter (PCA) for 1,000 random docs
      - run simple retrieval over the built index
    """
    if st is None:
        raise RuntimeError("Streamlit not available. Install streamlit to use this UI helper.")

    st.header("🧩 Embeddings & Retrieval")

    # Pick text column
    all_cols = list(df.columns)
    default_col = None
    for cand in (["text", "passage", "passages", "content", "body"] + (candidate_text_cols or [])):
        if cand in all_cols:
            default_col = cand
            break
    text_col = st.selectbox(
        "Select text column to embed",
        options=all_cols,
        index=(all_cols.index(default_col) if default_col in all_cols else 0)
    )

    # Model picker
    model_name = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    batch_size = st.number_input("Batch size", min_value=8, max_value=2048, value=256, step=8)

    if st.button("Build embeddings"):
        with st.spinner("Encoding corpus..."):
            embedder = TextEmbedder(model_name=model_name, normalize=True, batch_size=batch_size)
            encoder = CorpusEncoder(embedder=embedder)
            embeddings, doc_ids, store = encoder.build_from_dataframe(df, text_col, id_prefix="doc")

        # --- Sizes & footprints ---
        # 1) dict store approx RAM
        dict_bytes = _sizeof_dict_bytes(store, sample=200)
        dict_mb = dict_bytes / (1024 * 1024)

        # 2) embeddings as ndarray RAM size
        emb_nd_mb = embeddings.nbytes / (1024 * 1024)

        # 3) embeddings as DataFrame RAM size
        emb_df = pd.DataFrame(embeddings)
        emb_df_mb = emb_df.memory_usage(deep=True).sum() / (1024 * 1024)

        # 4) pickle disk size for embeddings ndarray
        tmp_pkl = "embeddings_tmp.pkl"
        try:
            import pickle
            with open(tmp_pkl, "wb") as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            emb_pkl_mb = os.path.getsize(tmp_pkl) / (1024 * 1024)
        finally:
            try:
                os.remove(tmp_pkl)
            except Exception:
                pass

        st.success(
            f"Created embeddings for {len(doc_ids):,} docs "
            f"(dim={embeddings.shape[1]})."
        )

        # Show metrics in a compact row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Store dict (approx RAM)", f"{dict_mb:.2f} MB")
        with c2:
            st.metric("Embeddings ndarray (RAM)", f"{emb_nd_mb:.2f} MB")
        with c3:
            st.metric("Embeddings DataFrame (RAM)", f"{emb_df_mb:.2f} MB")
        with c4:
            st.metric("Embeddings Pickle (Disk)", f"{emb_pkl_mb:.2f} MB")

        # Scatter (PCA)
        fig = plot_embeddings_pca(embeddings, doc_ids, sample=1000)
        st.plotly_chart(fig, use_container_width=True)

        # Keep in session for live retrieval
        st.session_state["__embeddings"] = embeddings
        st.session_state["__doc_ids"] = doc_ids
        st.session_state["__store"] = store
        st.session_state["__embedder_name"] = model_name

    # Retrieval demo (if available)
    if "__embeddings" in st.session_state:
        st.subheader("🔎 Try a query")
        q = st.text_input("Query text", value="how to relieve sore throat")
        k = st.number_input("Top-K", min_value=1, max_value=50, value=5)
        if st.button("Search"):
            with st.spinner("Embedding query & searching..."):
                q_embedder = TextEmbedder(model_name=st.session_state["__embedder_name"], normalize=True)
                retriever = CosineRetriever(
                    embeddings=st.session_state["__embeddings"],
                    doc_ids=st.session_state["__doc_ids"],
                    store=st.session_state["__store"],
                )
                hits = retriever.search(q, q_embedder, k=int(k))

            for rank, (doc_id, score, text) in enumerate(hits, 1):
                st.markdown(f"**{rank}. {doc_id}** — cos={score:.4f}")
                st.write(text[:600] + ("..." if len(text) > 600 else ""))
                st.markdown("---")

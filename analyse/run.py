import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
import re
import json
from collections import Counter
from sklearn.datasets import fetch_20newsgroups, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
# Add near top of run.py
from info_density import InformationDensityEvaluator


# HF datasets (optional)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- Robust NLTK sentence tokenizer setup ---
def _ensure_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass

_ensure_nltk_punkt()

def fallback_sentence_split(text: str):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p for p in parts if p]

class DatasetAnalyzer:
    def __init__(self):
        self.dataset = None
        self.dataset_name = None
        self.data_dict = None
        self.text_columns = []
        self.target_column = None

    def load_sample_dataset(self, dataset_name):
        self.dataset_name = dataset_name
        if dataset_name == "20newsgroups":
            data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
            df = pd.DataFrame({
                'text': data.data,
                'target': data.target,
                'target_names': [data.target_names[i] for i in data.target]
            })
            self.text_columns = ['text']
            self.target_column = 'target'
        elif dataset_name == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_names'] = [data.target_names[i] for i in data.target]
            self.target_column = 'target'
        elif dataset_name == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_names'] = [data.target_names[i] for i in data.target]
            self.target_column = 'target'
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['target_names'] = [data.target_names[i] for i in data.target]
            self.target_column = 'target'
        else:
            df = pd.DataFrame()
        self.dataset = df
        return df

    def load_custom_dataset(self, file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
                df = pd.read_json(file_path, lines=file_path.endswith('.jsonl'))
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                st.error("Unsupported file format. Please use CSV, JSON/JSONL, or Excel files.")
                return None

            self.dataset = df
            self.dataset_name = os.path.basename(file_path)

            # Auto-detect text columns
            self.text_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 20:
                        self.text_columns.append(col)
            return df
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None

    def load_hf_dataset(self, path, name=None, split="train", sample_rows=None, text_guess_threshold=20):
        if not HF_AVAILABLE:
            st.error("Install `datasets`: pip install datasets")
            return None
        try:
            ds = load_dataset(path, name=name, split=split)
        except Exception as e:
            st.error(f"Failed to load dataset '{path}' (name={name}, split={split}): {e}")
            return None

        if sample_rows is not None and sample_rows > 0:
            try:
                ds_small = ds.shuffle(seed=42).select(range(min(sample_rows, len(ds))))
            except Exception:
                ds_small = ds.select(range(min(sample_rows, len(ds))))
            df = ds_small.to_pandas()
        else:
            df = ds.to_pandas()

        self.dataset = df
        self.dataset_name = f"HF:{path}" + (f"/{name}" if name else "") + f" [{split}]"

        self.text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len and avg_len > text_guess_threshold:
                    self.text_columns.append(col)

        self.target_column = None
        for guess in ["label", "target", "answers", "relevance", "is_selected"]:
            if guess in df.columns:
                self.target_column = guess
                break
        return df

    def get_basic_info(self):
        if self.dataset is None:
            return None
        return {
            'total_rows': len(self.dataset),
            'total_columns': len(self.dataset.columns),
            'column_names': list(self.dataset.columns),
            'column_types': dict(self.dataset.dtypes.astype(str)),
            'missing_values': dict(self.dataset.isnull().sum()),
            'dataset_name': self.dataset_name
        }

    def calculate_memory_usage(self):
        if self.dataset is None:
            return None
        memory_usage_mb = self.dataset.memory_usage(deep=True).sum() / 1024 / 1024
        self.data_dict = self.dataset.to_dict('records')

        dict_size_bytes = sys.getsizeof(self.data_dict)
        for record in self.data_dict[: min(100, len(self.data_dict))]:
            dict_size_bytes += sys.getsizeof(record)
            for key, value in record.items():
                dict_size_bytes += sys.getsizeof(key) + sys.getsizeof(value)

        multiplier = (len(self.data_dict) / min(100, len(self.data_dict))) if self.data_dict else 1
        dict_size_mb = (dict_size_bytes * multiplier) / 1024 / 1024

        temp_file = 'temp_dataset.pkl'
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(self.data_dict, f)
            disk_size_mb = os.path.getsize(temp_file) / 1024 / 1024
            os.remove(temp_file)
        except Exception:
            disk_size_mb = dict_size_mb * 0.3

        return {
            'dataframe_memory_mb': memory_usage_mb,
            'dictionary_memory_mb': dict_size_mb,
            'disk_size_mb': disk_size_mb
        }

    def analyze_class_distribution(self):
        if self.dataset is None or self.target_column is None or self.target_column not in self.dataset.columns:
            return None
        class_counts = self.dataset[self.target_column].value_counts(dropna=False)
        if 'target_names' in self.dataset.columns:
            class_names = self.dataset.groupby(self.target_column)['target_names'].first()
            class_info = pd.DataFrame({
                'class_id': class_counts.index,
                'class_name': [class_names.get(i, f'Class_{i}') for i in class_counts.index],
                'count': class_counts.values,
                'percentage': (class_counts.values / len(self.dataset)) * 100
            })
        else:
            class_info = pd.DataFrame({
                'class_id': class_counts.index,
                'count': class_counts.values,
                'percentage': (class_counts.values / len(self.dataset)) * 100
            })
        return class_info

    def _safe_sent_tokenize(self, text: str):
        try:
            return sent_tokenize(text)
        except Exception:
            return fallback_sentence_split(text)

    def _coerce_nested_text(self, cell):
        """
        For MS MARCO-like rows where a column may contain a dict/JSON with 'passage_text' list.
        """
        if isinstance(cell, dict):
            if "passage_text" in cell:
                return " ".join([str(x) for x in cell.get("passage_text", [])])
            for k in ("text", "passage", "content", "body"):
                if k in cell:
                    v = cell[k]
                    return " ".join(v) if isinstance(v, list) else str(v)
            return json.dumps(cell, ensure_ascii=False)
        if isinstance(cell, str) and cell.strip().startswith("{"):
            try:
                parsed = json.loads(cell)
                return self._coerce_nested_text(parsed)
            except Exception:
                return cell
        if isinstance(cell, (list, tuple)):
            return " ".join([self._coerce_nested_text(x) for x in cell])
        return str(cell)

    def analyze_text_statistics(self):
        if self.dataset is None or not self.text_columns:
            return None

        text_stats = {}
        for col in self.text_columns:
            if col not in self.dataset.columns:
                continue

            # IMPORTANT: don't stringify too early; keep objects to unpack nested
            col_series_raw = self.dataset[col].dropna()
            if col_series_raw.empty:
                continue

            col_stats = {
                'sentences_per_doc': [],
                'sentence_lengths': [],
                'doc_sentence_stats': []
            }

            for cell in col_series_raw:
                doc = self._coerce_nested_text(cell)  # unpack nested dicts/jsons if present

                sentences = self._safe_sent_tokenize(doc)
                num_sentences = len(sentences)
                col_stats['sentences_per_doc'].append(num_sentences)

                doc_sentence_lengths = [len(s.strip().split()) for s in sentences if s.strip()]
                col_stats['sentence_lengths'].extend(doc_sentence_lengths)

                if doc_sentence_lengths:
                    doc_stats = {
                        'mean_length': float(np.mean(doc_sentence_lengths)),
                        'std_length': float(np.std(doc_sentence_lengths)),
                        'min_length': int(np.min(doc_sentence_lengths)),
                        'max_length': int(np.max(doc_sentence_lengths)),
                        'num_sentences': int(num_sentences)
                    }
                else:
                    doc_stats = {'mean_length': 0.0, 'std_length': 0.0, 'min_length': 0, 'max_length': 0, 'num_sentences': 0}
                col_stats['doc_sentence_stats'].append(doc_stats)

            sentences_per_doc = col_stats['sentences_per_doc']
            sentence_lengths = col_stats['sentence_lengths']

            overall_stats = {
                'sentences_per_doc': {
                    'mean': float(np.mean(sentences_per_doc)) if sentences_per_doc else 0.0,
                    'std': float(np.std(sentences_per_doc)) if sentences_per_doc else 0.0,
                    'min': int(np.min(sentences_per_doc)) if sentences_per_doc else 0,
                    'max': int(np.max(sentences_per_doc)) if sentences_per_doc else 0,
                    'distribution': sentences_per_doc
                },
                'sentence_lengths': {
                    'mean': float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
                    'std': float(np.std(sentence_lengths)) if sentence_lengths else 0.0,
                    'min': int(np.min(sentence_lengths)) if sentence_lengths else 0,
                    'max': int(np.max(sentence_lengths)) if sentence_lengths else 0,
                    'distribution': sentence_lengths
                },
                'doc_stats': col_stats['doc_sentence_stats']
            }
            text_stats[col] = overall_stats

        return text_stats


def create_visualizations(analyzer):
    st.header("📊 Dataset Overview")
    basic_info = analyzer.get_basic_info()
    memory_info = analyzer.calculate_memory_usage()

    if basic_info and memory_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Rows", f"{basic_info['total_rows']:,}")
        with col2: st.metric("Total Columns", basic_info['total_columns'])
        with col3: st.metric("Memory Usage", f"{memory_info['dataframe_memory_mb']:.2f} MB")
        with col4: st.metric("Disk Size", f"{memory_info['disk_size_mb']:.2f} MB")

    st.subheader("💾 Memory Usage Comparison")
    if memory_info:
        memory_data = pd.DataFrame({
            'Storage Type': ['DataFrame (RAM)', 'Dictionary (RAM)', 'Pickled (Disk)'],
            'Size (MB)': [
                memory_info['dataframe_memory_mb'],
                memory_info['dictionary_memory_mb'],
                memory_info['disk_size_mb']
            ]
        })
        fig = px.bar(memory_data, x='Storage Type', y='Size (MB)', title="Memory Usage by Storage Type", color='Storage Type')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Column Information")
    if basic_info:
        col_info = pd.DataFrame({
            'Column': basic_info['column_names'],
            'Data Type': [basic_info['column_types'][c] for c in basic_info['column_names']],
            'Missing Values': [basic_info['missing_values'][c] for c in basic_info['column_names']]
        })
        st.dataframe(col_info, use_container_width=True)

    # Class Distribution — skip for HF datasets
    if analyzer.dataset_name and str(analyzer.dataset_name).startswith("HF:"):
        pass
    else:
        class_info = analyzer.analyze_class_distribution()
        if class_info is not None and not class_info.empty:
            st.subheader("🎯 Class Distribution")
            c1, c2 = st.columns(2)
            name_field = 'class_name' if 'class_name' in class_info.columns else 'class_id'
            with c1:
                fig = px.pie(class_info, values='count', names=name_field, title="Class Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(class_info, x=name_field, y='count', title="Class Counts")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(class_info, use_container_width=True)

    # Text Analysis
    text_stats = analyzer.analyze_text_statistics()
    st.header("📝 Text Analysis")
    if not analyzer.text_columns:
        st.info("No text columns selected. Pick one or more in the sidebar to see sentence statistics.")
    elif not text_stats:
        st.warning("No sentence statistics to show (selected text columns may be empty).")
    else:
        for col_name, stats in text_stats.items():
            st.subheader(f"Text Column: {col_name}")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Avg Sentences/Doc", f"{stats['sentences_per_doc']['mean']:.2f}")
            with c2: st.metric("Std Sentences/Doc", f"{stats['sentences_per_doc']['std']:.2f}")
            with c3: st.metric("Min Sentences/Doc", int(stats['sentences_per_doc']['min']))
            with c4: st.metric("Max Sentences/Doc", int(stats['sentences_per_doc']['max']))

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Avg Sentence Length", f"{stats['sentence_lengths']['mean']:.2f} words")
            with c2: st.metric("Std Sentence Length", f"{stats['sentence_lengths']['std']:.2f} words")
            with c3: st.metric("Min Sentence Length", f"{int(stats['sentence_lengths']['min'])} words")
            with c4: st.metric("Max Sentence Length", f"{int(stats['sentence_lengths']['max'])} words")

            cc1, cc2 = st.columns(2)
            if stats['sentences_per_doc']['distribution']:
                with cc1:
                    fig = px.histogram(x=stats['sentences_per_doc']['distribution'],
                                       title="Distribution of Sentences per Document",
                                       labels={'x': 'Number of Sentences', 'y': 'Frequency'})
                    st.plotly_chart(fig, use_container_width=True)
            if stats['sentence_lengths']['distribution']:
                with cc2:
                    lengths = stats['sentence_lengths']['distribution']
                    if len(lengths) > 10000:
                        lengths = np.random.choice(lengths, 10000, replace=False)
                    fig = px.histogram(x=lengths,
                                       title="Distribution of Sentence Lengths",
                                       labels={'x': 'Sentence Length (words)', 'y': 'Frequency'})
                    st.plotly_chart(fig, use_container_width=True)

            if stats['doc_stats']:
                doc_df = pd.DataFrame(stats['doc_stats'])
                st.subheader(f"Document-level Statistics for {col_name}")
                summary_stats = pd.DataFrame({
                    'Metric': ['Mean Sentence Length', 'Std Sentence Length', 'Min Sentence Length', 'Max Sentence Length'],
                    'Document Average': [
                        doc_df['mean_length'].mean(),
                        doc_df['std_length'].mean(),
                        doc_df['min_length'].mean(),
                        doc_df['max_length'].mean()
                    ],
                    'Document Std': [
                        doc_df['mean_length'].std(),
                        doc_df['std_length'].std(),
                        doc_df['min_length'].std(),
                        doc_df['max_length'].std()
                    ]
                })
                st.dataframe(summary_stats, use_container_width=True)


# ---------- PERSISTENCE HELPERS ----------
def _persist_after_load(analyzer: DatasetAnalyzer):
    st.session_state["_ds"] = analyzer.dataset
    st.session_state["_ds_name"] = analyzer.dataset_name
    st.session_state["_text_cols"] = analyzer.text_columns
    st.session_state["_target_col"] = analyzer.target_column

def _restore_into(analyzer: DatasetAnalyzer):
    if "_ds" in st.session_state and st.session_state["_ds"] is not None:
        analyzer.dataset = st.session_state["_ds"]
        analyzer.dataset_name = st.session_state.get("_ds_name")
        analyzer.text_columns = st.session_state.get("_text_cols", [])
        analyzer.target_column = st.session_state.get("_target_col")


def main():
    st.set_page_config(page_title="ML Dataset Analyzer", page_icon="📊", layout="wide")
    st.title("🔍 ML Dataset Analyzer")
    st.markdown("Comprehensive analysis of machine learning datasets with visual insights")

    analyzer = DatasetAnalyzer()
    _restore_into(analyzer)  # restore persisted dataset & choices if any

    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")

    dataset_source = st.sidebar.radio(
        "Choose dataset source:",
        ["Sample Datasets", "Upload Custom Dataset", "Hugging Face Datasets (load_dataset)"],
        key="__src_radio"
    )

    if dataset_source == "Sample Datasets":
        dataset_name = st.sidebar.selectbox(
            "Select a sample dataset:",
            ["20newsgroups", "iris", "wine", "breast_cancer"],
            key="__sample_select"
        )
        if st.sidebar.button("Load Dataset", key="__load_sample"):
            with st.spinner("Loading dataset..."):
                analyzer.load_sample_dataset(dataset_name)
                st.success(f"Loaded {dataset_name} dataset!")
                _persist_after_load(analyzer)

    elif dataset_source == "Upload Custom Dataset":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset",
            type=['csv', 'json', 'jsonl', 'xlsx'],
            key="__uploader"
        )
        if uploaded_file is not None and st.sidebar.button("Load Dataset", key="__load_upload"):
            with st.spinner("Loading dataset..."):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                analyzer.load_custom_dataset(temp_path)
                os.remove(temp_path)
                st.success(f"Loaded {uploaded_file.name}!")
                _persist_after_load(analyzer)

    else:  # Hugging Face
        st.sidebar.caption("Load datasets from the Hugging Face Hub via `datasets.load_dataset`.")
        if not HF_AVAILABLE:
            st.sidebar.error("Install the 'datasets' library to use this source: pip install datasets")
        dataset_path = st.sidebar.text_input("Dataset path (e.g., ms_marco)", value=st.session_state.get("__hf_path", ""), key="__hf_path")
        dataset_name = st.sidebar.text_input("Config / Name (optional)", value=st.session_state.get("__hf_name", ""), key="__hf_name")
        dataset_split = st.sidebar.text_input("Split", value=st.session_state.get("__hf_split", "train"), key="__hf_split")
        sample_rows = st.sidebar.number_input("Sample up to N rows (0 = all)", min_value=0, value=20000, step=1000, key="__hf_sample")

        if st.sidebar.button("Load HF Dataset", key="__load_hf"):
            if not dataset_path:
                st.sidebar.error("Please enter a dataset path (e.g., ms_marco).")
            else:
                with st.spinner("Loading from Hugging Face..."):
                    analyzer.load_hf_dataset(
                        path=dataset_path,
                        name=dataset_name if dataset_name.strip() else None,
                        split=dataset_split.strip() or "train",
                        sample_rows=sample_rows if sample_rows > 0 else None
                    )
                    if analyzer.dataset is not None:
                        st.success(f"Loaded {analyzer.dataset_name}!")
                        _persist_after_load(analyzer)

    # If we have a dataset, render config & visuals
    if analyzer.dataset is not None:
        st.sidebar.header("Column Configuration")

        all_columns = list(analyzer.dataset.columns)

        # Use session defaults if present; otherwise analyzer defaults
        default_text_cols = st.session_state.get("_text_cols", analyzer.text_columns)
        selected_text_cols = st.sidebar.multiselect(
            "Select text columns for analysis:",
            all_columns,
            default=[c for c in default_text_cols if c in all_columns],
            key="__text_cols"
        )
        analyzer.text_columns = selected_text_cols
        st.session_state["_text_cols"] = selected_text_cols  # persist

        target_options = ["None"] + all_columns
        default_target = st.session_state.get("_target_col", analyzer.target_column)
        default_idx = target_options.index(default_target) if default_target in target_options else 0
        selected_target = st.sidebar.selectbox(
            "Select target column (for classification):",
            target_options,
            index=default_idx,
            key="__target_col"
        )
        analyzer.target_column = selected_target if selected_target != "None" else None
        st.session_state["_target_col"] = analyzer.target_column  # persist

        # Create visualizations
        create_visualizations(analyzer)

                # Raw data preview
        st.header("🔍 Data Preview")
        st.dataframe(analyzer.dataset.head(100), use_container_width=True)

        # --- Information Density (average across docs) ---
        st.header("🧠 Information Density")
        from info_density import InformationDensityEvaluator  # local import to keep changes minimal

        if analyzer.text_columns:
            # Let the user pick which text column to evaluate
            id_text_col = st.selectbox(
                "Text column for information density:",
                analyzer.text_columns,
                index=0,
                help="We will compute per-doc deviations on this column and show the dataset average."
            )
            # Config knobs
            c1, c2, c3 = st.columns(3)
            with c1:
                trials = st.number_input("Trials per doc", min_value=3, max_value=50, value=10, step=1)
            with c2:
                remove_frac = st.slider("Removal fraction per trial", min_value=0.05, max_value=0.30, value=0.10, step=0.01)
            with c3:
                max_docs = st.number_input("Max docs to evaluate (for speed)", min_value=10, max_value=5000, value=500, step=10,
                                           help="Set a cap to avoid very long runs on huge datasets.")

            if st.button("Compute average information density"):
                docs_series = analyzer.dataset[id_text_col].dropna().astype(str)
                if len(docs_series) > max_docs:
                    docs_series = docs_series.head(max_docs)

                evaluator = InformationDensityEvaluator(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                deviations = []
                with st.spinner(f"Computing info density on up to {len(docs_series)} docs..."):
                    for doc in docs_series:
                        try:
                            res = evaluator.evaluate(doc, trials=int(trials), remove_frac=float(remove_frac))
                            deviations.append(res["avg_deviation"])
                        except Exception:
                            # Skip problematic rows quietly
                            continue

                if deviations:
                    avg_density = float(np.mean(deviations))
                    std_density = float(np.std(deviations))
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Average Information Density", f"{avg_density:.4f}")
                    with c2:
                        st.metric("Std. Dev.", f"{std_density:.4f}")
                    st.caption("Density = mean embedding deviation after removing a 10% contiguous span, averaged over trials:contentReference[oaicite:2]{index=2}.")
                else:
                    st.info("No deviations computed (empty/invalid docs?). Try a different column or adjust settings.")
        else:
            st.info("Select at least one text column in the sidebar to compute information density.")

        # Retrieval sandbox (embedding UI)
        st.header("🧭 Retrieval Sandbox")
        from retrieval import render_embeddings_block, render_clustering_block
        render_embeddings_block(analyzer.dataset, candidate_text_cols=analyzer.text_columns)
        render_clustering_block()


if __name__ == "__main__":
    main()

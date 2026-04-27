from __future__ import annotations

import pandas as pd
import streamlit as st

from embedding_explorer.core import EmbeddingConfig, build_explorer, make_plot

DEFAULT_TEXT = """Retrieval augmented generation connects language models to private documents.
Vector databases store embeddings for semantic search.
UMAP preserves local neighborhoods when projecting high dimensional data.
t-SNE is useful for exploring clusters in embedding spaces.
Prompt engineering changes model behavior through carefully designed instructions.
Evaluation harnesses compare model outputs against benchmark tasks.
Chinese API integrations can make multilingual assistants practical.
Transformers use attention to combine token context.
OpenAI and Anthropic models can both power GenAI prototypes.
Local embeddings are useful when privacy or cost matters."""

st.set_page_config(page_title="Embedding Explorer", page_icon="🧭", layout="wide")
st.title("🧭 Embedding Explorer")
st.caption("Visualize how text samples cluster after embedding and dimensionality reduction.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Embedding backend", ["tfidf-svd", "sentence-transformers", "openai"], help="tfidf-svd runs locally without API keys.")
    reducer = st.selectbox("Reducer", ["pca", "umap", "tsne"])
    dimensions = st.radio("Dimensions", [2, 3], horizontal=True)
    random_state = st.number_input("Random seed", value=42, step=1)
    sentence_model = st.text_input("SentenceTransformer model", "sentence-transformers/all-MiniLM-L6-v2")
    openai_model = st.text_input("OpenAI embedding model", "text-embedding-3-small")

uploaded = st.file_uploader("Upload CSV with a text column and optional label column", type=["csv"])
col_a, col_b = st.columns([2, 1])

if uploaded:
    df = pd.read_csv(uploaded)
    with col_a:
        text_col = st.selectbox("Text column", list(df.columns), index=list(df.columns).index("text") if "text" in df.columns else 0)
    with col_b:
        label_options = ["(none)"] + list(df.columns)
        label_col = st.selectbox("Label column", label_options, index=label_options.index("label") if "label" in label_options else 0)
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = None if label_col == "(none)" else df[label_col].fillna("unlabeled").astype(str).tolist()
else:
    raw_text = st.text_area("Paste one text sample per line", DEFAULT_TEXT, height=240)
    texts = [line.strip() for line in raw_text.splitlines() if line.strip()]
    labels = ["pasted"] * len(texts)

if st.button("Build visualization", type="primary"):
    try:
        with st.spinner("Embedding and reducing text samples..."):
            config = EmbeddingConfig(
                backend=backend,
                sentence_model=sentence_model,
                openai_model=openai_model,
                random_state=int(random_state),
            )
            result = build_explorer(texts, labels=labels, config=config, reducer=reducer, dimensions=dimensions)
            fig = make_plot(result)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Coordinates")
        st.dataframe(result.frame, use_container_width=True)
        st.download_button("Download coordinates CSV", result.frame.to_csv(index=False), file_name="embedding_coordinates.csv", mime="text/csv")
    except Exception as exc:
        st.error(str(exc))
        st.info("Tip: choose tfidf-svd + PCA for the most dependency-light local demo.")
else:
    st.info("Choose settings, then click **Build visualization**. The default backend runs locally without API keys.")

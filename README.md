# Embedding Explorer

Interactive GenAI demo for turning text into embeddings and exploring semantic structure in 2D/3D.

It supports:

- **Sentence Transformers** embeddings for high-quality local semantic vectors
- **OpenAI embeddings** when `OPENAI_API_KEY` is available
- **TF-IDF + SVD fallback** so the demo runs without API keys or model downloads
- **UMAP**, **t-SNE**, and **PCA** dimensionality reduction
- Interactive **Streamlit + Plotly** visualization
- CLI export to HTML/CSV for reproducible experiments

## Quick start

```bash
git clone https://github.com/kevinmalana/embedding-explorer.git
cd embedding-explorer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL, paste text lines, choose an embedding backend, and explore clusters.

## CLI demo

```bash
python -m embedding_explorer.cli \
  --input data/sample_texts.csv \
  --text-column text \
  --label-column label \
  --embedding-backend tfidf-svd \
  --reducer pca \
  --output-html out/explorer.html \
  --output-csv out/coordinates.csv
```

For semantic transformer embeddings:

```bash
python -m embedding_explorer.cli --embedding-backend sentence-transformers --reducer umap
```

For OpenAI embeddings:

```bash
export OPENAI_API_KEY=sk-...
python -m embedding_explorer.cli --embedding-backend openai --openai-model text-embedding-3-small
```

## Input format

CSV files should include a text column and may include a label column:

```csv
text,label
"Retrieval augmented generation grounds answers in documents",RAG
"Transformers use attention to mix token information",LLM
```

You can also paste one text per line in the Streamlit app.

## How it works

1. Load text samples from CSV or the built-in playground data.
2. Generate vector embeddings.
3. Project vectors into 2D or 3D with UMAP/t-SNE/PCA.
4. Render an interactive Plotly scatter plot with hoverable source text.

## Project structure

```text
embedding_explorer/
  cli.py          # command-line exporter
  core.py         # embeddings, reducers, plotting helpers
app.py            # Streamlit interface
data/             # sample dataset
tests/            # runnable smoke tests
```

## Notes

- `tfidf-svd` is intentionally included as a zero-key fallback; it is less semantic than transformer embeddings but ideal for quick demos.
- UMAP is optional at runtime. If unavailable, choose PCA or t-SNE.
- t-SNE is stochastic; use `--random-state` for repeatability.

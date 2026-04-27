from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

EmbeddingBackend = Literal["tfidf-svd", "sentence-transformers", "openai"]
ReducerName = Literal["pca", "tsne", "umap"]


@dataclass
class EmbeddingConfig:
    backend: EmbeddingBackend = "tfidf-svd"
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    svd_components: int = 64
    random_state: int = 42


@dataclass
class ExplorerResult:
    frame: pd.DataFrame
    embeddings: np.ndarray
    reducer: str
    embedding_backend: str


def clean_texts(texts: Iterable[str]) -> list[str]:
    cleaned = [str(t).strip() for t in texts if str(t).strip()]
    if len(cleaned) < 2:
        raise ValueError("Provide at least two non-empty text samples.")
    return cleaned


def embed_texts(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    if config.backend == "tfidf-svd":
        return _embed_tfidf_svd(texts, config)
    if config.backend == "sentence-transformers":
        return _embed_sentence_transformers(texts, config)
    if config.backend == "openai":
        return _embed_openai(texts, config)
    raise ValueError(f"Unknown embedding backend: {config.backend}")


def _embed_tfidf_svd(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    max_components = max(1, min(config.svd_components, len(texts) - 1))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    sparse = vectorizer.fit_transform(texts)
    if sparse.shape[1] <= 1:
        return sparse.toarray().astype(float)
    n_components = min(max_components, sparse.shape[1] - 1)
    pipeline = make_pipeline(TruncatedSVD(n_components=n_components, random_state=config.random_state), Normalizer(copy=False))
    return pipeline.fit_transform(sparse).astype(float)


def _embed_sentence_transformers(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers or use --embedding-backend tfidf-svd.") from exc
    model = SentenceTransformer(config.sentence_model)
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=float)


def _embed_openai(texts: list[str], config: EmbeddingConfig) -> np.ndarray:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai or use --embedding-backend tfidf-svd.") from exc
    client = OpenAI()
    response = client.embeddings.create(model=config.openai_model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype=float)


def reduce_embeddings(embeddings: np.ndarray, reducer: ReducerName = "pca", dimensions: int = 2, random_state: int = 42) -> np.ndarray:
    if dimensions not in (2, 3):
        raise ValueError("dimensions must be 2 or 3")
    if embeddings.shape[0] <= dimensions:
        raise ValueError(f"Need more than {dimensions} samples to make a {dimensions}D plot.")

    if reducer == "pca":
        model = PCA(n_components=dimensions, random_state=random_state)
        return model.fit_transform(embeddings)

    if reducer == "tsne":
        perplexity = max(2, min(30, (embeddings.shape[0] - 1) // 3 or 2))
        model = TSNE(n_components=dimensions, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
        return model.fit_transform(embeddings)

    if reducer == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("Install umap-learn or choose reducer='pca' / 'tsne'.") from exc
        n_neighbors = max(2, min(15, embeddings.shape[0] - 1))
        model = umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors, min_dist=0.08, metric="cosine", random_state=random_state)
        return model.fit_transform(embeddings)

    raise ValueError(f"Unknown reducer: {reducer}")


def build_explorer(
    texts: Iterable[str],
    labels: Iterable[str] | None = None,
    config: EmbeddingConfig | None = None,
    reducer: ReducerName = "pca",
    dimensions: int = 2,
) -> ExplorerResult:
    config = config or EmbeddingConfig()
    cleaned = clean_texts(texts)
    labels_list = list(labels) if labels is not None else ["sample"] * len(cleaned)
    if len(labels_list) != len(cleaned):
        raise ValueError("labels must have the same length as texts")
    embeddings = embed_texts(cleaned, config)
    coords = reduce_embeddings(embeddings, reducer=reducer, dimensions=dimensions, random_state=config.random_state)
    data = {"text": cleaned, "label": labels_list, "x": coords[:, 0], "y": coords[:, 1]}
    if dimensions == 3:
        data["z"] = coords[:, 2]
    frame = pd.DataFrame(data)
    return ExplorerResult(frame=frame, embeddings=embeddings, reducer=reducer, embedding_backend=config.backend)


def make_plot(result: ExplorerResult, title: str | None = None):
    title = title or f"Embedding Explorer: {result.embedding_backend} + {result.reducer.upper()}"
    if "z" in result.frame.columns:
        return px.scatter_3d(result.frame, x="x", y="y", z="z", color="label", hover_data={"text": True}, title=title)
    return px.scatter(result.frame, x="x", y="y", color="label", hover_data={"text": True}, title=title)

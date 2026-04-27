import pandas as pd

from embedding_explorer.core import EmbeddingConfig, build_explorer, clean_texts


def test_clean_texts_filters_empty_values():
    assert clean_texts([" hello ", "", "world"]) == ["hello", "world"]


def test_build_explorer_with_local_backend_pca():
    texts = [
        "RAG retrieves documents",
        "Vector search finds similar text",
        "Prompting changes model behavior",
        "Evaluation compares model answers",
        "UMAP visualizes embeddings",
    ]
    labels = ["a", "a", "b", "b", "c"]
    result = build_explorer(texts, labels=labels, config=EmbeddingConfig(backend="tfidf-svd"), reducer="pca")
    assert list(result.frame.columns) == ["text", "label", "x", "y"]
    assert len(result.frame) == len(texts)
    assert result.embeddings.shape[0] == len(texts)


def test_build_explorer_3d():
    texts = [f"sample document about topic {i}" for i in range(6)]
    result = build_explorer(texts, config=EmbeddingConfig(backend="tfidf-svd"), reducer="pca", dimensions=3)
    assert "z" in result.frame.columns

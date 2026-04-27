from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import EmbeddingConfig, build_explorer, make_plot


def load_input(path: str, text_column: str, label_column: str | None):
    df = pd.read_csv(path)
    if text_column not in df.columns:
        raise SystemExit(f"Text column '{text_column}' not found. Available columns: {', '.join(df.columns)}")
    labels = df[label_column].fillna("unlabeled").astype(str).tolist() if label_column and label_column in df.columns else None
    return df[text_column].fillna("").astype(str).tolist(), labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate interactive embedding visualizations from text CSV data.")
    parser.add_argument("--input", default="data/sample_texts.csv", help="CSV file containing text samples")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--embedding-backend", choices=["tfidf-svd", "sentence-transformers", "openai"], default="tfidf-svd")
    parser.add_argument("--sentence-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--openai-model", default="text-embedding-3-small")
    parser.add_argument("--reducer", choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--dimensions", type=int, choices=[2, 3], default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-html", default="out/explorer.html")
    parser.add_argument("--output-csv", default="out/coordinates.csv")
    args = parser.parse_args(argv)

    texts, labels = load_input(args.input, args.text_column, args.label_column)
    config = EmbeddingConfig(
        backend=args.embedding_backend,
        sentence_model=args.sentence_model,
        openai_model=args.openai_model,
        random_state=args.random_state,
    )
    result = build_explorer(texts, labels=labels, config=config, reducer=args.reducer, dimensions=args.dimensions)
    fig = make_plot(result)

    html_path = Path(args.output_html)
    csv_path = Path(args.output_csv)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path)
    result.frame.to_csv(csv_path, index=False)
    print(f"Wrote {html_path} and {csv_path} ({len(result.frame)} samples).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

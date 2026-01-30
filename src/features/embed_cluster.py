"""Embed works with TF-IDF and cluster with KMeans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.paths import artifacts_dir


DEFAULT_MAX_FEATURES = 5000
DEFAULT_RANDOM_STATE = 42


def build_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("")
    abstract = df.get("abstract", pd.Series([""] * len(df))).fillna("")
    return title + "\n" + abstract


def embed_text(text: pd.Series) -> "scipy.sparse.spmatrix":
    vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES, stop_words="english")
    return vectorizer.fit_transform(text)


def cluster_embeddings(embeddings: "scipy.sparse.spmatrix", k: int, random_state: int) -> pd.Series:
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = model.fit_predict(embeddings)
    return pd.Series(labels, name="cluster_id")


def save_cluster_meta(cluster_ids: pd.Series, output_path: Path) -> None:
    counts = cluster_ids.value_counts().sort_index()
    payload: Dict[str, int] = {str(k): int(v) for k, v in counts.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed and cluster works.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=artifacts_dir() / "works.parquet",
        help="Input parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=artifacts_dir() / "clustered.parquet",
        help="Output parquet path.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of clusters.")
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for clustering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input_path)
    text = build_text(df)
    embeddings = embed_text(text)
    cluster_ids = cluster_embeddings(embeddings, k=args.k, random_state=args.random_state)

    df = df.copy()
    df["cluster_id"] = cluster_ids.values

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)

    meta_path = artifacts_dir() / "cluster_meta.json"
    save_cluster_meta(cluster_ids, meta_path)


if __name__ == "__main__":
    main()
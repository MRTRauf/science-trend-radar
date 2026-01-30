"""Summarize clustered works with optional LLM support."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.llm.openai_client import chat_complete
from src.utils.paths import artifacts_dir


DEFAULT_N_SAMPLES = 12
DEFAULT_MAX_FEATURES = 2000


@dataclass
class ClusterSummary:
    cluster_id: int
    label: str
    summary: str
    keywords: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "summary": self.summary,
            "keywords": self.keywords,
        }


def build_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("")
    abstract = df.get("abstract", pd.Series([""] * len(df))).fillna("")
    return title + "\n" + abstract


def sample_cluster(df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    if len(df) <= n_samples:
        return df
    return df.sample(n=n_samples, random_state=0)


def fallback_summary(texts: Iterable[str]) -> ClusterSummary:
    vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES, stop_words="english")
    matrix = vectorizer.fit_transform(list(texts))
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    top_idx = scores.argsort()[::-1]
    top_terms = [terms[i] for i in top_idx[:3] if scores[i] > 0]
    keywords = top_terms if top_terms else ["general"]

    label = " / ".join(keywords[:2])
    summary = f"Cluster focused on {', '.join(keywords[:3])} based on TF-IDF terms."

    return ClusterSummary(cluster_id=-1, label=label, summary=summary, keywords=keywords[:3])


def build_prompt(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system = (
        "You are summarizing a research cluster. "
        "Use only the provided titles/abstracts. "
        "Do not add facts beyond the text. "
        "Respond as strict JSON with keys: label, summary, keywords."
    )

    lines = []
    for idx, record in enumerate(records, start=1):
        title = record.get("title") or ""
        abstract = record.get("abstract") or ""
        if len(abstract) > 600:
            abstract = abstract[:600] + "..."
        lines.append(f"{idx}. Title: {title}\n   Abstract: {abstract}")

    user = "\n".join(lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def llm_summary(records: List[Dict[str, str]]) -> ClusterSummary:
    messages = build_prompt(records)
    raw = chat_complete(messages, response_format={"type": "json_object"})
    data = json.loads(raw)

    label = data.get("label", "")
    summary = data.get("summary", "")
    keywords = data.get("keywords", [])

    if not isinstance(keywords, list):
        keywords = []

    return ClusterSummary(cluster_id=-1, label=label, summary=summary, keywords=keywords[:3])


def summarize_clusters(df: pd.DataFrame, n_samples: int) -> List[ClusterSummary]:
    summaries: List[ClusterSummary] = []
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    mode = "LLM" if use_llm else "fallback"
    print(f"Mode: {mode}")

    for cluster_id, group in df.groupby("cluster_id"):
        sampled = sample_cluster(group, n_samples)
        records = sampled[["title", "abstract"]].fillna("").to_dict(orient="records")

        if use_llm:
            try:
                summary = llm_summary(records)
            except (json.JSONDecodeError, KeyError):
                summary = fallback_summary(build_text(sampled))
        else:
            summary = fallback_summary(build_text(sampled))

        summary.cluster_id = int(cluster_id)
        summaries.append(summary)

    return summaries


def save_summaries(summaries: List[ClusterSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [summary.to_dict() for summary in summaries]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize clusters.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=artifacts_dir() / "clustered.parquet",
        help="Input clustered parquet path.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=artifacts_dir() / "cluster_summaries.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Docs sampled per cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input_path)
    summaries = summarize_clusters(df, n_samples=args.n_samples)
    save_summaries(summaries, args.output_path)


if __name__ == "__main__":
    main()

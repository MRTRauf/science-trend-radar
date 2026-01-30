"""Preprocessing utilities for OpenAlex works."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _normalize_whitespace(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(value.split())


def preprocess_works(df: pd.DataFrame) -> pd.DataFrame:
    """Clean works data for downstream use."""
    if df.empty:
        return df

    df = df.dropna(subset=["title"]).copy()

    for col in ["title", "abstract"]:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_whitespace)

    if "publication_year" in df.columns:
        df = df.drop_duplicates(subset=["title", "publication_year"])
    else:
        df = df.drop_duplicates(subset=["title"])

    return df
"""Streamlit dashboard for science-trend-radar."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import content_card, load_css, metric_card, render_hero, section_header
from src.utils.paths import artifacts_dir


CLUSTERED_PATH = artifacts_dir() / "clustered.parquet"
SUMMARIES_PATH = artifacts_dir() / "cluster_summaries.json"
PLOTLY_ACCENT = "#f97316"


def load_clustered(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Missing clustered data at {path}")
        st.stop()
    return pd.read_parquet(path)


def load_summaries(path: Path) -> Dict[int, Dict[str, object]]:
    if not path.exists():
        st.warning(f"Missing summaries at {path}. Cluster labels will be empty.")
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    summaries: Dict[int, Dict[str, object]] = {}
    for item in data:
        try:
            summaries[int(item.get("cluster_id"))] = item
        except (TypeError, ValueError):
            continue
    return summaries


def apply_filters(
    df: pd.DataFrame,
    year_range: List[int],
    min_cited_by: int,
    cluster_filter: List[int],
) -> pd.DataFrame:
    filtered = df.copy()

    if "publication_year" in filtered.columns:
        filtered = filtered[
            (filtered["publication_year"] >= year_range[0])
            & (filtered["publication_year"] <= year_range[1])
        ]

    if "cited_by_count" in filtered.columns and min_cited_by > 0:
        filtered = filtered[filtered["cited_by_count"].fillna(0) >= min_cited_by]

    if cluster_filter:
        filtered = filtered[filtered["cluster_id"].isin(cluster_filter)]

    return filtered


def apply_plotly_theme(fig: "px.Figure") -> "px.Figure":
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#0f172a"},
    )
    fig.update_traces(marker_color=PLOTLY_ACCENT)
    return fig


def overview_section(df: pd.DataFrame) -> None:
    section_header("Overview")

    if "publication_year" not in df.columns or df.empty:
        st.info("No publication year data available.")
        return

    by_year = df["publication_year"].value_counts().sort_index()
    chart_df = by_year.reset_index()
    chart_df.columns = ["year", "count"]

    fig = px.bar(chart_df, x="year", y="count", labels={"count": "Papers"})
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def clusters_section(
    df: pd.DataFrame,
    summaries: Dict[int, Dict[str, object]],
) -> None:
    section_header("Clusters")

    if df.empty:
        st.info("No rows available for clusters.")
        return

    clusters = sorted(df["cluster_id"].dropna().unique().tolist())
    selected = st.selectbox("Cluster", options=clusters, key="cluster_select", help="Select a cluster to review.")
    cluster_df = df[df["cluster_id"] == selected]

    summary = summaries.get(selected, {})
    label = summary.get("label", "(no label)")
    text = summary.get("summary", "(no summary)")
    keywords = summary.get("keywords", [])
    keyword_text = ", ".join(str(k) for k in keywords) if keywords else "(none)"

    content_card(
        "Cluster summary",
        f"<strong>{label}</strong><br />{text}<br /><em>Keywords:</em> {keyword_text}",
    )

    st.markdown(f"**Count:** {len(cluster_df)}")

    example_cols = ["title", "publication_year", "cited_by_count", "venue", "doi"]
    available_cols = [c for c in example_cols if c in cluster_df.columns]
    examples = cluster_df.sort_values("cited_by_count", ascending=False).head(12)

    if "doi" in examples.columns:
        def to_doi_url(value: object) -> str:
            if not value:
                return ""
            text_val = str(value)
            if text_val.startswith("http"):
                return text_val
            return f"https://doi.org/{text_val}"

        examples = examples.copy()
        examples["doi_url"] = examples["doi"].apply(to_doi_url)
        available_cols = [c for c in available_cols if c != "doi"] + ["doi_url"]

        column_config = None
        if hasattr(st, "column_config") and hasattr(st.column_config, "LinkColumn"):
            column_config = {
                "doi_url": st.column_config.LinkColumn("DOI", display_text="open"),
            }

        try:
            if column_config:
                st.dataframe(
                    examples[available_cols],
                    use_container_width=True,
                    column_config=column_config,
                )
            else:
                st.dataframe(examples[available_cols], use_container_width=True)
        except TypeError:
            st.dataframe(examples[available_cols], use_container_width=True)
    else:
        st.dataframe(examples[available_cols], use_container_width=True)


def search_section(df: pd.DataFrame) -> None:
    section_header("Search")

    query = st.text_input("Keyword search", help="Search titles and abstracts.")
    if not query:
        return

    haystack = (
        df["title"].fillna("")
        + " "
        + df.get("abstract", pd.Series([""] * len(df))).fillna("")
    )
    mask = haystack.str.contains(query, case=False, regex=False)
    results = df[mask]

    st.markdown(f"**Matches:** {len(results)}")
    cols = [c for c in ["title", "abstract", "publication_year", "cited_by_count", "cluster_id"] if c in results.columns]
    st.dataframe(results[cols], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Science Trend Radar", layout="wide")
    load_css()
    render_hero("Science Trend Radar", "Clustered research signals with summaries and search.")

    df = load_clustered(CLUSTERED_PATH)
    summaries = load_summaries(SUMMARIES_PATH)

    if df.empty:
        st.warning("Clustered dataset is empty.")
        return

    year_min = int(df["publication_year"].min()) if "publication_year" in df.columns else 1900
    year_max = int(df["publication_year"].max()) if "publication_year" in df.columns else 2100

    st.sidebar.header("Filters")
    cluster_options = sorted(df["cluster_id"].dropna().unique().tolist())
    default_year_range = (year_min, year_max)
    default_clusters = cluster_options

    if st.sidebar.button("Reset filters"):
        st.session_state["year_range"] = default_year_range
        st.session_state["min_cited_by"] = 0
        st.session_state["clusters"] = default_clusters

    st.sidebar.subheader("Year")
    year_range = st.sidebar.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=default_year_range,
        key="year_range",
        help="Limit results to a publication window.",
    )

    st.sidebar.subheader("Citations")
    min_cited_by = st.sidebar.number_input(
        "Min cited_by_count",
        min_value=0,
        value=0,
        step=1,
        key="min_cited_by",
        help="Filter by minimum citation count.",
    )

    st.sidebar.subheader("Cluster")
    selected_clusters = st.sidebar.multiselect(
        "Clusters",
        options=cluster_options,
        default=default_clusters,
        key="clusters",
        help="Limit results to specific clusters.",
    )

    filtered_df = apply_filters(df, list(year_range), int(min_cited_by), selected_clusters)

    total_papers = len(filtered_df)
    year_series = filtered_df["publication_year"].dropna() if "publication_year" in filtered_df.columns else pd.Series([])
    year_range_text = f"{int(year_series.min())}-{int(year_series.max())}" if not year_series.empty else "n/a"
    cluster_count = filtered_df["cluster_id"].nunique() if "cluster_id" in filtered_df.columns else 0
    citations = filtered_df["cited_by_count"].dropna() if "cited_by_count" in filtered_df.columns else pd.Series([])
    median_citations = f"{int(citations.median())}" if not citations.empty else "n/a"
    if "abstract" in filtered_df.columns and not filtered_df.empty:
        abstract_pct = (filtered_df["abstract"].fillna("").str.len() > 0).mean() * 100
        abstract_text = f"{abstract_pct:.0f}%"
    else:
        abstract_text = "n/a"

    kpi_cols = st.columns(5)
    with kpi_cols[0]:
        metric_card("Total papers", f"{total_papers}")
    with kpi_cols[1]:
        metric_card("Year range", year_range_text)
    with kpi_cols[2]:
        metric_card("Clusters", f"{cluster_count}")
    with kpi_cols[3]:
        metric_card("Median citations", median_citations)
    with kpi_cols[4]:
        metric_card("% with abstracts", abstract_text)

    with st.expander("How to use", expanded=False):
        st.markdown(
            "- Use filters to narrow the dataset.\n"
            "- Review clusters to see summaries and example papers.\n"
            "- Search for keywords across titles and abstracts."
        )

    overview_tab, clusters_tab, search_tab = st.tabs(["Overview", "Clusters", "Search"])
    with overview_tab:
        overview_section(filtered_df)
    with clusters_tab:
        clusters_section(filtered_df, summaries)
    with search_tab:
        search_section(filtered_df)


if __name__ == "__main__":
    main()

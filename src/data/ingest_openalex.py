"""Ingest works from the OpenAlex API."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

from src.data.preprocess import preprocess_works
from src.utils.paths import artifacts_dir


OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_PER_PAGE = 200
REQUEST_TIMEOUT = 30


def abstract_from_inverted_index(abstract_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
    if not abstract_index:
        return None
    positions: Dict[int, str] = {}
    for term, offsets in abstract_index.items():
        for offset in offsets:
            positions[offset] = term
    return " ".join(positions[idx] for idx in sorted(positions))


def build_filters(year_from: Optional[int], year_to: Optional[int]) -> str:
    filters: List[str] = []
    if year_from is not None:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to is not None:
        filters.append(f"to_publication_date:{year_to}-12-31")
    return ",".join(filters)


def fetch_works(
    query: str,
    year_from: Optional[int],
    year_to: Optional[int],
    limit: int,
    per_page: int = DEFAULT_PER_PAGE,
    polite_pause_s: float = 0.2,
) -> Iterable[Dict[str, Any]]:
    cursor = "*"
    fetched = 0
    session = requests.Session()
    session.headers.update({"User-Agent": "science-trend-radar/0.1 (mailto:contact@example.com)"})

    while cursor and fetched < limit:
        params = {
            "search": query,
            "per-page": min(per_page, limit - fetched),
            "cursor": cursor,
        }
        filters = build_filters(year_from, year_to)
        if filters:
            params["filter"] = filters

        response = session.get(OPENALEX_WORKS_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

        results = payload.get("results", [])
        for item in results:
            yield item
            fetched += 1
            if fetched >= limit:
                break

        cursor = payload.get("meta", {}).get("next_cursor")
        time.sleep(polite_pause_s)


def extract_record(item: Dict[str, Any]) -> Dict[str, Any]:
    primary_location = item.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name")

    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "publication_year": item.get("publication_year"),
        "abstract": abstract_from_inverted_index(item.get("abstract_inverted_index")),
        "cited_by_count": item.get("cited_by_count"),
        "venue": venue,
        "authorships": item.get("authorships"),
        "doi": item.get("doi"),
    }


def save_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows saved.")
        return

    years = df["publication_year"].dropna()
    year_min = int(years.min()) if not years.empty else None
    year_max = int(years.max()) if not years.empty else None
    venues = df["venue"].dropna().value_counts().head(5)

    print(f"Rows saved: {len(df)}")
    if year_min is not None and year_max is not None:
        print(f"Year range: {year_min}-{year_max}")
    else:
        print("Year range: (unknown)")
    print("Top venues:")
    if venues.empty:
        print("  (none)")
    else:
        for name, count in venues.items():
            print(f"  {name}: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest OpenAlex works.")
    parser.add_argument("--query", required=True, help="Search query string.")
    parser.add_argument("--year_from", type=int, help="Start publication year.")
    parser.add_argument("--year_to", type=int, help="End publication year.")
    parser.add_argument("--limit", type=int, required=True, help="Max number of works to fetch.")
    parser.add_argument(
        "--out",
        type=Path,
        default=artifacts_dir() / "works.parquet",
        help="Output parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = [extract_record(item) for item in fetch_works(
        query=args.query,
        year_from=args.year_from,
        year_to=args.year_to,
        limit=args.limit,
    )]

    df = pd.DataFrame.from_records(records)
    df = preprocess_works(df)
    save_parquet(df, args.out)
    print_summary(df)


if __name__ == "__main__":
    main()

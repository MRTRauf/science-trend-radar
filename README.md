# science-trend-radar

## Goal
Track and summarize emerging scientific trends from public sources.

## Quickstart
```bash
# 1) Create venv
python -m venv .venv

# 2) Activate venv (Windows)
.venv\Scripts\Activate.ps1

# 3) Install deps
pip install -r requirements.txt

# 4) Run tests
python -m pytest -q
```

## Repro Steps
```bash
# 1) Ingest
python -m src.data.ingest_openalex --query "graph neural networks" --year_from 2020 --year_to 2024 --limit 500

# 2) Cluster
python -m src.features.embed_cluster --k 10

# 3) Summarize
python -m src.llm.summarize_clusters --n_samples 12

# 4) Dashboard
streamlit run app/dashboard.py
```

## Outputs (artifacts)
- `artifacts/works.parquet`
- `artifacts/clustered.parquet`
- `artifacts/cluster_meta.json`
- `artifacts/cluster_summaries.json`

## Structure
```
app/
artifacts/
src/
  data/
  features/
  llm/
  utils/
    paths.py
tests/
```

## Theme
The Streamlit theme is configured in `.streamlit/config.toml`.

## LLM (optional)
- If `OPENAI_API_KEY` is set, summaries use the LLM.
- Otherwise summaries fall back to TF-IDF terms.
- You can store the key in `.env` (do not commit it).

## Responsible AI
- Summaries are derived only from the titles/abstracts shown in each cluster.
- Clusters and summaries can reflect source coverage, field bias, and missing data.
- Treat results as exploratory signals, not definitive conclusions.

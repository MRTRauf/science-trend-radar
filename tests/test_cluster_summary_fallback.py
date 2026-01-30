import json
from pathlib import Path

import pandas as pd

from src.llm.summarize_clusters import summarize_clusters


def test_cluster_summary_fallback(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "cluster_id": [0, 0, 1],
        "title": ["Graph networks", "Neural graphs", "Quantum materials"],
        "abstract": ["Graph neural networks", "Graphs for learning", "Quantum states"],
    })

    summaries = summarize_clusters(df, n_samples=2)

    payload = [summary.to_dict() for summary in summaries]
    out_path = tmp_path / "summaries.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data
    for item in data:
        assert "cluster_id" in item
        assert "label" in item
        assert "summary" in item
        assert "keywords" in item
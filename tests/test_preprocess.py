import pandas as pd

from src.data.preprocess import preprocess_works


def test_preprocess_drops_missing_titles() -> None:
    df = pd.DataFrame({
        "title": ["Good title", None, ""],
        "publication_year": [2020, 2021, 2022],
    })

    out = preprocess_works(df)

    assert out["title"].isna().sum() == 0
    assert len(out) == 2


def test_preprocess_normalizes_whitespace() -> None:
    df = pd.DataFrame({
        "title": ["  A   title  ", "Another\nTitle"],
        "abstract": ["  Some   text\n here ", None],
        "publication_year": [2020, 2021],
    })

    out = preprocess_works(df)

    assert out.loc[out.index[0], "title"] == "A title"
    assert out.loc[out.index[0], "abstract"] == "Some text here"
    assert out.loc[out.index[1], "title"] == "Another Title"


def test_preprocess_deduplicates() -> None:
    df = pd.DataFrame({
        "title": ["Same title", "Same title", "Same title"],
        "publication_year": [2020, 2020, 2021],
        "abstract": ["a", "b", "c"],
    })

    out = preprocess_works(df)

    assert len(out) == 2
    assert set(out["publication_year"].tolist()) == {2020, 2021}
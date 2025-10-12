import pandas as pd
import polars as pl

# noqa: F401


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Outcome": ["Yes", "No", "Yes", "No"],
            "Segment": ["A", "A", "B", "B"],
        }
    )


def test_binarize_polars_accessor():
    sample = _sample_dataframe()
    pl_df = pl.from_pandas(sample)

    result = pl_df.tk.binarize(
        n_bins=2,
        thresh_infreq=0.0,
        name_infreq="-OTHER",
        one_hot=True,
    )

    assert isinstance(result, pl.DataFrame)
    result_columns = set(result.columns)

    assert "Outcome__Yes" in result_columns
    assert "Segment__A" in result_columns


def test_correlate_polars_accessor():
    sample = _sample_dataframe()
    pl_df = pl.from_pandas(sample)

    binarized = pl_df.tk.binarize(
        n_bins=2,
        thresh_infreq=0.0,
        name_infreq="-OTHER",
        one_hot=True,
    )

    result = binarized.tk.correlate(target="Outcome__Yes")

    assert isinstance(result, pl.DataFrame)

    result_pd = result.to_pandas()
    assert result_pd.columns.tolist() == ["feature", "bin", "correlation"]

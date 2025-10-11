import polars as pl
import pytimetk as tk
import pytest


@pytest.fixture(scope="module")
def sample_df():
    df = tk.load_dataset("stocks_daily", parse_dates=["date"])
    return df[df["symbol"].isin(["AAPL", "MSFT"])].copy()


def test_polars_namespace_dataframe(sample_df):
    pl_df = pl.from_pandas(sample_df)

    result = pl_df.tk.augment_cmo(
        date_column="date",
        close_column="close",
        periods=14,
    )
    assert isinstance(result, pl.DataFrame)
    assert "close_cmo_14" in result.columns

    # Chaining a second augmenter should remain within Polars.
    chained = (
        result.tk.augment_roc(
            date_column="date",
            close_column="close",
            periods=5,
        )
    )
    assert isinstance(chained, pl.DataFrame)
    roc_columns = [c for c in chained.columns if c.startswith("close_roc_") and c.endswith("_5")]
    assert roc_columns, "Expected a close ROC column ending with '_5'"


def test_polars_namespace_groupby(sample_df):
    pl_df = pl.from_pandas(sample_df)
    grouped = pl_df.group_by("symbol", maintain_order=True)

    result = grouped.tk.augment_cmo(
        date_column="date",
        close_column="close",
        periods=14,
    )
    assert isinstance(result, pl.DataFrame)
    assert "close_cmo_14" in result.columns
    # Ensure grouping columns are preserved after augmentation.
    assert "symbol" in result.columns

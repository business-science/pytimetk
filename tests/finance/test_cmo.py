import pytest
import pandas as pd
import pytimetk as tk
import os
import multiprocess as mp
from itertools import product

# Setup to avoid multiprocessing warnings
mp.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@pytest.fixture(scope="module")
def df():
    return tk.load_dataset("stocks_daily", parse_dates=["date"])


@pytest.mark.parametrize(
    "engine,periods", product(["pandas", "polars"], [[14], [14, 28]])
)
def test_cmo(df, engine, periods):
    """Test CMO with grouped and ungrouped data, different engines, and periods."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_cmo(
        date_column="date", close_column="close", periods=periods, engine=engine
    )
    expected_cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted",
    ]
    expected_cols += [f"close_cmo_{p}" for p in periods]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    assert (result_grouped[expected_cols[8:]].dropna() >= -100).all().all(), (
        "CMO values below -100"
    )
    assert (result_grouped[expected_cols[8:]].dropna() <= 100).all().all(), (
        "CMO values above 100"
    )
    # Verify NaNs (p-1 NaNs per group due to implementation)
    for p in periods:
        nan_counts = result_grouped.groupby("symbol")[f"close_cmo_{p}"].apply(
            lambda x: x.isna().sum()
        )
        print(f"NaN counts for period {p}:", nan_counts.to_dict())
        assert all(nan_counts == p - 1), (
            f"Expected {p - 1} NaNs per group for period {p}"
        )
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_cmo(
        date_column="date", close_column="close", periods=periods, engine=engine
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    for p in periods:
        assert result_single[f"close_cmo_{p}"].isna().sum() == p - 1, (
            f"Expected {p - 1} NaNs for period {p}"
        )
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def test_cmo_edge_cases(df):
    """Test CMO with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_cmo(
        date_column="date", close_column="close", periods=[14], engine="pandas"
    )
    assert result_small.shape[0] == 10
    assert result_small["close_cmo_14"].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(
        ValueError, match="`value_column` \\(close\\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_cmo(
            date_column="date", close_column="close", periods=[14], engine="pandas"
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_cmo(
            date_column="date", close_column="close", periods=[14], engine="pandas"
        )

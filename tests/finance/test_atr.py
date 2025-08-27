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
    "engine,normalize,periods",
    product(["pandas", "polars"], [True, False], [[14], [14, 28]]),
)
def test_atr(df, engine, normalize, periods):
    """Test ATR with grouped and ungrouped data, different engines, normalization, and periods."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_atr(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=periods,
        normalize=normalize,
        engine=engine,
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
    expected_cols += [f"close_{'natr' if normalize else 'atr'}_{p}" for p in periods]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    assert (result_grouped[expected_cols[8:]].dropna() >= 0).all().all(), (
        "ATR/NATR values below 0"
    )
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_atr(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=periods,
        normalize=normalize,
        engine=engine,
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def test_atr_edge_cases(df):
    """Test ATR with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_atr(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=[14],
        normalize=True,
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    # Note: augment_atr produces valid values even for small datasets (no NaNs observed)
    assert result_small["close_natr_14"].isna().sum() == 0, (
        "Unexpected NaNs in small dataset"
    )

    # Missing columns
    with pytest.raises(ValueError, match=".*`value_column`.*not found.*"):
        df[["symbol", "date"]].augment_atr(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14],
            engine="pandas",
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "high", "low", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_atr(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14],
            engine="pandas",
        )

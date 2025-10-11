import pytest
import pandas as pd
import polars as pl
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


@pytest.fixture(scope="module")
def pl_df(df):
    return pl.from_pandas(df)


def _to_pandas(df):
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


@pytest.mark.parametrize(
    "engine,periods,std_dev",
    product(["pandas", "polars"], [[20], [20, 40]], [2, [1.5, 2]]),
)
def test_bbands(df, engine, periods, std_dev):
    """Test BBANDS with grouped and ungrouped data, different engines, periods, and std_dev."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_bbands(
        date_column="date",
        close_column="close",
        periods=periods,
        std_dev=std_dev,
        engine=engine,
    )
    result_grouped = _to_pandas(result_grouped)
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
    periods_list = periods if isinstance(periods, list) else [periods]
    std_dev_list = std_dev if isinstance(std_dev, list) else [std_dev]
    for p in periods_list:
        for s in std_dev_list:
            expected_cols += [
                f"close_bband_middle_{p}_{s:.1f}",
                f"close_bband_upper_{p}_{s:.1f}",
                f"close_bband_lower_{p}_{s:.1f}",
            ]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "AAPL"').augment_bbands(
        date_column="date",
        close_column="close",
        periods=periods,
        std_dev=std_dev,
        engine=engine,
    )
    result_single = _to_pandas(result_single)
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def test_bbands_edge_cases(df):
    """Test BBANDS with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "AAPL"').head(10)
    result_small = small_df.augment_bbands(
        date_column="date",
        close_column="close",
        periods=[20],
        std_dev=2,
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    assert result_small["close_bband_lower_20_2.0"].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(ValueError, match=".*`value_column`.*not found.*"):
        df[["symbol", "date"]].augment_bbands(
            date_column="date",
            close_column="close",
            periods=[20],
            std_dev=2,
            engine="pandas",
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_bbands(
            date_column="date",
            close_column="close",
            periods=[20],
            std_dev=2,
            engine="pandas",
        )


def test_bbands_polars_dataframe_roundtrip(pl_df):
    pandas_single = (
        tk.load_dataset("stocks_daily", parse_dates=["date"])
        .query("symbol == 'AAPL'")
    )

    pandas_result = pandas_single.augment_bbands(
        date_column="date",
        close_column="close",
        periods=[20],
        std_dev=[1.5, 2],
    )

    polars_result = tk.augment_bbands(
        data=pl_df.filter(pl.col("symbol") == "AAPL"),
        date_column="date",
        close_column="close",
        periods=[20],
        std_dev=[1.5, 2],
    )

    pd.testing.assert_frame_equal(
        pandas_result.reset_index(drop=True),
        polars_result.to_pandas().reset_index(drop=True),
    )


def test_bbands_polars_groupby_roundtrip(pl_df):
    pandas_group = (
        tk.load_dataset("stocks_daily", parse_dates=["date"])
        .groupby("symbol")
        .augment_bbands(
            date_column="date",
            close_column="close",
            periods=[20],
            std_dev=2,
        )
    )

    polars_group = tk.augment_bbands(
        data=pl_df.group_by("symbol"),
        date_column="date",
        close_column="close",
        periods=[20],
        std_dev=2,
    )

    pd.testing.assert_frame_equal(
        pandas_group.reset_index(drop=True),
        polars_group.to_pandas().reset_index(drop=True),
    )

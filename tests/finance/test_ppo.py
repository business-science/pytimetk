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
    "engine,fast_period,slow_period", product(["pandas", "polars"], [12], [26])
)
def test_ppo(df, engine, fast_period, slow_period):
    """Test PPO with grouped and ungrouped data, different engines."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_ppo(
        date_column="date",
        close_column="close",
        fast_period=fast_period,
        slow_period=slow_period,
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
        f"close_ppo_line_{fast_period}_{slow_period}",
    ]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    # Verify NaNs (no NaNs observed due to implementation)
    col = f"close_ppo_line_{fast_period}_{slow_period}"
    nan_counts = result_grouped.groupby("symbol")[col].apply(lambda x: x.isna().sum())
    print(f"NaN counts for {col}:", nan_counts.to_dict())
    assert all(nan_counts == 0), f"Expected 0 NaNs per group for {col}"
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_ppo(
        date_column="date",
        close_column="close",
        fast_period=fast_period,
        slow_period=slow_period,
        engine=engine,
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    assert result_single[col].isna().sum() == 0, f"Expected 0 NaNs for {col}"
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def test_ppo_edge_cases(df):
    """Test PPO with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_ppo(
        date_column="date",
        close_column="close",
        fast_period=12,
        slow_period=26,
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    # Note: augment_ppo produces valid values even for small datasets (no NaNs observed)
    assert result_small["close_ppo_line_12_26"].isna().sum() == 0, (
        "Unexpected NaNs in small dataset"
    )

    # Invalid periods
    with pytest.raises(ValueError, match="span must satisfy: span >= 1"):
        df.augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=0,
            slow_period=26,
            engine="pandas",
        )
    with pytest.raises(ValueError, match="span must satisfy: span >= 1"):
        df.augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=0,
            engine="pandas",
        )

    # Missing columns
    with pytest.raises(
        ValueError, match="`value_column` \\(close\\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=26,
            engine="pandas",
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=26,
            engine="pandas",
        )

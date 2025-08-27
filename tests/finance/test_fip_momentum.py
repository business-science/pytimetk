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
    "engine,window,skip_window", product(["pandas", "polars"], [[63], [63, 252]], [21])
)
def test_fip_momentum(df, engine, window, skip_window):
    """Test FIP Momentum with grouped and ungrouped data, different engines, windows, and skip_window."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=window,
        skip_window=skip_window,
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
    expected_cols += [f"close_fip_momentum_{w}" for w in window]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    # Verify NaNs (engine-specific expectations due to implementation differences)
    for w in window:
        nan_counts = result_grouped.groupby("symbol")[f"close_fip_momentum_{w}"].apply(
            lambda x: x.isna().sum()
        )
        print(f"NaN counts for window {w} ({engine}):", nan_counts.to_dict())
        if engine == "pandas":
            assert all(nan_counts >= w - 1), (
                f"Expected at least {w - 1} NaNs per group for window {w} (pandas)"
            )
        else:  # polars
            assert all(nan_counts >= w // 2 - 1), (
                f"Expected at least {w // 2 - 1} NaNs per group for window {w} (polars)"
            )
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=window,
        skip_window=skip_window,
        engine=engine,
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    for w in window:
        if engine == "pandas":
            assert result_single[f"close_fip_momentum_{w}"].isna().sum() >= w - 1, (
                f"Expected at least {w - 1} NaNs for window {w} (pandas)"
            )
        else:  # polars
            assert (
                result_single[f"close_fip_momentum_{w}"].isna().sum() >= w // 2 - 1
            ), f"Expected at least {w // 2 - 1} NaNs for window {w} (polars)"
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def test_fip_momentum_edge_cases(df):
    """Test FIP Momentum with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=[63],
        skip_window=21,
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    assert result_small["close_fip_momentum_63"].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(
        ValueError, match="`value_column` \\(close\\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[63],
            skip_window=21,
            engine="pandas",
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[63],
            skip_window=21,
            engine="pandas",
        )

    # Invalid window
    with pytest.raises(ValueError, match="All window values must be positive integers"):
        df.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[0],
            skip_window=21,
            engine="pandas",
        )

    # Invalid skip_window (note: no validation in function, so no error raised; consider adding to pytimetk)
    # Run the code and check for reasonable output or add assertion if negative skip_window causes issues
    result_negative_skip = df.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=[63],
        skip_window=-1,
        engine="pandas",
    )
    assert result_negative_skip.shape[0] == 16194, (
        "Expected full shape for negative skip_window"
    )
    assert result_negative_skip["close_fip_momentum_63"].isna().sum() > 0, (
        "Expected NaNs with negative skip_window"
    )

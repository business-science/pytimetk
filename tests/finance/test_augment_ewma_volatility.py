# tests/finance/test_augment_ewma_volatility.py

import pytest
import pandas as pd
import pytimetk as tk
import os
import multiprocess as mp

# Setup to avoid multiprocessing/threading warnings
mp.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@pytest.fixture(scope="module")
def df():
    return tk.load_dataset("stocks_daily", parse_dates=["date"])


@pytest.mark.parametrize(
    "engine,windows",
    [
        ("pandas", [20, 50]),
        ("polars", [20, 50]),
        ("pandas", [20]),
        ("polars", [50]),
    ],
)
def test_ewma_volatility(df, engine, windows):
    """
    Test augment_ewma_volatility with grouped/ungrouped data, multiple engines, and windows.
    Verifies:
      - expected EWMA vol columns exist with correct naming
      - values are non-negative
    """
    decay_factor = 0.94
    decay_str = str(decay_factor)
    window_list = windows if isinstance(windows, list) else [windows]

    # --- Grouped case ---
    result_grouped = df.groupby("symbol").augment_ewma_volatility(
        date_column="date",
        close_column="close",
        decay_factor=decay_factor,
        window=windows,
        engine=engine,
    )

    # Expected volatility columns (match actual function naming)
    expected_vol_cols = [f"close_ewma_vol_{w}_{decay_str}" for w in window_list]

    # Check that all expected vol columns were added (subset check; allow extra cols)
    for col in expected_vol_cols:
        assert col in result_grouped.columns, f"Missing expected column: {col}"
        # Non-negative values
        assert (result_grouped[col].dropna() >= 0).all(), (
            f"Negative volatility in {col}"
        )

    # --- Ungrouped case (single symbol) ---
    result_single = df.query("symbol == 'AAPL'").augment_ewma_volatility(
        date_column="date",
        close_column="close",
        decay_factor=decay_factor,
        window=windows,
        engine=engine,
    )
    for col in expected_vol_cols:
        assert col in result_single.columns, (
            f"Missing expected column (ungrouped): {col}"
        )
        assert (result_single[col].dropna() >= 0).all(), (
            f"Negative volatility in {col} (ungrouped)"
        )


def test_ewma_volatility_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - very small dataset (window > length) -> expect NaNs
      - missing columns -> ValueError
      - empty DataFrame with non-numeric dtype -> TypeError (pytimetk’s validators)
      - invalid window values (non-integer) -> ValueError/TypeError
      - invalid decay_factor (outside [0,1]) -> ValueError
    """
    decay_factor = 0.94
    decay_str = str(decay_factor)

    # Small dataset (insufficient rows => NaNs)
    small_df = df.query("symbol == 'AAPL'").head(5)
    result_small = small_df.augment_ewma_volatility(
        date_column="date", close_column="close", decay_factor=decay_factor, window=[20]
    )
    small_col = f"close_ewma_vol_20_{decay_str}"
    assert small_col in result_small.columns, f"Missing expected column: {small_col}"
    # With window=20 and only 5 rows, expect NaNs
    assert result_small[small_col].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(ValueError, match=r"value_column.*close.*not.*found.*data"):
        df[["symbol", "date"]].augment_ewma_volatility(
            date_column="date",
            close_column="close",
            decay_factor=decay_factor,
            window=[20],
        )

    # Empty DataFrame -> non-numeric dtype for value column
    with pytest.raises(TypeError, match=r"value_column.*close.*not.*numeric"):
        empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
        empty_df.augment_ewma_volatility(
            date_column="date",
            close_column="close",
            decay_factor=decay_factor,
            window=[20],
        )

    # Invalid window (non-integer item) -> should raise (robust across implementations)
    with pytest.raises((ValueError, TypeError), match=r"window|int|integer|numeric"):
        df.augment_ewma_volatility(
            date_column="date",
            close_column="close",
            decay_factor=decay_factor,
            window=["bad"],
        )

    # Invalid decay_factor (outside [0, 1])
    with pytest.raises(ValueError, match=r"decay_factor.*0.*1"):
        df.augment_ewma_volatility(
            date_column="date", close_column="close", decay_factor=1.5, window=[20]
        )

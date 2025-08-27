# tests/finance/test_augment_hurst_exponent.py

import re
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


def _resolve_hurst_colnames(columns, value_col: str, windows):
    """
    Try to resolve actual output column names for Hurst exponent across
    possible naming conventions:
      - {value_col}_hurst_exponent_{w}
      - {value_col}_hurst_{w}
      - {value_col}_hurstexp_{w}
    Returns a dict: {window: column_name}
    Raises AssertionError if a window's column cannot be found.
    """
    win_list = windows if isinstance(windows, list) else [windows]
    resolved = {}

    for w in win_list:
        candidates = [
            f"{value_col}_hurst_exponent_{w}",
            f"{value_col}_hurst_{w}",
            f"{value_col}_hurstexp_{w}",
        ]
        # Also allow hyphen/underscore variants or minor changes via regex
        regex_candidates = [
            re.compile(rf"^{re.escape(value_col)}_hurst[_\-]?exponent_{w}$"),
            re.compile(rf"^{re.escape(value_col)}_hurst_{w}$"),
            re.compile(rf"^{re.escape(value_col)}_hurstexp_{w}$"),
        ]

        found = None
        # Exact match first
        for c in candidates:
            if c in columns:
                found = c
                break
        # Regex fallback
        if not found:
            for rc in regex_candidates:
                match = [c for c in columns if rc.match(c)]
                if match:
                    found = match[0]
                    break

        assert found is not None, (
            f"Could not find expected Hurst column for window {w}. Checked candidates: {candidates}"
        )

        resolved[w] = found

    return resolved


@pytest.mark.parametrize(
    "engine,windows",
    [
        ("pandas", [100, 200]),
        ("polars", [100, 200]),
        ("pandas", [100]),
        ("polars", [200]),
    ],
)
def test_hurst_exponent(df, engine, windows):
    """
    Test augment_hurst_exponent with grouped/ungrouped data, multiple engines, and windows.
    Verifies:
      - Hurst columns exist (robust to naming)
      - Hurst values are in [0, 1] when present (common expectation)
      - Works with groupby and ungrouped cases
    """
    value_col = "close"

    # --- Grouped case ---
    res_grouped = df.groupby("symbol").augment_hurst_exponent(
        date_column="date",
        close_column=value_col,
        window=windows,
        engine=engine,
    )

    hurst_cols = _resolve_hurst_colnames(res_grouped.columns, value_col, windows)

    # Check that columns exist and are within [0, 1] when not NaN
    for w, col in hurst_cols.items():
        assert col in res_grouped.columns, f"Missing Hurst column: {col}"
        non_na = res_grouped[col].dropna()
        # Allow empty (if early NaNs), otherwise enforce range
        if len(non_na) > 0:
            assert (non_na >= 0).all() and (non_na <= 1).all(), (
                f"Values out of [0,1] in {col}"
            )

    # --- Ungrouped case (single symbol) ---
    res_single = df.query("symbol == 'AAPL'").augment_hurst_exponent(
        date_column="date",
        close_column=value_col,
        window=windows,
        engine=engine,
    )

    hurst_cols_single = _resolve_hurst_colnames(res_single.columns, value_col, windows)
    for w, col in hurst_cols_single.items():
        assert col in res_single.columns, f"Missing Hurst column (ungrouped): {col}"
        non_na = res_single[col].dropna()
        if len(non_na) > 0:
            assert (non_na >= 0).all() and (non_na <= 1).all(), (
                f"Values out of [0,1] in {col} (ungrouped)"
            )


def test_hurst_exponent_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - very small dataset (window > length) -> expect NaNs
      - missing columns -> ValueError
      - empty DataFrame with non-numeric dtype -> TypeError (pytimetk’s validators)
      - invalid window values (non-integer) -> ValueError/TypeError
    """
    value_col = "close"
    small_df = df.query("symbol == 'AAPL'").head(50)

    # Small dataset with large window -> NaNs expected
    res_small = small_df.augment_hurst_exponent(
        date_column="date",
        close_column=value_col,
        window=[100],
    )
    hurst_cols = _resolve_hurst_colnames(res_small.columns, value_col, [100])
    col_100 = hurst_cols[100]
    assert res_small[col_100].isna().sum() > 0, (
        "Expected NaNs for insufficient data (Hurst, window=100 on 50 rows)."
    )

    # Missing columns
    with pytest.raises(ValueError, match=r"value_column.*close.*not.*found.*data"):
        df[["symbol", "date"]].augment_hurst_exponent(
            date_column="date", close_column=value_col, window=[100]
        )

    # Empty DataFrame -> non-numeric dtype for value column
    with pytest.raises(TypeError, match=r"value_column.*close.*not.*numeric"):
        empty_df = pd.DataFrame(columns=["symbol", "date", value_col])
        empty_df.augment_hurst_exponent(
            date_column="date", close_column=value_col, window=[100]
        )

    # Invalid window (non-integer item) -> should raise (robust across implementations)
    with pytest.raises((ValueError, TypeError), match=r"window|int|integer|numeric"):
        df.augment_hurst_exponent(
            date_column="date", close_column=value_col, window=["bad"]
        )

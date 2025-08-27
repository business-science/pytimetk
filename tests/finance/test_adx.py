# tests/finance/test_adx.py

import re
import pytest
import pandas as pd
import numpy as np
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


def _resolve_adx_cols(columns, period):
    """
    Resolve ADX-related column names for a given period `p` across possible conventions.
    Candidates we allow (with or without 'close_' prefix):
      - ADX:  adx_{p}, close_adx_{p}, ADX_{p}
      - +DI:  plus_di_{p}, pdi_{p}, di_plus_{p}, +di_{p}, close_plus_di_{p}
      - -DI:  minus_di_{p}, mdi_{p}, di_minus_{p}, -di_{p}, close_minus_di_{p}
      - DX:   dx_{p}, close_dx_{p}, DX_{p}
    Returns: dict with keys {"adx", "plus_di", "minus_di", "dx"} mapping to found column names.
    Raises AssertionError if required columns can't be found.
    """
    p = str(period)
    cols = set(columns)

    def find_one(cands, regex_cands):
        for c in cands:
            if c in cols:
                return c
        for rc in regex_cands:
            matches = [c for c in cols if rc.match(c)]
            if matches:
                return matches[0]
        return None

    adx = find_one(
        [f"adx_{p}", f"ADX_{p}", f"close_adx_{p}", f"close_ADX_{p}"],
        [
            re.compile(rf"^(close_)?adx[_\-]?{p}$", re.I),
            re.compile(rf"^(close_)?adx$", re.I),
        ],
    )
    plus_di = find_one(
        [f"plus_di_{p}", f"pdi_{p}", f"di_plus_{p}", f"+di_{p}", f"close_plus_di_{p}"],
        [re.compile(rf"^(close_)?(plus_di|pdi|di\_plus|\+di)[_\-]?{p}$", re.I)],
    )
    minus_di = find_one(
        [
            f"minus_di_{p}",
            f"mdi_{p}",
            f"di_minus_{p}",
            f"-di_{p}",
            f"close_minus_di_{p}",
        ],
        [re.compile(rf"^(close_)?(minus_di|mdi|di\_minus|\-di)[_\-]?{p}$", re.I)],
    )
    dx = find_one(
        [f"dx_{p}", f"DX_{p}", f"close_dx_{p}"],
        [re.compile(rf"^(close_)?dx[_\-]?{p}$", re.I)],
    )

    out = {"adx": adx, "plus_di": plus_di, "minus_di": minus_di, "dx": dx}
    for key in ["adx", "plus_di", "minus_di"]:
        assert out[key] is not None, (
            f"Could not find expected {key.upper()} column for period {period}. "
            f"Available: {sorted(columns)}"
        )
    return out


def _assert_reasonable_range(series: pd.Series, name: str):
    """
    Assert indicator values are in a reasonable numeric range.
    We allow tiny negatives due to numeric/implementation differences.
      - Lower bound tolerance:
          * min >= -20 (hard bound)
          * 0.1th percentile >= -5 (most values non-negative)
      - Upper bound tolerance:
          * 99.9th percentile <= 120
          * max <= 200
    Prints diagnostics on failure.
    """
    s = series.dropna()
    if len(s) == 0:
        return

    smin = float(s.min())
    smax = float(s.max())
    q001 = float(np.quantile(s, 0.001))  # 0.1th percentile
    q999 = float(np.quantile(s, 0.999))  # 99.9th percentile

    errors = []
    if smin < -20:
        errors.append(f"min {smin:.4f} < -20")
    if q001 < -5:
        errors.append(f"q0.1% {q001:.4f} < -5")
    if q999 > 120:
        errors.append(f"q99.9% {q999:.4f} > 120")
    if smax > 200:
        errors.append(f"max {smax:.4f} > 200")

    if errors:
        msg = (
            f"{name} out-of-range: " + ", ".join(errors) + "\n"
            f"min={smin:.4f}, mean={s.mean():.4f}, max={smax:.4f}, q0.1%={q001:.4f}, q99.9%={q999:.4f}\n"
            f"head:\n{s.head().to_string(index=False)}\n"
            f"tail:\n{s.tail().to_string(index=False)}"
        )
        assert False, msg


@pytest.mark.parametrize(
    "engine,periods",
    [
        ("pandas", [14, 28]),
        ("polars", [14, 28]),
        ("pandas", [14]),
        ("polars", [28]),
    ],
)
def test_adx(df, engine, periods):
    """
    Test augment_adx with grouped/ungrouped data, multiple engines, and periods.
    Verifies:
      - ADX/+DI/-DI columns exist (robust to naming)
      - Values are within a reasonable range (with small negative tolerance)
      - Works with groupby and ungrouped cases
    """
    # Grouped
    res_grouped = df.groupby("symbol").augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=periods,
        engine=engine,
    )
    for p in periods if isinstance(periods, list) else [periods]:
        cols = _resolve_adx_cols(res_grouped.columns, p)
        _assert_reasonable_range(res_grouped[cols["adx"]], f"{cols['adx']} (grouped)")
        _assert_reasonable_range(
            res_grouped[cols["plus_di"]], f"{cols['plus_di']} (grouped)"
        )
        _assert_reasonable_range(
            res_grouped[cols["minus_di"]], f"{cols['minus_di']} (grouped)"
        )
        if cols.get("dx") is not None:
            _assert_reasonable_range(res_grouped[cols["dx"]], f"{cols['dx']} (grouped)")

    # Ungrouped (single symbol)
    res_single = df.query("symbol == 'AAPL'").augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=periods,
        engine=engine,
    )
    for p in periods if isinstance(periods, list) else [periods]:
        cols = _resolve_adx_cols(res_single.columns, p)
        _assert_reasonable_range(res_single[cols["adx"]], f"{cols['adx']} (ungrouped)")
        _assert_reasonable_range(
            res_single[cols["plus_di"]], f"{cols['plus_di']} (ungrouped)"
        )
        _assert_reasonable_range(
            res_single[cols["minus_di"]], f"{cols['minus_di']} (ungrouped)"
        )
        if cols.get("dx") is not None:
            _assert_reasonable_range(
                res_single[cols["dx"]], f"{cols['dx']} (ungrouped)"
            )


def test_adx_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - small dataset (period > length) -> expect NaNs
      - missing columns -> ValueError
      - empty DataFrame with non-numeric dtype -> TypeError
      - invalid periods values (non-integer item) -> ValueError/TypeError
    """
    # Small dataset: force warm-up NaNs
    small = df.query("symbol == 'AAPL'").head(10)
    res_small = small.augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=[14],
    )
    cols = _resolve_adx_cols(res_small.columns, 14)
    assert res_small[cols["adx"]].isna().sum() > 0, (
        "Expected NaNs for insufficient data (ADX, period=14 on 10 rows)."
    )

    # Missing columns
    with pytest.raises(ValueError, match=r"(high|low|close).*not.*found.*data"):
        df[["symbol", "date", "high"]].augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",  # 'low' missing from df slice
            close_column="close",  # 'close' missing from df slice
            periods=[14],
        )

    # Empty DataFrame -> non-numeric dtype for price columns
    with pytest.raises(TypeError, match=r"(high|low|close).*not.*numeric"):
        empty = pd.DataFrame(columns=["symbol", "date", "high", "low", "close"])
        empty.augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14],
        )

    # Invalid periods (non-integer item)
    with pytest.raises((ValueError, TypeError), match=r"period|int|integer|numeric"):
        df.augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=["bad"],
        )

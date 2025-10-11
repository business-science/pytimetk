# tests/finance/test_stochastic_oscillator.py

import re
import pytest
import pandas as pd
import polars as pl
import numpy as np
import pytimetk as tk
import os
import multiprocess as mp

# Avoid multiprocessing/threading warnings / over-subscription
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


def _to_pandas(frame):
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


# ---------- Helpers ----------


def _find_one(columns, candidates, regex_candidates):
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    for rc in regex_candidates:
        hits = [c for c in cols if rc.match(c)]
        if hits:
            return hits[0]
    return None


def _resolve_stoch_k_col(columns, value_prefix: str, k_period: int):
    """
    Resolve %K column name for a given k_period across common conventions:
      close_stoch_k_14, close_stochastic_k_14, stoch_k_14, stochastic_k_14, k_14, %K_14
    """
    k = str(k_period)
    candidates = [
        f"{value_prefix}stoch_k_{k}",
        f"{value_prefix}stochastic_k_{k}",
        f"stoch_k_{k}",
        f"stochastic_k_{k}",
        f"{value_prefix}k_{k}",
        f"k_{k}",
        f"{value_prefix}K_{k}",
        f"K_{k}",
    ]
    regex_candidates = [
        re.compile(
            rf"^{re.escape(value_prefix)}?(stoch|stochastic)?[_\-]?[kK][_\-]?{k}$", re.I
        ),
        re.compile(rf"^(stoch|stochastic)?[_\-]?[kK][_\-]?{k}$", re.I),
    ]
    col = _find_one(columns, candidates, regex_candidates)
    assert col is not None, (
        f"Could not find stochastic %K column for k={k_period}. Searched variants among: {sorted(columns)}"
    )
    return col


def _resolve_stoch_d_col(columns, value_prefix: str, k_period: int, d_period: int):
    """
    Resolve %D column name for (k_period, d_period) across common conventions:
      close_stoch_d_14_3, close_stochastic_d_14_3, stoch_d_14_3, D_14_3, %D_14_3
    Fallback: sometimes only d_{d} (no k) exists.
    """
    k, d = str(k_period), str(d_period)
    candidates = [
        f"{value_prefix}stoch_d_{k}_{d}",
        f"{value_prefix}stochastic_d_{k}_{d}",
        f"stoch_d_{k}_{d}",
        f"stochastic_d_{k}_{d}",
        f"{value_prefix}d_{k}_{d}",
        f"d_{k}_{d}",
        f"{value_prefix}D_{k}_{d}",
        f"D_{k}_{d}",
    ]
    regex_candidates = [
        re.compile(
            rf"^{re.escape(value_prefix)}?(stoch|stochastic)?[_\-]?[dD][_\-]?{k}[_\-]{d}$",
            re.I,
        ),
        re.compile(rf"^(stoch|stochastic)?[_\-]?[dD][_\-]?{k}[_\-]{d}$", re.I),
    ]
    col = _find_one(columns, candidates, regex_candidates)
    if col is None:
        candidates_fallback = [
            f"{value_prefix}stoch_d_{d}",
            f"{value_prefix}stochastic_d_{d}",
            f"stoch_d_{d}",
            f"stochastic_d_{d}",
            f"{value_prefix}d_{d}",
            f"d_{d}",
            f"{value_prefix}D_{d}",
            f"D_{d}",
        ]
        regex_candidates_fallback = [
            re.compile(
                rf"^{re.escape(value_prefix)}?(stoch|stochastic)?[_\-]?[dD][_\-]?{d}$",
                re.I,
            ),
            re.compile(rf"^(stoch|stochastic)?[_\-]?[dD][_\-]?{d}$", re.I),
        ]
        col = _find_one(columns, candidates_fallback, regex_candidates_fallback)
    assert col is not None, (
        f"Could not find stochastic %D column for k={k_period}, d={d_period}. Available: {sorted(columns)}"
    )
    return col


def _assert_stoch_range(series: pd.Series, name: str):
    """
    Stochastic %K / %D should be around [0, 100].
    Allow tiny numerical wiggles:
      - hard lower bound: min >= -5 and q0.1% >= -2
      - soft upper bound: q99.9% <= 102 and max <= 110
    """
    s = series.dropna()
    if len(s) == 0:
        return
    smin = float(s.min())
    smax = float(s.max())
    q001 = float(np.quantile(s, 0.001))
    q999 = float(np.quantile(s, 0.999))
    errs = []
    if smin < -5:
        errs.append(f"min {smin:.4f} < -5")
    if q001 < -2:
        errs.append(f"q0.1% {q001:.4f} < -2")
    if q999 > 102:
        errs.append(f"q99.9% {q999:.4f} > 102")
    if smax > 110:
        errs.append(f"max {smax:.4f} > 110")
    if errs:
        msg = (
            f"{name} out-of-range: {', '.join(errs)}\n"
            f"min={smin:.4f}, mean={s.mean():.4f}, max={smax:.4f}, q0.1%={q001:.4f}, q99.9%={q999:.4f}\n"
            f"head:\n{s.head().to_string(index=False)}\n"
            f"tail:\n{s.tail().to_string(index=False)}"
        )
        assert False, msg


# ---------- Main tests ----------


@pytest.mark.parametrize(
    "engine,k_periods,d_periods",
    [
        ("pandas", [14, 21], [3, 9]),
        ("polars", [14, 21], [3, 9]),
        ("pandas", [14], 3),
        ("polars", [21], 3),
    ],
)
def test_stochastic_oscillator(df, engine, k_periods, d_periods):
    """
    Test augment_stochastic_oscillator for grouped and ungrouped cases across engines.
    Verifies:
      - %K and %D columns exist for all requested k (and d) periods
      - values are within reasonable [0,100] with small tolerance
    """
    value_prefix = "close_"

    # Grouped
    res_g = df.groupby("symbol").augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=k_periods,
        d_periods=d_periods,
        engine=engine,
    )
    res_g = _to_pandas(res_g)
    k_list = k_periods if isinstance(k_periods, list) else [k_periods]
    d_list = (
        d_periods
        if isinstance(d_periods, list)
        else ([d_periods] if d_periods is not None else [])
    )

    for k in k_list:
        k_col = _resolve_stoch_k_col(res_g.columns, value_prefix, k)
        _assert_stoch_range(res_g[k_col], f"{k_col} (grouped)")
        for d in d_list:
            d_col = _resolve_stoch_d_col(res_g.columns, value_prefix, k, d)
            _assert_stoch_range(res_g[d_col], f"{d_col} (grouped)")

    # Ungrouped (single symbol)
    res_u = df.query('symbol == "AAPL"').augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=k_periods,
        d_periods=d_periods,
        engine=engine,
    )
    res_u = _to_pandas(res_u)
    for k in k_list:
        k_col = _resolve_stoch_k_col(res_u.columns, value_prefix, k)
        _assert_stoch_range(res_u[k_col], f"{k_col} (ungrouped)")
        for d in d_list:
            d_col = _resolve_stoch_d_col(res_u.columns, value_prefix, k, d)
            _assert_stoch_range(res_u[d_col], f"{d_col} (ungrouped)")


def test_stochastic_oscillator_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - small dataset (may compute with min_periods=1) -> allow <= warm-up NaNs
      - missing columns -> ValueError
      - empty DataFrame with non-numeric dtype -> TypeError
      - invalid k_periods/d_periods (non-integers) -> ValueError/TypeError
    """
    value_prefix = "close_"
    k = 14
    d = 3

    # Small dataset: some implementations compute with min_periods=1 (no NaNs); others may emit warm-up NaNs.
    small = df.query('symbol == "AAPL"').head(5)
    res_small = small.augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[k],
        d_periods=[d],
        engine="pandas",
    )
    k_col = _resolve_stoch_k_col(res_small.columns, value_prefix, k)
    d_col = _resolve_stoch_d_col(res_small.columns, value_prefix, k, d)

    # Allow up to (k-1) NaNs for %K and up to (d-1) NaNs for %D; also allow zero NaNs.
    assert res_small[k_col].isna().sum() <= max(0, k - 1), (
        f"Too many NaNs in %K with k={k} on 5 rows"
    )
    assert res_small[d_col].isna().sum() <= max(0, d - 1), (
        f"Too many NaNs in %D with k={k}, d={d} on 5 rows"
    )

    # Values (non-NaN) stay in reasonable range
    _assert_stoch_range(res_small[k_col], f"{k_col} (small)")
    _assert_stoch_range(res_small[d_col], f"{d_col} (small)")

    # Missing columns
    with pytest.raises(ValueError, match=r"(high|low|close).*not.*found.*data"):
        df[["symbol", "date", "high"]].augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",  # missing in slice
            close_column="close",  # missing in slice
            k_periods=[14],
            d_periods=[3],
        )

    # Empty DataFrame -> non-numeric dtype
    with pytest.raises(TypeError, match=r"(high|low|close).*not.*numeric"):
        empty = pd.DataFrame(columns=["symbol", "date", "high", "low", "close"])
        empty.augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            k_periods=[14],
            d_periods=[3],
        )

    # Invalid k_periods / d_periods (non-integers)
    with pytest.raises(
        (ValueError, TypeError), match=r"k[_\-]?period|int|integer|numeric"
    ):
        df.augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            k_periods=["bad"],
            d_periods=[3],
        )

    with pytest.raises(
        (ValueError, TypeError), match=r"d[_\-]?period|int|integer|numeric"
    ):
        df.augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            k_periods=[14],
            d_periods=["bad"],
        )


def test_stochastic_oscillator_polars_dataframe_roundtrip(pl_df, df):
    pandas_single = df.query('symbol == "AAPL"')
    pandas_result = pandas_single.augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[14, 21],
        d_periods=[3],
    )

    polars_result = tk.augment_stochastic_oscillator(
        data=pl_df.filter(pl.col("symbol") == "AAPL"),
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[14, 21],
        d_periods=[3],
    )

    pd.testing.assert_frame_equal(
        pandas_result.reset_index(drop=True),
        polars_result.to_pandas().reset_index(drop=True),
    )


def test_stochastic_oscillator_polars_groupby_roundtrip(pl_df, df):
    pandas_group = (
        df.groupby("symbol")
        .augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            k_periods=[14],
            d_periods=[3, 5],
        )
        .reset_index(drop=True)
    )

    polars_group = tk.augment_stochastic_oscillator(
        data=pl_df.group_by("symbol"),
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[14],
        d_periods=[3, 5],
    ).to_pandas().reset_index(drop=True)

    pd.testing.assert_frame_equal(pandas_group, polars_group)

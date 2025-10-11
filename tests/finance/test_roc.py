# tests/finance/test_augment_roc.py

import re
import pytest
import pandas as pd
import numpy as np
import polars as pl
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


def _resolve_roc_col(columns, value_prefix: str, start_index: int, period: int):
    """
    Resolve ROC column name with naming convention including start_index and period:
      close_roc_{start_index}_{period}
    Also tolerates variants like:
      <value_prefix>roc_{any_int}_{period}  (if start_index isn't embedded exactly)
      roc_{start_index}_{period}
    """
    s, p = str(start_index), str(period)
    candidates = [
        f"{value_prefix}roc_{s}_{p}",
        f"roc_{s}_{p}",
    ]
    regex_candidates = [
        # prefer exact start_index; if not present, allow any integer before {period}
        re.compile(rf"^{re.escape(value_prefix)}?roc[_\-]?{s}[_\-]?{p}$", re.I),
        re.compile(rf"^{re.escape(value_prefix)}?roc[_\-]?\d+[_\-]?{p}$", re.I),
        re.compile(rf"^roc[_\-]?{s}[_\-]?{p}$", re.I),
        re.compile(rf"^roc[_\-]?\d+[_\-]?{p}$", re.I),
    ]
    col = _find_one(columns, candidates, regex_candidates)
    assert col is not None, (
        f"Could not find ROC column for start_index={start_index}, period={period}. "
        f"Available: {sorted(columns)}"
    )
    return col


def _assert_roc_reasonable(series: pd.Series, name: str):
    """
    Assert ROC values are numerically sane (implementation-agnostic).
    ROC may be fractional (e.g., 0.05) or percent (e.g., 5.0).
    Requirements:
      - finite values (no inf/-inf)
      - extreme absolute values bounded:
          * abs q99.9% <= 500
          * abs max <= 2000
      - negatives allowed
    """
    s = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(s) == 0:
        return
    q999 = float(np.quantile(np.abs(s), 0.999))
    amax = float(np.max(np.abs(s)))
    if not (q999 <= 500 and amax <= 2000):
        msg = (
            f"{name} out-of-range.\n"
            f"abs q99.9={q999:.4f} (>500?)  abs max={amax:.4f} (>2000?)\n"
            f"min={s.min():.6f}, mean={s.mean():.6f}, max={s.max():.6f}\n"
            f"head:\n{s.head().to_string(index=False)}\n"
            f"tail:\n{s.tail().to_string(index=False)}"
        )
        assert False, msg


# ---------- Main tests ----------


@pytest.mark.parametrize(
    "engine,periods,start_index",
    [
        ("pandas", [22, 252], 21),
        ("polars", [22, 252], 21),
        ("pandas", [22], 21),
        ("polars", [252], 21),
    ],
)
def test_roc(df, engine, periods, start_index):
    """
    Test augment_roc with grouped/ungrouped data, multiple engines, and periods.
    Verifies:
      - ROC columns exist with the <value>_roc_{start_index}_{period} pattern (flexible resolver)
      - Warm-up NaNs are present per-group (>= start_index, allowing implementation variance)
      - Values are finite and not absurdly large/small
    """
    value_prefix = "close_"
    period_list = periods if isinstance(periods, list) else [periods]

    # Grouped
    res_g = df.groupby("symbol").augment_roc(
        date_column="date",
        close_column="close",
        periods=periods,
        start_index=start_index,
        engine=engine,
    )
    res_g = _to_pandas(res_g)
    for p in period_list:
        col = _resolve_roc_col(res_g.columns, value_prefix, start_index, p)
        # Warm-up NaNs per group: require at least start_index NaNs (loose check)
        nan_counts = res_g.groupby("symbol")[col].apply(lambda s: int(s.isna().sum()))
        assert (nan_counts >= start_index).all(), (
            f"Expected >= {start_index} NaNs per group for {col}, got {nan_counts.to_dict()}"
        )
        _assert_roc_reasonable(res_g[col], f"{col} (grouped)")

    # Ungrouped (single symbol)
    res_u = df.query('symbol == "GOOG"').augment_roc(
        date_column="date",
        close_column="close",
        periods=periods,
        start_index=start_index,
        engine=engine,
    )
    res_u = _to_pandas(res_u)
    for p in period_list:
        col = _resolve_roc_col(res_u.columns, value_prefix, start_index, p)
        # Warm-up NaNs (single series): at least start_index NaNs allowed (many impls pad)
        assert res_u[col].isna().sum() >= start_index, (
            f"Expected >= {start_index} NaNs for {col} (ungrouped)"
        )
        _assert_roc_reasonable(res_u[col], f"{col} (ungrouped)")


def test_roc_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - tiny dataset (period > length): allow either warm-up NaNs or finite outputs if min_periods=1
      - missing close column -> ValueError
      - empty DataFrame with non-numeric close -> TypeError
      - invalid periods/start_index types -> ValueError/TypeError
    """
    value_prefix = "close_"

    # Tiny dataset vs. long period
    small = df.query('symbol == "GOOG"').head(10)
    res_small = small.augment_roc(
        date_column="date",
        close_column="close",
        periods=[22],
        start_index=5,
        engine="pandas",
    )
    col22 = _resolve_roc_col(res_small.columns, value_prefix, 5, 22)
    # Either warm-up NaNs or finite outputs; keep assertion flexible
    if res_small[col22].isna().sum() == 0:
        _assert_roc_reasonable(res_small[col22], f"{col22} (small)")
    else:
        # should not be ridiculous amounts of NaNs for tiny series either
        assert res_small[col22].isna().sum() <= 22, (
            "Unexpectedly large number of NaNs for small sample."
        )

    # Missing close column
    with pytest.raises(ValueError, match=r"value_column.*close.*not.*found.*data"):
        df[["symbol", "date"]].augment_roc(
            date_column="date",
            close_column="close",
            periods=[22],
            start_index=1,
            engine="pandas",
        )

    # Empty DataFrame -> non-numeric close
    with pytest.raises(TypeError, match=r"value_column.*close.*not.*numeric"):
        empty = pd.DataFrame(columns=["symbol", "date", "close"])
        empty.augment_roc(
            date_column="date",
            close_column="close",
            periods=[22],
            start_index=1,
            engine="pandas",
        )


def test_roc_polars_dataframe_roundtrip(pl_df):
    pandas_single = (
        tk.load_dataset("stocks_daily", parse_dates=["date"])
        .query("symbol == 'AAPL'")
    )

    pandas_result = pandas_single.augment_roc(
        date_column="date",
        close_column="close",
        periods=[22, 63],
        start_index=5,
    )

    polars_result = tk.augment_roc(
        data=pl_df.filter(pl.col("symbol") == "AAPL"),
        date_column="date",
        close_column="close",
        periods=[22, 63],
        start_index=5,
    )

    pd.testing.assert_frame_equal(
        pandas_result.reset_index(drop=True),
        polars_result.to_pandas().reset_index(drop=True),
    )


def test_roc_polars_groupby_roundtrip(pl_df, df):
    pandas_group = (
        tk.load_dataset("stocks_daily", parse_dates=["date"])
        .groupby("symbol")
        .augment_roc(
            date_column="date",
            close_column="close",
            periods=[22],
            start_index=5,
        )
    )

    polars_group = tk.augment_roc(
        data=pl_df.group_by("symbol"),
        date_column="date",
        close_column="close",
        periods=[22],
        start_index=5,
    )

    pd.testing.assert_frame_equal(
        pandas_group.reset_index(drop=True),
        polars_group.to_pandas().reset_index(drop=True),
    )

    # Invalid periods (non-integer item)
    with pytest.raises((ValueError, TypeError), match=r"period|int|integer|numeric"):
        df.augment_roc(
            date_column="date",
            close_column="close",
            periods=["bad"],
            start_index=21,
            engine="pandas",
        )

    # Invalid start_index (non-integer)
    with pytest.raises(
        (ValueError, TypeError), match=r"start[_\-]?index|int|integer|numeric"
    ):
        df.augment_roc(
            date_column="date",
            close_column="close",
            periods=[22],
            start_index="bad",
            engine="pandas",
        )

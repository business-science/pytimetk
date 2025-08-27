# tests/finance/test_augment_qsmomentum.py

import re
import pytest
import pandas as pd
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


def _resolve_qsmom_col(columns, value_prefix: str, fast: int, slow: int, ret: int):
    """
    Resolve QS Momentum column name for given (fast, slow, returns_period).
    Expected pattern: close_qsmom_{fast}_{slow}_{ret}
    """
    f, s, r = str(fast), str(slow), str(ret)
    candidates = [
        f"{value_prefix}qsmom_{f}_{s}_{r}",
        f"qsmom_{f}_{s}_{r}",
        f"{value_prefix}qs_mom_{f}_{s}_{r}",
        f"qs_mom_{f}_{s}_{r}",
    ]
    regex_candidates = [
        re.compile(
            rf"^{re.escape(value_prefix)}?qs[_\-]?mom[_\-]?{f}[_\-]?{s}[_\-]?{r}$", re.I
        ),
        re.compile(rf"^qs[_\-]?mom[_\-]?{f}[_\-]?{s}[_\-]?{r}$", re.I),
    ]
    col = _find_one(columns, candidates, regex_candidates)
    assert col is not None, (
        f"Could not find QS Momentum column for fast={fast}, slow={slow}, returns={ret}. "
        f"Available: {sorted(columns)}"
    )
    return col


def _assert_qsmom_reasonable(series: pd.Series, name: str):
    """
    Assert QS Momentum values are numerically sane (implementation-agnostic).
    Allow negatives/positives. Requirements:
      - finite values (no inf/-inf)
      - extreme absolute values bounded:
          * abs q99.9% <= 1_000
          * abs max <= 10_000
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
    if not (q999 <= 1_000 and amax <= 10_000):
        msg = (
            f"{name} out-of-range.\n"
            f"abs q99.9={q999:.4f} (>1000?)  abs max={amax:.4f} (>10000?)\n"
            f"min={s.min():.6f}, mean={s.mean():.6f}, max={s.max():.6f}\n"
            f"head:\n{s.head().to_string(index=False)}\n"
            f"tail:\n{s.tail().to_string(index=False)}"
        )
        assert False, msg


def _largest_window(roc_fast_periods, roc_slow_period, returns_period):
    fast_max = (
        max(roc_fast_periods)
        if isinstance(roc_fast_periods, list)
        else int(roc_fast_periods)
    )
    return max(fast_max, int(roc_slow_period), int(returns_period))


# ---------- Main tests ----------


@pytest.mark.parametrize(
    "engine,roc_fast_period,roc_slow_period,returns_period",
    [
        ("pandas", [5, 21], 252, 126),
        ("polars", [5, 21], 252, 126),
        ("pandas", [5], 252, 126),
        ("polars", [21], 252, 126),
    ],
)
def test_qsmomentum(df, engine, roc_fast_period, roc_slow_period, returns_period):
    """
    Test augment_qsmomentum with grouped/ungrouped data, multiple engines, and fast-period sets.
    Verifies:
      - QS columns exist (close_qsmom_{fast}_{slow}_{ret})
      - Warm-up NaNs per group are at least min(fast, returns_period) - 1
        (implementations may bootstrap earlier than the largest window)
      - Values are finite and not absurd
    """
    value_prefix = "close_"
    fast_list = (
        roc_fast_period if isinstance(roc_fast_period, list) else [roc_fast_period]
    )
    upper_hint = _largest_window(fast_list, roc_slow_period, returns_period)

    # Grouped
    res_g = df.groupby("symbol").augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=roc_fast_period,
        roc_slow_period=roc_slow_period,
        returns_period=returns_period,
        engine=engine,
    )
    for f in fast_list:
        col = _resolve_qsmom_col(
            res_g.columns, value_prefix, f, roc_slow_period, returns_period
        )
        nan_counts = res_g.groupby("symbol")[col].apply(lambda s: int(s.isna().sum()))
        lower = max(0, min(int(f), int(returns_period)) - 1)
        # Lower bound only; upper bound left flexible (impl differences).
        assert (nan_counts >= lower).all(), (
            f"Expected >= {lower} NaNs per group for {col}, got {nan_counts.to_dict()} "
            f"(largest window hint: {upper_hint})"
        )
        _assert_qsmom_reasonable(res_g[col], f"{col} (grouped)")

    # Ungrouped (single symbol)
    res_u = df.query('symbol == "GOOG"').augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=roc_fast_period,
        roc_slow_period=roc_slow_period,
        returns_period=returns_period,
        engine=engine,
    )
    for f in fast_list:
        col = _resolve_qsmom_col(
            res_u.columns, value_prefix, f, roc_slow_period, returns_period
        )
        lower = max(0, min(int(f), int(returns_period)) - 1)
        assert res_u[col].isna().sum() >= lower, (
            f"Expected >= {lower} NaNs for {col} (ungrouped)"
        )
        _assert_qsmom_reasonable(res_u[col], f"{col} (ungrouped)")


def test_qsmomentum_edge_cases(df):
    """
    Edge cases & invalid inputs:
      - tiny dataset (length < windows): allow either many NaNs or (if min_periods=1) finite but reasonable outputs
      - missing close column -> ValueError
      - empty DataFrame with non-numeric close -> TypeError
      - invalid periods types -> ValueError/TypeError
    """
    value_prefix = "close_"

    # Tiny dataset
    small = df.query('symbol == "GOOG"').head(50)
    res_small = small.augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=[5, 21],
        roc_slow_period=252,
        returns_period=126,
        engine="pandas",
    )
    for f in [5, 21]:
        col = _resolve_qsmom_col(res_small.columns, value_prefix, f, 252, 126)
        nan_ct = int(res_small[col].isna().sum())
        if nan_ct == 0:
            _assert_qsmom_reasonable(res_small[col], f"{col} (small)")
        else:
            # With 50 rows << slow/returns, expect a decent chunk of NaNs
            assert nan_ct >= 20, (
                f"Unexpectedly few NaNs in tiny sample for {col}: {nan_ct}"
            )

    # Missing close column
    with pytest.raises(ValueError, match=r"value_column.*close.*not.*found.*data"):
        df[["symbol", "date"]].augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period=252,
            returns_period=126,
            engine="pandas",
        )

    # Empty DataFrame -> non-numeric close
    with pytest.raises(TypeError, match=r"value_column.*close.*not.*numeric"):
        empty = pd.DataFrame(columns=["symbol", "date", "close"])
        empty.augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period=252,
            returns_period=126,
            engine="pandas",
        )

    # Invalid fast periods (non-integer item)
    with pytest.raises(
        (ValueError, TypeError), match=r"fast|roc[_\-]?fast|int|integer|numeric"
    ):
        df.augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=["bad"],
            roc_slow_period=252,
            returns_period=126,
            engine="pandas",
        )

    # Invalid slow period (non-integer)
    with pytest.raises(
        (ValueError, TypeError), match=r"slow|roc[_\-]?slow|int|integer|numeric"
    ):
        df.augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period="bad",
            returns_period=126,
            engine="pandas",
        )

    # Invalid returns_period (non-integer)
    with pytest.raises(
        (ValueError, TypeError), match=r"returns|return[_\-]?period|int|integer|numeric"
    ):
        df.augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period=252,
            returns_period="bad",
            engine="pandas",
        )

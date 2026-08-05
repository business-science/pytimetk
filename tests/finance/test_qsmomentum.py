# tests/finance/test_augment_qsmomentum.py

import re
import pytest
import pandas as pd
import numpy as np
import pytimetk as tk
import os
import multiprocessing as mp
from pytimetk.utils.selection import contains

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


def _manual_qsmomentum(close, fast, slow, returns_period):
    close = np.asarray(close, dtype=float)
    fast_close = close[-(fast + 1)]
    slow_close = close[-(slow + 1)]
    slow_leg = (fast_close - slow_close) / (slow_close + 1e-10)
    fast_leg = (close[-1] - fast_close) / (fast_close + 1e-10)
    returns = close[1:] / close[:-1] - 1
    volatility = np.std(returns[-returns_period:], ddof=0)
    return (slow_leg - fast_leg) / volatility


def _make_qsmomentum_data(rows_per_group=40):
    frames = []
    for group_number, symbol in enumerate(["A", "B"]):
        row = np.arange(rows_per_group, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "market": f"M{group_number}",
                    "symbol": symbol,
                    "date": pd.date_range("2024-01-01", periods=rows_per_group),
                    "close": (
                        100.0 * (group_number + 1)
                        + 0.3 * row
                        + np.sin(row / 2.0)
                        + 0.02 * row**2
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


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
      - Each group has exactly slow-period warm-up NaNs
      - Values are finite and not absurd
    """
    value_prefix = "close_"
    fast_list = (
        roc_fast_period if isinstance(roc_fast_period, list) else [roc_fast_period]
    )
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
        assert (nan_counts == roc_slow_period).all(), (
            f"Expected {roc_slow_period} NaNs per group for {col}, "
            f"got {nan_counts.to_dict()}"
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
        assert res_u[col].isna().sum() == roc_slow_period, (
            f"Expected {roc_slow_period} NaNs for {col} (ungrouped)"
        )
        _assert_qsmom_reasonable(res_u[col], f"{col} (ungrouped)")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_qsmomentum_matches_documented_formula(engine):
    data = _make_qsmomentum_data(rows_per_group=20).query("symbol == 'A'")
    result = data.augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=3,
        roc_slow_period=10,
        returns_period=10,
        engine=engine,
    )
    expected = _manual_qsmomentum(
        data["close"].iloc[-11:],
        fast=3,
        slow=10,
        returns_period=10,
    )

    assert result["close_qsmom_3_10_10"].isna().sum() == 10
    assert result["close_qsmom_3_10_10"].iloc[-1] == pytest.approx(
        expected, rel=1e-10, abs=1e-12
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "fast_periods,slow_periods,returns_periods",
    [
        ([2, 4], [8, 10], [3, 5]),
        ([4, 2], [10, 8], [5, 3]),
    ],
)
def test_qsmomentum_multi_parameter_calls_match_single_calls(
    engine, fast_periods, slow_periods, returns_periods
):
    data = _make_qsmomentum_data()
    grouped = data.groupby("symbol")
    multi = grouped.augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=fast_periods,
        roc_slow_period=slow_periods,
        returns_period=returns_periods,
        engine=engine,
    )

    for fast in fast_periods:
        for slow in slow_periods:
            for returns_period in returns_periods:
                if fast >= slow or returns_period > slow:
                    continue
                column = f"close_qsmom_{fast}_{slow}_{returns_period}"
                single = grouped.augment_qsmomentum(
                    date_column="date",
                    close_column="close",
                    roc_fast_period=fast,
                    roc_slow_period=slow,
                    returns_period=returns_period,
                    engine=engine,
                )
                np.testing.assert_allclose(
                    multi[column],
                    single[column],
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                )


def test_qsmomentum_pandas_and_polars_match_with_shuffled_null_input():
    data = _make_qsmomentum_data()
    data.loc[(data["symbol"] == "B") & (data["date"] == "2024-01-15"), "close"] = np.nan
    data = data.sample(frac=1, random_state=123)
    outputs = {}

    for engine in ["pandas", "polars"]:
        outputs[engine] = data.groupby("symbol").augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[2, 4],
            roc_slow_period=[8, 10],
            returns_period=[3, 5],
            engine=engine,
        )

    qsmomentum_columns = [
        column for column in outputs["pandas"].columns if "_qsmom_" in column
    ]
    for column in qsmomentum_columns:
        np.testing.assert_allclose(
            outputs["pandas"][column],
            outputs["polars"][column],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_qsmomentum_supports_multiple_grouping_columns(engine):
    data = _make_qsmomentum_data()
    grouped = data.groupby(["market", "symbol"]).augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=2,
        roc_slow_period=10,
        returns_period=5,
        engine=engine,
    )
    isolated = pd.concat(
        [
            group.augment_qsmomentum(
                date_column="date",
                close_column="close",
                roc_fast_period=2,
                roc_slow_period=10,
                returns_period=5,
                engine=engine,
            )
            for _, group in data.groupby(["market", "symbol"])
        ]
    ).sort_index()

    np.testing.assert_allclose(
        grouped["close_qsmom_2_10_5"],
        isolated["close_qsmom_2_10_5"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


@pytest.mark.parametrize("invalid_period", [0, -1, [], [2.5], True])
@pytest.mark.parametrize(
    "parameter", ["roc_fast_period", "roc_slow_period", "returns_period"]
)
def test_qsmomentum_rejects_invalid_period_values(parameter, invalid_period):
    data = _make_qsmomentum_data(rows_per_group=12)
    arguments = {
        "roc_fast_period": 2,
        "roc_slow_period": 10,
        "returns_period": 5,
    }
    arguments[parameter] = invalid_period

    with pytest.raises((TypeError, ValueError), match="positive integer"):
        data.augment_qsmomentum(
            date_column="date",
            close_column="close",
            **arguments,
        )


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
    with pytest.raises(
        ValueError, match=r"`value_column` \(close\) not found in `data`"
    ):
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


def test_qsmomentum_supports_tidy_selectors(df):
    result = df.groupby("symbol").augment_qsmomentum(
        date_column=contains("dat"),
        close_column=contains("clos"),
        roc_fast_period=5,
        roc_slow_period=252,
        returns_period=126,
    )
    assert "close_qsmom_5_252_126" in result.columns

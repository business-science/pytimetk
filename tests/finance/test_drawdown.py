import pytest
import pandas as pd
import pytimetk as tk
import os
import multiprocess as mp
import numpy as np

# Align multiprocessing behaviour with existing finance tests to avoid warnings.
mp.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@pytest.fixture(scope="module")
def df():
    return tk.load_dataset("stocks_daily", parse_dates=["date"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_drawdown_grouped_and_ungrouped(df, engine):
    """Ensure augment_drawdown behaves for both pandas and polars engines."""
    expected_cols = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted",
        "close_peak",
        "close_drawdown",
        "close_drawdown_pct",
    ]

    # Grouped case
    result_grouped = df.groupby("symbol").augment_drawdown(
        date_column="date",
        close_column="close",
        engine=engine,
    )
    assert list(result_grouped.columns) == expected_cols
    assert result_grouped.shape == (16194, len(expected_cols))

    # Peak should be the cumulative max per symbol
    expected_peak = result_grouped.groupby("symbol")["close"].cummax()
    pd.testing.assert_series_equal(
        expected_peak.reset_index(drop=True),
        result_grouped["close_peak"].reset_index(drop=True),
        check_names=False,
    )

    # Drawdown is non-positive and equals close - peak
    dd = result_grouped["close"] - result_grouped["close_peak"]
    pd.testing.assert_series_equal(
        dd.reset_index(drop=True),
        result_grouped["close_drawdown"].reset_index(drop=True),
        check_names=False,
    )
    assert (result_grouped["close_drawdown"] <= 1e-9).all()

    # Percentage drawdown equals drawdown / peak (guard zeros)
    pct_expected = np.divide(
        result_grouped["close_drawdown"],
        result_grouped["close_peak"],
        out=np.zeros_like(result_grouped["close_drawdown"]),
        where=result_grouped["close_peak"] != 0,
    )
    np.testing.assert_allclose(
        pct_expected,
        result_grouped["close_drawdown_pct"],
        rtol=1e-7,
        atol=1e-9,
    )

    # Ungrouped (single symbol) retains behaviour
    single = df.query('symbol == "AAPL"').augment_drawdown(
        date_column="date",
        close_column="close",
        engine=engine,
    )
    assert list(single.columns) == expected_cols
    assert single.shape == (2699, len(expected_cols))
    np.testing.assert_allclose(
        single["close"].cummax(),
        single["close_peak"],
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        single["close"] - single["close_peak"],
        single["close_drawdown"],
        rtol=1e-7,
        atol=1e-9,
    )


def test_drawdown_invalid_inputs(df):
    """Validate guard clauses shared across engines."""
    # Missing close column
    with pytest.raises(
        ValueError, match=r"`value_column` \(close\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_drawdown(
            date_column="date",
            close_column="close",
            engine="pandas",
        )

    # Non-numeric close column
    bad = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2020-01-01"]), "close": ["oops"]})
    with pytest.raises(
        TypeError, match=r"`value_column` \(close\) is not a numeric dtype"
    ):
        bad.augment_drawdown(
            date_column="date",
            close_column="close",
            engine="pandas",
        )

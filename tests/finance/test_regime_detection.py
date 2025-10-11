import pytest
import pandas as pd
import pytimetk as tk
import os
import multiprocess as mp
import numpy as np

# Skip the entire module if hmmlearn (optional dependency) is not present.
pytest.importorskip("hmmlearn.hmm")

# Align multiprocessing/threading config with the rest of the finance suite.
mp.set_start_method("spawn", force=True)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@pytest.fixture(scope="module")
def regime_df():
    base = tk.load_dataset("stocks_daily", parse_dates=["date"])
    symbols = ["AAPL", "MSFT", "GOOG"]
    filtered = base[base["symbol"].isin(symbols)].copy()
    # Limit rows per symbol for test speed.
    return filtered.groupby("symbol").head(180).reset_index(drop=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_regime_detection_grouped_and_single(regime_df, engine):
    """Regime detection adds regime columns for both pandas and polars engines."""
    column_name = "close_regime_60"

    grouped = regime_df.groupby("symbol").augment_regime_detection(
        date_column="date",
        close_column="close",
        window=60,
        n_regimes=2,
        step_size=20,
        n_iter=20,
        n_jobs=1,
        engine=engine,
    )
    assert column_name in grouped.columns
    assert grouped.shape[0] == regime_df.shape[0]

    valid = grouped[column_name].dropna()
    if len(valid) > 0:
        unique_vals = set(valid.unique())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"Unexpected regime labels {unique_vals} for engine={engine}"
        )

    # Ungrouped (single symbol) behaviour
    single = regime_df.query('symbol == "AAPL"').augment_regime_detection(
        date_column="date",
        close_column="close",
        window=60,
        n_regimes=2,
        step_size=20,
        n_iter=20,
        n_jobs=1,
        engine=engine,
    )
    assert column_name in single.columns
    valid_single = single[column_name].dropna()
    if len(valid_single) > 0:
        assert set(valid_single.unique()).issubset({0.0, 1.0})


def test_regime_detection_invalid_inputs(regime_df):
    """Basic input validation that should trigger before engine dispatch."""
    with pytest.raises(ValueError, match="n_regimes must be at least 2"):
        regime_df.augment_regime_detection(
            date_column="date",
            close_column="close",
            window=60,
            n_regimes=1,
        )

    with pytest.raises(ValueError, match="step_size must be at least 1"):
        regime_df.augment_regime_detection(
            date_column="date",
            close_column="close",
            window=60,
            n_regimes=2,
            step_size=0,
        )

    bad = pd.DataFrame({"symbol": ["A"], "date": pd.to_datetime(["2020-01-01"]), "close": ["oops"]})
    with pytest.raises(
        TypeError, match=r"`value_column` \(close\) is not a numeric dtype"
    ):
        bad.augment_regime_detection(
            date_column="date",
            close_column="close",
            window=30,
        )

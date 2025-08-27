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
    "engine,window,metrics",
    [
        ("pandas", [63, 252], None),
        ("pandas", [252], ["volatility_annualized", "sharpe_ratio"]),
        ("polars", [63, 252], None),
        ("polars", [252], ["volatility_annualized", "sharpe_ratio"]),
    ],
)
def test_rolling_risk_metrics(df, engine, window, metrics):
    """Test Rolling Risk Metrics with grouped and ungrouped data, different engines, windows, and metrics."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=window,
        metrics=metrics,
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
    window_list = window if isinstance(window, list) else [window]
    metric_list = (
        metrics
        if metrics
        else [
            "sharpe_ratio",
            "sortino_ratio",
            "volatility_annualized",
            "omega_ratio",
            "skewness",
            "kurtosis",
        ]
    )  # Adjusted to 6 default metrics from output
    expected_cols += [f"close_{m}_{w}" for w in window_list for m in metric_list]
    if result_grouped.shape[1] != len(expected_cols):
        print("Actual columns:", result_grouped.columns.tolist())
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)}), got {result_grouped.shape}"
    )
    # Verify NaNs (engine-specific, loose assertion based on observed ~w/2 for polars, w/2 for pandas)
    for w in window_list:
        for m in metric_list:
            col = f"close_{m}_{w}"
            nan_counts = result_grouped.groupby("symbol")[col].apply(
                lambda x: x.isna().sum()
            )
            print(f"NaN counts for {col} ({engine}):", nan_counts.to_dict())
            assert all(nan_counts >= w // 2 - 1), (
                f"Expected at least {w // 2 - 1} NaNs per group for {col}"
            )
    # Verify volatility_annualized >= 0
    for w in window_list:
        assert (
            result_grouped[f"close_volatility_annualized_{w}"].dropna() >= 0
        ).all(), f"Volatility_annualized {w} values below 0"
    # Check column set equality (ignore order)
    assert set(result_grouped.columns) == set(expected_cols), "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=window,
        metrics=metrics,
        engine=engine,
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    for w in window_list:
        for m in metric_list:
            col = f"close_{m}_{w}"
            assert result_single[col].isna().sum() >= w // 2 - 1, (
                f"Expected at least {w // 2 - 1} NaNs for {col}"
            )
    assert set(result_single.columns) == set(expected_cols), "Incorrect column names"


def test_rolling_risk_metrics_edge_cases(df):
    """Test Rolling Risk Metrics with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=[63],
        metrics=["volatility_annualized", "sharpe_ratio"],
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    assert result_small["close_volatility_annualized_63"].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(
        ValueError, match="`value_column` \\(close\\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_rolling_risk_metrics(
            date_column="date", close_column="close", window=[63], engine="pandas"
        )

    # Empty DataFrame
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
        empty_df.augment_rolling_risk_metrics(
            date_column="date", close_column="close", window=[63], engine="pandas"
        )

    # Invalid window (use negative to trigger min_periods error)
    with pytest.raises(ValueError, match="min_periods must be >= 0"):
        df.augment_rolling_risk_metrics(
            date_column="date", close_column="close", window=[-1], engine="pandas"
        )

    # Invalid metrics
    with pytest.raises(ValueError, match=r"Invalid metrics:.*invalid_metric"):
        df.augment_rolling_risk_metrics(
            date_column="date",
            close_column="close",
            window=[63],
            metrics=["invalid_metric"],
            engine="pandas",
        )

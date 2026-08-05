import pytest
import numpy as np
import pandas as pd
import pytimetk as tk
import os
import multiprocessing as mp
from itertools import product
from pytimetk.finance.fip_momentum import _compute_fip_series
from pytimetk.utils.selection import contains

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
    "engine,window,skip_window", product(["pandas", "polars"], [[63], [63, 252]], [21])
)
def test_fip_momentum(df, engine, window, skip_window):
    """Test FIP Momentum with grouped and ungrouped data, different engines, windows, and skip_window."""
    # Grouped test
    result_grouped = df.groupby("symbol").augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=window,
        skip_window=skip_window,
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
    expected_cols += [f"close_fip_momentum_{w}" for w in window]
    assert result_grouped.shape == (16194, len(expected_cols)), (
        f"Expected shape (16194, {len(expected_cols)})"
    )
    # A complete window of returns requires w + 1 closing prices, so each
    # symbol starts with exactly w null feature values.
    for w in window:
        nan_counts = result_grouped.groupby("symbol")[f"close_fip_momentum_{w}"].apply(
            lambda x: x.isna().sum()
        )
        print(f"NaN counts for window {w} ({engine}):", nan_counts.to_dict())
        assert all(nan_counts == w), (
            f"Expected exactly {w} NaNs per group for window {w} ({engine})"
        )
    assert list(result_grouped.columns) == expected_cols, "Incorrect column names"

    # Ungrouped test (single symbol)
    result_single = df.query('symbol == "GOOG"').augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=window,
        skip_window=skip_window,
        engine=engine,
    )
    assert result_single.shape == (2699, len(expected_cols)), (
        f"Expected shape (2699, {len(expected_cols)})"
    )
    for w in window:
        assert result_single[f"close_fip_momentum_{w}"].isna().sum() == w, (
            f"Expected exactly {w} NaNs for window {w} ({engine})"
        )
    assert list(result_single.columns) == expected_cols, "Incorrect column names"


def _make_fip_test_data(rows_per_symbol=30):
    frames = []
    for symbol_index, symbol in enumerate(["A", "B"]):
        row = np.arange(rows_per_symbol, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": pd.date_range("2024-01-01", periods=rows_per_symbol),
                    "close": (
                        100.0 * (symbol_index + 1) + 0.2 * row + np.sin(row / 3.0)
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("fip_method", ["original", "modified"])
@pytest.mark.parametrize("windows", [[4, 10], [10, 4]])
def test_fip_momentum_multi_window_matches_single_window(engine, fip_method, windows):
    data = _make_fip_test_data()
    grouped = data.groupby("symbol")
    multi = grouped.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=windows,
        engine=engine,
        fip_method=fip_method,
    )

    for window in windows:
        column = f"close_fip_momentum_{window}"
        single = grouped.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=window,
            engine=engine,
            fip_method=fip_method,
        )
        assert multi[column].notna().any()
        np.testing.assert_allclose(
            multi[column],
            single[column],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fip_momentum_group_boundaries_match_isolated_symbols(engine):
    data = pd.DataFrame(
        {
            "symbol": ["A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "close": [100.0, 100.0, 10.0, 10.0, 10.0],
        }
    )
    grouped = data.groupby("symbol").augment_fip_momentum(
        date_column="date", close_column="close", window=2, engine=engine
    )
    isolated = pd.concat(
        [
            group.augment_fip_momentum(
                date_column="date", close_column="close", window=2, engine=engine
            )
            for _, group in data.groupby("symbol")
        ]
    ).sort_index()

    np.testing.assert_allclose(
        grouped["close_fip_momentum_2"],
        isolated["close_fip_momentum_2"],
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )
    assert (
        grouped.groupby("symbol")["close_fip_momentum_2"]
        .apply(lambda values: values.iloc[:2].isna().all())
        .all()
    )


@pytest.mark.parametrize("fip_method", ["original", "modified"])
def test_fip_momentum_engines_match_with_shuffled_null_input(fip_method):
    data = _make_fip_test_data()
    data.loc[(data["symbol"] == "B") & (data["date"] == "2024-01-13"), "close"] = np.nan
    data = data.sample(frac=1, random_state=123)

    results = {}
    for engine in ["pandas", "polars"]:
        results[engine] = data.groupby("symbol").augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[4, 10],
            engine=engine,
            fip_method=fip_method,
        )

    for window in [4, 10]:
        column = f"close_fip_momentum_{window}"
        np.testing.assert_allclose(
            results["pandas"][column],
            results["polars"][column],
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fip_momentum_skip_window_is_applied_per_group(engine):
    data = _make_fip_test_data(rows_per_symbol=8)
    result = data.groupby("symbol").augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=2,
        skip_window=4,
        engine=engine,
    )

    for _, group in result.groupby("symbol"):
        assert group["close_fip_momentum_2"].iloc[:4].isna().all()
        assert group["close_fip_momentum_2"].iloc[4:].notna().all()


def test_compute_fip_series_requires_a_complete_window():
    returns = np.array([np.nan, 0.01, 0.02, 0.03, 0.04])
    result = _compute_fip_series(
        returns,
        window=3,
        fip_method="original",
        skip_window=0,
    )

    assert np.isnan(result[:3]).all()
    assert np.isfinite(result[3:]).all()


def test_fip_momentum_edge_cases(df):
    """Test FIP Momentum with edge cases and invalid inputs."""
    # Small dataset
    small_df = df.query('symbol == "GOOG"').head(10)
    result_small = small_df.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=[63],
        skip_window=21,
        engine="pandas",
    )
    assert result_small.shape[0] == 10
    assert result_small["close_fip_momentum_63"].isna().sum() > 0, (
        "Expected NaNs for insufficient data"
    )

    # Missing columns
    with pytest.raises(
        ValueError, match=r"`value_column` \(close\) not found in `data`"
    ):
        df[["symbol", "date"]].augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[63],
            skip_window=21,
            engine="pandas",
        )

    # Empty DataFrame
    empty_df = pd.DataFrame(columns=["symbol", "date", "close"])
    with pytest.raises(
        TypeError, match="`value_column` \\(close\\) is not a numeric dtype"
    ):
        empty_df.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[63],
            skip_window=21,
            engine="pandas",
        )

    # Invalid window
    with pytest.raises(ValueError, match="All window values must be positive integers"):
        df.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[0],
            skip_window=21,
            engine="pandas",
        )

    # Invalid skip_window (note: no validation in function, so no error raised; consider adding to pytimetk)
    # Run the code and check for reasonable output or add assertion if negative skip_window causes issues
    result_negative_skip = df.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=[63],
        skip_window=-1,
        engine="pandas",
    )
    assert result_negative_skip.shape[0] == 16194, (
        "Expected full shape for negative skip_window"
    )
    assert result_negative_skip["close_fip_momentum_63"].isna().sum() > 0, (
        "Expected NaNs with negative skip_window"
    )


def test_fip_momentum_supports_tidy_selectors(df):
    result = df.groupby("symbol").augment_fip_momentum(
        date_column=contains("dat"),
        close_column=contains("clos"),
        window=63,
    )
    assert "close_fip_momentum_63" in result.columns

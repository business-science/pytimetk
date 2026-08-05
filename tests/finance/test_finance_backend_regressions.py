import numpy as np
import pandas as pd
import pytest

import pytimetk  # noqa: F401 - registers pandas and Polars accessors


@pytest.fixture()
def grouped_prices():
    n = 48
    step = np.arange(n)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close_a = 900 + np.linspace(0, 80, n) + 8 * np.sin(step / 2)
    close_b = 20 + np.linspace(0, 4, n) + 1.5 * np.sin(step * 0.8)

    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "exchange": "X",
                    "symbol": "A",
                    "date": dates,
                    "close": close_a,
                }
            ),
            pd.DataFrame(
                {
                    "exchange": "Y",
                    "symbol": "B",
                    "date": dates,
                    "close": close_b,
                }
            ),
        ],
        ignore_index=True,
    )
    frame["high"] = frame["close"] + np.where(frame["symbol"].eq("A"), 10, 0.7)
    frame["low"] = frame["close"] - np.where(frame["symbol"].eq("A"), 10, 0.7)
    grouped_step = np.tile(step, 2)
    frame["benchmark"] = np.where(
        frame["symbol"].eq("A"),
        4000 + 2 * grouped_step,
        100 + 0.3 * grouped_step,
    )
    frame["benchmark"] += np.sin(np.arange(len(frame)) * 0.2)
    return frame


def _assert_columns_close(left, right, columns):
    pd.testing.assert_frame_equal(
        left.loc[:, columns].reset_index(drop=True),
        right.loc[:, columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=1e-8,
        atol=1e-10,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_adx_grouped_calculation_is_isolated(grouped_prices, engine):
    grouped = grouped_prices.groupby("symbol").augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=5,
        engine=engine,
    )
    isolated = grouped_prices.loc[grouped_prices["symbol"].eq("B")].augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=5,
        engine=engine,
    )

    columns = ["close_plus_di_5", "close_minus_di_5", "close_adx_5"]
    grouped_b = grouped.loc[grouped["symbol"].eq("B")].sort_values("date")
    _assert_columns_close(grouped_b, isolated.sort_values("date"), columns)


def test_adx_engines_use_the_same_warmup_and_values(grouped_prices):
    outputs = {}
    for engine in ["pandas", "polars"]:
        outputs[engine] = grouped_prices.groupby("symbol").augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=5,
            engine=engine,
        )

    columns = ["close_plus_di_5", "close_minus_di_5", "close_adx_5"]
    _assert_columns_close(outputs["pandas"], outputs["polars"], columns)


def test_rsi_zero_loss_behavior_matches_across_engines():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rising = pd.DataFrame(
        {"date": dates, "close": np.arange(1, len(dates) + 1, dtype=float)}
    )
    flat = pd.DataFrame({"date": dates, "close": 10.0})

    for engine in ["pandas", "polars"]:
        rising_result = rising.augment_rsi(
            date_column="date", close_column="close", periods=5, engine=engine
        )
        flat_result = flat.augment_rsi(
            date_column="date", close_column="close", periods=5, engine=engine
        )

        assert rising_result["close_rsi_5"].iloc[:4].isna().all()
        assert rising_result["close_rsi_5"].iloc[4:].eq(100.0).all()
        assert flat_result["close_rsi_5"].isna().all()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_rolling_risk_metrics_are_isolated_by_group(grouped_prices, engine):
    metrics = [
        "sharpe_ratio",
        "sortino_ratio",
        "omega_ratio",
        "volatility_annualized",
        "skewness",
        "kurtosis",
    ]
    grouped = grouped_prices.groupby("symbol").augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=8,
        metrics=metrics,
        engine=engine,
    )
    isolated = grouped_prices.loc[
        grouped_prices["symbol"].eq("B")
    ].augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=8,
        metrics=metrics,
        engine=engine,
    )

    columns = [f"close_{metric}_8" for metric in metrics]
    grouped_b = grouped.loc[grouped["symbol"].eq("B")].sort_values("date")
    _assert_columns_close(grouped_b, isolated.sort_values("date"), columns)


def test_rolling_risk_metrics_and_benchmark_metrics_match_engines(grouped_prices):
    metrics = [
        "sharpe_ratio",
        "sortino_ratio",
        "treynor_ratio",
        "information_ratio",
        "omega_ratio",
        "volatility_annualized",
        "skewness",
        "kurtosis",
    ]
    outputs = {}
    for engine in ["pandas", "polars"]:
        outputs[engine] = grouped_prices.groupby("symbol").augment_rolling_risk_metrics(
            date_column="date",
            close_column="close",
            benchmark_column="benchmark",
            window=8,
            metrics=metrics,
            engine=engine,
        )

    columns = [f"close_{metric}_8" for metric in metrics]
    _assert_columns_close(outputs["pandas"], outputs["polars"], columns)

    symbol_b = grouped_prices.loc[grouped_prices["symbol"].eq("B")].sort_values("date")
    returns = np.log(symbol_b["close"] / symbol_b["close"].shift(1))
    downside_deviation = np.sqrt(
        returns.clip(upper=0).pow(2).rolling(8, min_periods=4).mean()
    ).replace(0, np.nan)
    expected_sortino = (
        returns.rolling(8, min_periods=4).mean() / downside_deviation * np.sqrt(252)
    )
    actual_sortino = outputs["pandas"].loc[
        outputs["pandas"]["symbol"].eq("B"), "close_sortino_ratio_8"
    ]
    np.testing.assert_allclose(
        actual_sortino,
        expected_sortino,
        rtol=1e-10,
        atol=1e-10,
        equal_nan=True,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_sortino_is_undefined_without_downside_returns(grouped_prices, engine):
    window = 8
    prices = grouped_prices.loc[grouped_prices["symbol"].eq("A")].copy()
    returns = np.log(prices["close"] / prices["close"].shift(1))
    rolling_returns = returns.rolling(window, min_periods=window // 2)
    no_downside = rolling_returns.min().ge(0)

    assert no_downside.any()

    result = prices.augment_rolling_risk_metrics(
        date_column="date",
        close_column="close",
        window=window,
        metrics=["sortino_ratio"],
        engine=engine,
    )

    assert result.loc[no_downside, "close_sortino_ratio_8"].isna().all()


@pytest.mark.parametrize(
    "method,kwargs",
    [
        (
            "augment_atr",
            {
                "date_column": "date",
                "high_column": "high",
                "low_column": "low",
                "close_column": "close",
                "periods": 5,
            },
        ),
        (
            "augment_ewma_volatility",
            {"date_column": "date", "close_column": "close", "window": 5},
        ),
        (
            "augment_bbands",
            {"date_column": "date", "close_column": "close", "periods": 5},
        ),
        (
            "augment_hurst_exponent",
            {"date_column": "date", "close_column": "close", "window": 8},
        ),
        (
            "augment_rolling_risk_metrics",
            {
                "date_column": "date",
                "close_column": "close",
                "window": 8,
                "metrics": ["sharpe_ratio"],
            },
        ),
    ],
)
def test_pandas_finance_functions_support_multiple_grouping_columns(
    grouped_prices, method, kwargs
):
    single_group = getattr(grouped_prices.groupby("symbol"), method)(
        engine="pandas", **kwargs
    )
    multiple_groups = getattr(grouped_prices.groupby(["exchange", "symbol"]), method)(
        engine="pandas", **kwargs
    )

    result_columns = [
        column
        for column in single_group.columns
        if column not in grouped_prices.columns
    ]
    _assert_columns_close(single_group, multiple_groups, result_columns)

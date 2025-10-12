import pandas as pd
import polars as pl
import pytest

import pytimetk as tk
# noqa: F401


@pytest.fixture(scope="module")
def stocks_daily_df():
    return tk.load_dataset("stocks_daily", parse_dates=["date"])


@pytest.fixture(scope="module")
def stocks_subset(stocks_daily_df):
    symbols = ["AAPL", "MSFT", "GOOG"]
    filtered = stocks_daily_df[stocks_daily_df["symbol"].isin(symbols)].copy()
    return filtered.groupby("symbol").head(600).reset_index(drop=True)


def test_docstring_adx_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=[14, 28],
    )
    assert {"close_adx_14", "close_plus_di_14", "close_minus_di_14"}.issubset(
        pandas_result.columns
    )
    assert {"close_adx_28", "close_plus_di_28", "close_minus_di_28"}.issubset(
        pandas_result.columns
    )

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=14,
    )
    assert {"close_adx_14", "close_plus_di_14", "close_minus_di_14"}.issubset(
        polars_result.columns
    )


def test_docstring_atr_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_atr(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=[14, 28],
        normalize=False,
    )
    assert {"close_atr_14", "close_atr_28"}.issubset(pandas_result.columns)

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_atr(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=14,
        normalize=True,
    )
    assert "close_natr_14" in polars_result.columns


def test_docstring_bbands_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_bbands(
        date_column="date",
        close_column="close",
        periods=[20, 40],
        std_dev=[1.5, 2.0],
    )
    assert "close_bband_middle_20_1.5" in pandas_result.columns
    assert "close_bband_upper_40_2.0" in pandas_result.columns

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_bbands(
        date_column="date",
        close_column="close",
        periods=(10, 15),
        std_dev=2,
    )
    assert "close_bband_lower_15_2.0" in polars_result.columns


def test_docstring_cmo_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_cmo(
        date_column="date",
        close_column="close",
        periods=[14, 28],
    )
    assert {"close_cmo_14", "close_cmo_28"}.issubset(pandas_result.columns)

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_cmo(
        date_column="date",
        close_column="close",
        periods=14,
    )
    assert "close_cmo_14" in polars_result.columns


def test_docstring_drawdown_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_drawdown(
        date_column="date",
        close_column="close",
    )
    assert {"close_peak", "close_drawdown", "close_drawdown_pct"}.issubset(
        pandas_result.columns
    )

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_drawdown(
        date_column="date",
        close_column="close",
    )
    assert {"close_peak", "close_drawdown", "close_drawdown_pct"}.issubset(
        polars_result.columns
    )


def test_docstring_ewma_volatility_examples(stocks_subset):
    single_stock = stocks_subset.query("symbol == 'AAPL'")

    example_one = single_stock.augment_ewma_volatility(
        date_column="date",
        close_column="close",
        decay_factor=0.94,
        window=[20, 50],
    )
    assert {"close_ewma_vol_20_0.94", "close_ewma_vol_50_0.94"}.issubset(
        example_one.columns
    )

    example_two = stocks_subset.groupby("symbol").augment_ewma_volatility(
        date_column="date",
        close_column="close",
        decay_factor=0.94,
        window=[20, 50],
    )
    assert {"close_ewma_vol_20_0.94", "close_ewma_vol_50_0.94"}.issubset(
        example_two.columns
    )

    example_three = pl.from_pandas(single_stock).tk.augment_ewma_volatility(
        date_column="date",
        close_column="close",
        decay_factor=0.94,
        window=[20, 50],
    )
    assert "close_ewma_vol_20_0.94" in example_three.columns

    example_four = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_ewma_volatility(
            date_column="date",
            close_column="close",
            decay_factor=0.94,
            window=[20, 50],
        )
    )
    assert "close_ewma_vol_50_0.94" in example_four.columns


def test_docstring_fip_momentum_examples(stocks_subset):
    single_stock = stocks_subset.query("symbol == 'AAPL'")
    pandas_result = single_stock.augment_fip_momentum(
        date_column="date",
        close_column="close",
        window=252,
    )
    assert "close_fip_momentum_252" in pandas_result.columns

    polars_result = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_fip_momentum(
            date_column="date",
            close_column="close",
            window=[63, 252],
            fip_method="modified",
        )
    )
    assert {"close_fip_momentum_63", "close_fip_momentum_252"}.issubset(
        polars_result.columns
    )


def test_docstring_hurst_exponent_examples(stocks_subset):
    single_stock = stocks_subset.query("symbol == 'AAPL'")

    example_one = single_stock.augment_hurst_exponent(
        date_column="date",
        close_column="close",
        window=[100, 200],
    )
    assert {"close_hurst_100", "close_hurst_200"}.issubset(example_one.columns)

    example_two = stocks_subset.groupby("symbol").augment_hurst_exponent(
        date_column="date",
        close_column="close",
        window=100,
    )
    assert "close_hurst_100" in example_two.columns

    example_three = pl.from_pandas(single_stock).tk.augment_hurst_exponent(
        date_column="date",
        close_column="close",
        window=[100, 200],
    )
    assert "close_hurst_200" in example_three.columns

    example_four = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_hurst_exponent(
            date_column="date",
            close_column="close",
            window=100,
        )
    )
    assert "close_hurst_100" in example_four.columns


def test_docstring_macd_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_macd(
        date_column="date",
        close_column="close",
        fast_period=12,
        slow_period=26,
        signal_period=9,
        engine="pandas",
    )
    assert {
        "close_macd_line_12_26_9",
        "close_macd_signal_line_12_26_9",
        "close_macd_histogram_12_26_9",
    }.issubset(pandas_result.columns)

    polars_result = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_macd(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )
    )
    assert {
        "close_macd_line_12_26_9",
        "close_macd_signal_line_12_26_9",
        "close_macd_histogram_12_26_9",
    }.issubset(polars_result.columns)


def test_docstring_ppo_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_ppo(
        date_column="date",
        close_column="close",
        fast_period=12,
        slow_period=26,
    )
    assert "close_ppo_line_12_26" in pandas_result.columns

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_ppo(
        date_column="date",
        close_column="close",
        fast_period=12,
        slow_period=26,
    )
    assert "close_ppo_line_12_26" in polars_result.columns


def test_docstring_qsmomentum_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=[5, 21],
        roc_slow_period=252,
        returns_period=126,
    )
    assert "close_qsmom_5_252_126" in pandas_result.columns
    assert "close_qsmom_21_252_126" in pandas_result.columns

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_qsmomentum(
        date_column="date",
        close_column="close",
        roc_fast_period=[5, 21],
        roc_slow_period=252,
        returns_period=126,
    )
    assert "close_qsmom_21_252_126" in polars_result.columns


def test_docstring_regime_detection_examples(stocks_subset):
    pytest.importorskip(
        "hmmlearn.hmm", reason="Regime detection example requires hmmlearn"
    )

    single_stock = stocks_subset.query("symbol == 'AAPL'")

    example_one = single_stock.augment_regime_detection(
        date_column="date",
        close_column="close",
        window=252,
        n_regimes=2,
        n_iter=20,
        n_jobs=1,
        step_size=20,
    )
    assert "close_regime_252" in example_one.columns

    example_two = stocks_subset.groupby("symbol").augment_regime_detection(
        date_column="date",
        close_column="close",
        window=[252, 504],
        n_regimes=3,
        n_iter=20,
        n_jobs=1,
        step_size=20,
    )
    assert {"close_regime_252", "close_regime_504"}.issubset(example_two.columns)

    example_three = pl.from_pandas(single_stock).tk.augment_regime_detection(
        date_column="date",
        close_column="close",
        window=252,
        n_regimes=2,
        n_iter=20,
        n_jobs=1,
        step_size=20,
    )
    assert "close_regime_252" in example_three.columns

    example_four = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_regime_detection(
            date_column="date",
            close_column="close",
            window=504,
            n_regimes=3,
            n_iter=20,
            n_jobs=1,
            step_size=20,
        )
    )
    assert "close_regime_504" in example_four.columns


def test_docstring_roc_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_roc(
        date_column="date",
        close_column="close",
        periods=[22, 63],
        start_index=5,
    )
    assert "close_roc_5_22" in pandas_result.columns
    assert "close_roc_5_63" in pandas_result.columns

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_roc(
        date_column="date",
        close_column="close",
        periods=(5, 10),
    )
    assert "close_roc_0_5" in polars_result.columns
    assert "close_roc_0_10" in polars_result.columns


def test_docstring_rolling_risk_metrics_examples(stocks_subset):
    single_stock = stocks_subset.query("symbol == 'AAPL'")

    example_one = single_stock.augment_rolling_risk_metrics(
        date_column="date",
        close_column="adjusted",
        window=252,
    )
    assert "adjusted_sharpe_ratio_252" in example_one.columns

    example_two = (
        pl.from_pandas(stocks_subset)
        .group_by("symbol")
        .tk.augment_rolling_risk_metrics(
            date_column="date",
            close_column="adjusted",
            window=60,
        )
    )
    assert "adjusted_sharpe_ratio_60" in example_two.columns

    example_three = stocks_subset.groupby("symbol").augment_rolling_risk_metrics(
        date_column="date",
        close_column="adjusted",
        window=252,
        metrics=["sharpe_ratio", "sortino_ratio", "volatility_annualized"],
    )
    assert {
        "adjusted_sharpe_ratio_252",
        "adjusted_sortino_ratio_252",
        "adjusted_volatility_annualized_252",
    }.issubset(example_three.columns)


def test_docstring_rsi_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_rsi(
        date_column="date",
        close_column="close",
        periods=[14, 28],
    )
    assert {"close_rsi_14", "close_rsi_28"}.issubset(pandas_result.columns)

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_rsi(
        date_column="date",
        close_column=["close"],
        periods=14,
    )
    assert "close_rsi_14" in polars_result.columns


def test_docstring_stochastic_oscillator_examples(stocks_subset):
    pandas_result = stocks_subset.groupby("symbol").augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[14, 21],
        d_periods=[3, 9],
    )
    assert "close_stoch_k_14" in pandas_result.columns
    assert "close_stoch_d_21_9" in pandas_result.columns

    polars_result = pl.from_pandas(
        stocks_subset.query("symbol == 'AAPL'")
    ).tk.augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=14,
        d_periods=[3],
    )
    assert "close_stoch_k_14" in polars_result.columns

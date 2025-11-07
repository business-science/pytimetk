import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pytimetk as tk


def _regression_df():
    rng = pd.date_range("2022-01-01", periods=60, freq="D")
    trend = np.arange(len(rng))
    noise = np.random.default_rng(123).normal(scale=2.0, size=len(rng))
    df = pd.DataFrame(
        {
            "date": rng,
            "trend": trend,
            "value": 0.8 * trend + noise,
            "segment": np.where(trend < 30, "A", "B"),
        }
    )
    return df


def test_plot_time_series_regression_basic():
    df = _regression_df()
    fig = tk.plot_time_series_regression(
        data=df,
        date_column="date",
        formula="value ~ trend",
        title="Basic Regression",
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_plot_time_series_regression_grouped():
    df = _regression_df()
    fig = tk.plot_time_series_regression(
        data=df.groupby("segment"),
        date_column="date",
        formula="value ~ trend",
        facet_ncol=1,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plot_time_series_regression_polars(engine):
    df = _regression_df()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(df)
    fig = tk.plot_time_series_regression(
        data=df,
        date_column="date",
        formula="value ~ trend",
    )
    assert isinstance(fig, go.Figure)

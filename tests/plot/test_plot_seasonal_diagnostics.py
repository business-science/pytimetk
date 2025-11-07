import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pytimetk as tk
from pytimetk.utils.selection import contains


def _sample_df():
    rng = pd.date_range("2020-01-01", periods=48, freq="H")
    df = pd.DataFrame(
        {
            "id": ["A"] * 24 + ["B"] * 24,
            "date": list(rng[:24]) + list(rng[:24]),
            "value": np.random.default_rng(123).normal(size=48),
        }
    )
    return df


def test_plot_seasonal_diagnostics_basic_box():
    df = _sample_df()
    fig = tk.plot_seasonal_diagnostics(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        feature_set=["hour", "wday.lbl"],
        geom="box",
    )
    assert isinstance(fig, go.Figure)


def test_plot_seasonal_diagnostics_tidy_selectors_violin():
    df = _sample_df()
    fig = tk.plot_seasonal_diagnostics(
        data=df,
        date_column="date",
        value_column=contains("val"),
        feature_set="auto",
        facet_vars="id",
        geom="violin",
        plotly_dropdown=True,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plot_seasonal_diagnostics_polars(engine):
    df = _sample_df()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        pl_df = pl.from_pandas(df)
        fig = tk.plot_seasonal_diagnostics(
            data=pl_df,
            date_column="date",
            value_column="value",
            feature_set="auto",
        )
        assert isinstance(fig, go.Figure)
    else:
        fig = tk.plot_seasonal_diagnostics(
            data=df,
            date_column="date",
            value_column="value",
            feature_set=["hour"],
        )
        assert isinstance(fig, go.Figure)

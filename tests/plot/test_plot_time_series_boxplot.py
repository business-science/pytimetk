import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pytimetk as tk
from pytimetk.utils.selection import contains


def _sample_df():
    rng = pd.date_range("2021-01-01", periods=240, freq="H")
    df = pd.DataFrame(
        {
            "id": np.where(np.arange(len(rng)) % 2 == 0, "A", "B"),
            "date": rng,
            "value": np.random.default_rng(123).normal(size=len(rng)),
            "segment": np.where(np.arange(len(rng)) % 3 == 0, "X", "Y"),
        }
    )
    return df


def test_plot_time_series_boxplot_basic():
    df = _sample_df()
    fig = tk.plot_time_series_boxplot(
        data=df,
        date_column="date",
        value_column="value",
        period="1 day",
        facet_vars="id",
        smooth=False,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_time_series_boxplot_tidy_dropdown():
    df = _sample_df()
    fig = tk.plot_time_series_boxplot(
        data=df.groupby("segment"),
        date_column="date",
        value_column=contains("val"),
        period="24 hours",
        color_column="id",
        plotly_dropdown=True,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plot_time_series_boxplot_polars(engine):
    df = _sample_df()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        pl_df = pl.from_pandas(df)
        fig = tk.plot_time_series_boxplot(
            data=pl_df,
            date_column="date",
            value_column="value",
            period="12 hours",
        )
    else:
        fig = tk.plot_time_series_boxplot(
            data=df,
            date_column="date",
            value_column="value",
            period="1 day",
        )
    assert isinstance(fig, go.Figure)

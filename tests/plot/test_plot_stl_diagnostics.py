import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import pytimetk as tk
from pytimetk.utils.selection import contains


def _sample_df():
    rng = pd.date_range("2020-01-01", periods=120, freq="D")
    id_series = np.where(np.arange(len(rng)) < len(rng) / 2, "A", "B")
    values = np.sin(np.linspace(0, 8 * np.pi, len(rng))) + np.random.default_rng(123).normal(
        scale=0.2, size=len(rng)
    )
    df = pd.DataFrame({"id": id_series, "date": rng, "value": values})
    return df


def test_plot_stl_diagnostics_basic():
    df = _sample_df()
    fig = tk.plot_stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        feature_set=["observed", "trend"],
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_plot_stl_diagnostics_tidy_dropdown():
    df = _sample_df()
    df["grp"] = df["id"]
    fig = tk.plot_stl_diagnostics(
        data=df,
        date_column="date",
        value_column=contains("value"),
        facet_vars="grp",
        feature_set=["observed", "season", "remainder"],
        plotly_dropdown=True,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plot_stl_diagnostics_polars(engine):
    df = _sample_df()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        df = pl.from_pandas(df)
    fig = tk.plot_stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        feature_set="observed",
    )
    assert isinstance(fig, go.Figure)

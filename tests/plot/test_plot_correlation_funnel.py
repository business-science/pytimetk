import pandas as pd
import polars as pl

# noqa: F401
import plotly.graph_objects as go
from plotnine import ggplot

from pytimetk.plot import plot_correlation_funnel


def _sample_correlation_frame():
    return pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "bin": ["bin_a", "bin_b", "bin_c"],
            "correlation": [0.6, -0.2, 0.1],
        }
    )


def test_plot_correlation_funnel_plotly():
    df = _sample_correlation_frame()
    fig = plot_correlation_funnel(df, engine="plotly")
    assert isinstance(fig, go.Figure)


def test_plot_correlation_funnel_plotnine():
    df = _sample_correlation_frame()
    fig = plot_correlation_funnel(df, engine="plotnine")
    assert isinstance(fig, ggplot)


def test_plot_correlation_funnel_polars_accessor():
    df = _sample_correlation_frame()
    pl_df = pl.from_pandas(df)

    fig_plotly = pl_df.tk.plot_correlation_funnel(engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = pl_df.tk.plot_correlation_funnel(engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)

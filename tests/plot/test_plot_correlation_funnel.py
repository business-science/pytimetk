import pytest

try:
    import polars as pl
except ImportError:
    pass

from pytimetk.plot import plot_correlation_funnel

@pytest.fixture
def df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "bin": ["bin_a", "bin_b", "bin_c"],
            "correlation": [0.6, -0.2, 0.1],
        }
    )


def test_plot_correlation_funnel_plotly(df):
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    fig = plot_correlation_funnel(df, engine="plotly")
    assert isinstance(fig, go.Figure)


def test_plot_correlation_funnel_plotnine(df):
    pytest.importorskip("plotnine")
    from plotnine import ggplot

    fig = plot_correlation_funnel(df, engine="plotnine")
    assert isinstance(fig, ggplot)


def test_plot_correlation_funnel_polars_accessor_plotly(df):
    pytest.importorskip("plotly")
    pl = pytest.importorskip("polars")
    import plotly.graph_objects as go

    pl_df = pl.from_pandas(df)

    fig_plotly = plot_correlation_funnel(pl_df, engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

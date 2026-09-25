import pandas as pd
import polars as pl
from matplotlib.collections import LineCollection

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


def test_plot_correlation_funnel_plotnine_dense_labels_do_not_overlap():
    df = pd.DataFrame(
        {
            "feature": ["City"] * 5 + ["Age"] * 4,
            "bin": [
                "New York",
                "Los Angeles",
                "Chicago",
                "Miami",
                "Houston",
                "18_29",
                "29_39",
                "39_53",
                "53_64",
            ],
            "correlation": [-0.04, -0.02, 0, 0.02, 0.04, -0.03, -0.01, 0.01, 0.03],
        }
    )
    original = df.copy(deep=True)

    figure = plot_correlation_funnel(df, engine="plotnine", height=500).draw()
    renderer = figure.canvas.get_renderer()
    labels = [
        text for text in figure.axes[0].texts if text.get_text() in set(df["bin"])
    ]
    boxes = [label.get_window_extent(renderer) for label in labels]
    connectors = [
        collection
        for collection in figure.axes[0].collections
        if isinstance(collection, LineCollection)
    ]

    assert len(labels) == len(df)
    assert all(
        not box.overlaps(other)
        for index, box in enumerate(boxes)
        for other in boxes[index + 1 :]
    )
    assert any(len(collection.get_segments()) == len(df) for collection in connectors)
    pd.testing.assert_frame_equal(df, original)


def test_plot_correlation_funnel_polars_accessor():
    df = _sample_correlation_frame()
    pl_df = pl.from_pandas(df)

    fig_plotly = pl_df.tk.plot_correlation_funnel(engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = pl_df.tk.plot_correlation_funnel(engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)

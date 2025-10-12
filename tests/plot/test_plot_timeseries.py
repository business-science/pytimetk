import pytest
import pandas as pd
import numpy as np
import polars as pl
import pytimetk

# noqa: F401
import plotly


# Prepare a sample dataframe for testing
data = pd.DataFrame(
    {
        "date": pd.date_range(start="1/1/2020", periods=5, freq="D"),
        "value": np.random.randn(5),
        "id": ["A", "A", "B", "B", "B"],
    }
)


def test_plotly_engine():
    fig = data.plot_timeseries("date", "value", engine="plotly")
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"


data.groupby("id").plot_timeseries(
    "date",
    "value",
    color_column="id",
    facet_ncol=2,
    x_axis_date_labels="%Y",
    engine="matplotlib",
)


def test_matplotlib_engine():
    fig = data.plot_timeseries(
        "date", "value", engine="matplotlib", width=1200, height=800
    )
    assert type(fig).__name__ == "Figure", "Expected a matplotlib Figure object"


def test_plotnine_engine():
    # Since I don't have the complete function, I'm assuming the function returns a ggplot object for plotnine
    fig = data.plot_timeseries("date", "value", engine="plotnine")
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"


# Test for groupby functionality
def test_groupby():
    fig = data.groupby("id").plot_timeseries("date", "value", engine="plotly")
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"


def test_matplotlib_groupby():
    fig = data.groupby("id").plot_timeseries(
        "date", "value", engine="matplotlib", width=1200, height=800
    )
    assert type(fig).__name__ == "Figure", "Expected a matplotlib_ Figure object"


def test_plotnine_groupby():
    fig = data.groupby("id").plot_timeseries("date", "value", engine="plotnine")
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"


# Test for smooth functionality
def test_smooth():
    fig = data.plot_timeseries("date", "value", smooth=True, engine="plotly")
    assert type(fig).__name__ == "Figure", (
        "Expected a plotly Figure object with smoothing"
    )


# Test for Handling GroupBy objects
def test_groupby_handling():
    group = data.groupby("id")
    fig = group.plot_timeseries(
        date_column="date", value_column="value", engine="plotly"
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure), (
        "Figure type doesn't match expected type"
    )


def test_plot_timeseries_polars_accessor_plotly():
    pl_df = pl.from_pandas(data)
    fig = pl_df.tk.plot_timeseries(
        date_column="date",
        value_column="value",
        engine="plotly",
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_plot_timeseries_polars_accessor_plotnine():
    pl_df = pl.from_pandas(data)
    fig = pl_df.tk.plot_timeseries(
        date_column="date",
        value_column="value",
        engine="plotnine",
    )
    assert str(type(fig)).endswith("ggplot'>")


def test_plot_timeseries_groupby_polars_accessor():
    pl_df = pl.from_pandas(data)
    fig = pl_df.group_by("id").tk.plot_timeseries(
        date_column="date",
        value_column="value",
        engine="plotly",
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


# Additional tests can be added based on other functionalities or edge cases

if __name__ == "__main__":
    pytest.main([__file__])

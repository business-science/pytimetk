import pytest

from pytimetk.utils.selection import contains
from pytimetk.plot import plot_timeseries


@pytest.fixture
def data():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    return pd.DataFrame(
        {
            "date": pd.date_range(start="1/1/2020", periods=5, freq="D"),
            "value": np.random.randn(5),
            "id": ["A", "A", "B", "B", "B"],
        }
    )


def test_plotly_engine(data):
    pytest.importorskip("plotly")
    fig = plot_timeseries(data, "date", "value", engine="plotly")
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"


def test_matplotlib_engine(data):
    pytest.importorskip("matplotlib")
    pytest.importorskip("plotnine")
    fig = plot_timeseries(
        data, "date", "value", engine="matplotlib", width=1200, height=800
    )
    assert type(fig).__name__ == "Figure", "Expected a matplotlib Figure object"


def test_plotnine_engine(data):
    pytest.importorskip("plotnine")
    fig = plot_timeseries(data, "date", "value", engine="plotnine")
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"


# Test for groupby functionality
def test_groupby(data):
    pytest.importorskip("plotly")
    fig = plot_timeseries(data.groupby("id"), "date", "value", engine="plotly")
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"


def test_matplotlib_groupby(data):
    pytest.importorskip("matplotlib")
    pytest.importorskip("plotnine")
    fig = plot_timeseries(
        data.groupby("id"), "date", "value", engine="matplotlib", width=1200, height=800
    )
    assert type(fig).__name__ == "Figure", "Expected a matplotlib_ Figure object"


def test_plotnine_groupby(data):
    pytest.importorskip("plotnine")
    fig = plot_timeseries(data.groupby("id"), "date", "value", engine="plotnine")
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"


# Test for smooth functionality
def test_smooth(data):
    pytest.importorskip("plotly")
    fig = plot_timeseries(data, "date", "value", smooth=True, engine="plotly")
    assert type(fig).__name__ == "Figure", (
        "Expected a plotly Figure object with smoothing"
    )


# Test for Handling GroupBy objects
def test_groupby_handling(data):
    pytest.importorskip("plotly")
    import plotly

    group = data.groupby("id")
    fig = plot_timeseries(
        group, date_column="date", value_column="value", engine="plotly"
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure), (
        "Figure type doesn't match expected type"
    )


def test_plot_timeseries_polars_accessor_plotly(data):
    pytest.importorskip("plotly")
    pl = pytest.importorskip("polars")
    import plotly

    pl_df = pl.from_pandas(data)
    fig = plot_timeseries(
        pl_df,
        date_column="date",
        value_column="value",
        engine="plotly",
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_plot_timeseries_polars_accessor_plotnine(data):
    pytest.importorskip("plotnine")
    pl = pytest.importorskip("polars")
    pl_df = pl.from_pandas(data)
    fig = plot_timeseries(
        pl_df,
        date_column="date",
        value_column="value",
        engine="plotnine",
    )
    assert str(type(fig)).endswith("ggplot'>")


def test_plot_timeseries_groupby_polars_accessor(data):
    pytest.importorskip("plotly")
    pl = pytest.importorskip("polars")
    import plotly

    pl_df = pl.from_pandas(data)
    fig = plot_timeseries(
        pl_df.group_by("id"),
        date_column="date",
        value_column="value",
        engine="plotly",
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


def test_plot_timeseries_tidy_selectors(data):
    pytest.importorskip("plotly")
    import plotly

    fig = plot_timeseries(
        data,
        date_column="date",
        value_column=contains("val"),
        color_column=contains("id"),
        engine="plotly",
    )
    assert isinstance(fig, plotly.graph_objs._figure.Figure)
    assert isinstance(fig, plotly.graph_objs._figure.Figure)


# Additional tests can be added based on other functionalities or edge cases

if __name__ == "__main__":
    pytest.main([__file__])

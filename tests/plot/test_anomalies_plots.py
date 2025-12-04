import pytest

try:
    import polars as pl
except ImportError:
    pass
from pytimetk.datasets.get_datasets import load_dataset
from pytimetk.plot import (
    plot_anomalies,
    plot_anomalies_decomp,
    plot_anomalies_cleaned,
)
from pytimetk.core import anomalize


def _sample_anomalize_frame():
    df = load_dataset("walmart_sales_weekly", parse_dates=["Date"])[
        ["id", "Date", "Weekly_Sales"]
    ]
    target_id = df["id"].iloc[0]
    df_filtered = df[df["id"] == target_id]
    anomalize_df = anomalize(
        df_filtered,
        date_column="Date",
        value_column="Weekly_Sales",
        method="stl",
        iqr_alpha=0.025,
        clean="min_max",
        show_progress=False,
    )
    return anomalize_df


def test_plot_anomalies_plotly():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    data = _sample_anomalize_frame()

    fig_plotly = plot_anomalies(data, date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)


def test_plot_anomalies_plotnine():
    pytest.importorskip("plotnine")
    from plotnine import ggplot

    data = _sample_anomalize_frame()

    fig_plotnine = plot_anomalies(data, date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_decomp_plotly():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    data = _sample_anomalize_frame()

    fig_plotly = plot_anomalies_decomp(data, date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)


def test_plot_anomalies_decomp_plotnine():
    pytest.importorskip("plotnine")
    from plotnine import ggplot

    data = _sample_anomalize_frame()

    fig_plotnine = plot_anomalies_decomp(data, date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_cleaned_plotly():
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    data = _sample_anomalize_frame()

    fig_plotly = plot_anomalies_cleaned(data, date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)


def test_plot_anomalies_cleaned_plotnine():
    pytest.importorskip("plotnine")
    from plotnine import ggplot

    data = _sample_anomalize_frame()

    fig_plotnine = plot_anomalies_cleaned(data, date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_polars_accessor_plotly():
    pytest.importorskip("plotly")
    pytest.importorskip("polars")
    import plotly.graph_objects as go

    data = _sample_anomalize_frame()
    pl_df = pl.from_pandas(data)

    fig_plotly = plot_anomalies(pl_df, date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)


def test_plot_anomalies_polars_accessor_plotnine():
    pytest.importorskip("plotnine")
    pytest.importorskip("polars")
    from plotnine import ggplot

    data = _sample_anomalize_frame()
    pl_df = pl.from_pandas(data)

    fig_plotnine = plot_anomalies(pl_df, date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_groupby_polars_accessor_plotly():
    pytest.importorskip("plotly")
    pytest.importorskip("polars")
    import plotly.graph_objects as go

    data = _sample_anomalize_frame().copy()
    data["group"] = "grp1"
    pl_df = pl.from_pandas(data)

    fig_plotly = plot_anomalies(
        pl_df.group_by("group"), date_column="Date", engine="plotly"
    )
    assert isinstance(fig_plotly, go.Figure)


def test_plot_anomalies_groupby_polars_accessor_plotnine():
    pytest.importorskip("plotnine")
    pytest.importorskip("polars")
    from plotnine import ggplot

    data = _sample_anomalize_frame().copy()
    data["group"] = "grp1"
    pl_df = pl.from_pandas(data)

    fig_plotnine = plot_anomalies(
        pl_df.group_by("group"), date_column="Date", engine="plotnine"
    )
    assert isinstance(fig_plotnine, ggplot)

import polars as pl
import pytimetk as tk

# noqa: F401
import plotly.graph_objects as go
from plotnine import ggplot


def _sample_anomalize_frame():
    df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[
        ["id", "Date", "Weekly_Sales"]
    ]
    target_id = df["id"].iloc[0]
    df_filtered = df[df["id"] == target_id]
    anomalize_df = df_filtered.anomalize(
        date_column="Date",
        value_column="Weekly_Sales",
        method="stl",
        iqr_alpha=0.025,
        clean="min_max",
        show_progress=False,
    )
    return anomalize_df


def test_plot_anomalies_engines():
    data = _sample_anomalize_frame()

    fig_plotly = data.plot_anomalies(date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = data.plot_anomalies(date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_decomp_engines():
    data = _sample_anomalize_frame()

    fig_plotly = data.plot_anomalies_decomp(date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = data.plot_anomalies_decomp(date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_cleaned_engines():
    data = _sample_anomalize_frame()

    fig_plotly = data.plot_anomalies_cleaned(date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = data.plot_anomalies_cleaned(date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_polars_accessor():
    data = _sample_anomalize_frame()
    pl_df = pl.from_pandas(data)

    fig_plotly = pl_df.tk.plot_anomalies(date_column="Date", engine="plotly")
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = pl_df.tk.plot_anomalies(date_column="Date", engine="plotnine")
    assert isinstance(fig_plotnine, ggplot)


def test_plot_anomalies_groupby_polars_accessor():
    data = _sample_anomalize_frame().copy()
    data["group"] = "grp1"
    pl_df = pl.from_pandas(data)

    fig_plotly = pl_df.group_by("group").tk.plot_anomalies(
        date_column="Date", engine="plotly"
    )
    assert isinstance(fig_plotly, go.Figure)

    fig_plotnine = pl_df.group_by("group").tk.plot_anomalies(
        date_column="Date", engine="plotnine"
    )
    assert isinstance(fig_plotnine, ggplot)

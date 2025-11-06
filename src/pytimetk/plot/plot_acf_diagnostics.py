import math
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytimetk.core.acf_diagnostics import acf_diagnostics


@pf.register_groupby_method
@pf.register_dataframe_method
def plot_acf_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: str,
    ccf_columns: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    lags: Union[str, int, Sequence[int], np.ndarray, range, slice] = 1000,
    show_white_noise: bool = True,
    title: str = "Lag Diagnostics",
    x_lab: str = "Lag",
    y_lab: str = "Correlation",
    width: Optional[int] = None,
    height: Optional[int] = None,
    # Group faceting
    group_ncols: Optional[int] = None,
    group_label_sep: str = ", ",
    # Line/marker styling
    line_color: str = "#2c3e50",
    line_width: float = 2.0,
    line_dash: str = "solid",
    marker_color: Optional[str] = None,
    marker_size: float = 6.0,
    marker_symbol: str = "circle",
    marker_line_color: Optional[str] = None,
    marker_line_width: float = 0.0,
    show_markers: bool = True,
    show_legend: bool = False,
    # Reference lines
    zero_line_color: str = "#999999",
    zero_line_width: float = 1.0,
    zero_line_dash: str = "dash",
    white_noise_color: str = "#A6CEE3",
    white_noise_dash: str = "dot",
    white_noise_width: float = 1.0,
    # Hover
    hovertemplate: Optional[str] = None,
):
    """
    Visualise ACF, PACF, and optional CCF diagnostics with Plotly, including
    grouped facets and extensive styling controls.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Time series data or grouped data produced via ``DataFrame.groupby``.
    date_column : str
        Name of the datetime column.
    value_column : str
        Numeric column to diagnose.
    ccf_columns : str or sequence, optional
        Additional numeric columns whose cross-correlations should be drawn.
        Accepts literal names or tidy selectors from :mod:`pytimetk.utils.selection`.
    lags : int, sequence, slice, or str, optional
        Lag specification forwarded to :func:`pytimetk.acf_diagnostics`. Defaults to ``1000``.
    show_white_noise : bool, optional
        Draw white-noise confidence bands around each subplot. Defaults to ``True``.
    title : str, optional
        Figure title. Defaults to ``"Lag Diagnostics"``.
    x_lab, y_lab : str, optional
        Axis labels for the bottom row / first column.
    width, height : int, optional
        Figure dimensions in pixels.
    group_ncols : int, optional
        Number of facet columns when grouped data are supplied. Defaults to a single row.
    group_label_sep : str, optional
        Separator used when concatenating multiple group labels for facet titles.
    line_color : str, optional
        Colour of the ACF/PACF/CCF traces.
    line_width : float, optional
        Line width in pixels. Defaults to ``2.0``.
    line_dash : str, optional
        Dash style for correlation lines (``"solid"``, ``"dash"``, etc.).
    marker_color : str, optional
        Colour for scatter markers. Defaults to ``line_color``.
    marker_size : float, optional
        Marker size in pixels. Defaults to ``6.0``.
    marker_symbol : str, optional
        Plotly marker symbol (``"circle"``, ``"square"``, ...).
    marker_line_color : str, optional
        Outline colour for markers. Defaults to ``marker_color``.
    marker_line_width : float, optional
        Outline width for markers. Defaults to ``0.0``.
    show_markers : bool, optional
        Toggle markers on/off. Defaults to ``True``.
    show_legend : bool, optional
        Display a legend for the metric traces. Defaults to ``False``.
    zero_line_color, zero_line_width, zero_line_dash : optional
        Styling for the horizontal zero reference line.
    white_noise_color, white_noise_dash, white_noise_width : optional
        Styling for the white-noise confidence bands.
    hovertemplate : str, optional
        Custom Plotly hover template. Defaults to ``"Lag=%{x}<br>Correlation=%{y}<extra></extra>"``.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with one subplot per diagnostic metric (ACF, PACF,
        CCF traces) and optional facets for grouped series.

    Examples
    --------
    ```{python}
    import numpy as np
    import pandas as pd
    import pytimetk as tk

    rng = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "id": ["A"] * 20 + ["B"] * 20,
            "date": list(rng[:20]) + list(rng[:20]),
            "value": np.sin(np.linspace(0, 4 * np.pi, 40)),
            "driver": np.cos(np.linspace(0, 4 * np.pi, 40)),
        }
    )

    fig = tk.plot_acf_diagnostics(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        ccf_columns="driver",
        lags=15,
        group_ncols=1,
        show_legend=True,
    )
    fig
    ```
    """

    diagnostics = acf_diagnostics(
        data=data,
        date_column=date_column,
        value_column=value_column,
        ccf_columns=ccf_columns,
        lags=lags,
    )

    expected_columns = [
        "metric",
        "lag",
        "value",
        "white_noise_upper",
        "white_noise_lower",
    ]
    group_columns = [col for col in diagnostics.columns if col not in expected_columns]

    # Ordering for metrics keeps ACF/PACF first, remaining metrics alphabetical.
    unique_metrics = diagnostics["metric"].unique().tolist()
    metric_priority = {"ACF": 0, "PACF": 1}
    order_lookup = {name: idx for idx, name in enumerate(unique_metrics)}
    ordered_metrics = sorted(
        unique_metrics,
        key=lambda name: (metric_priority.get(name, 100), order_lookup[name]),
    )
    if len(ordered_metrics) == 0:
        raise ValueError("No diagnostics computed; check your input data.")

    # Build facet layout
    if group_columns:
        group_frame = (
            diagnostics[group_columns].drop_duplicates().reset_index(drop=True)
        )
        group_labels = group_frame.apply(
            lambda row: group_label_sep.join(
                f"{col}={row[col]}" for col in group_columns
            ),
            axis=1,
        ).tolist()
    else:
        group_frame = pd.DataFrame({"__dummy__": [0]})
        group_labels = [""]

    n_groups = len(group_labels)
    if n_groups == 0:
        group_frame = pd.DataFrame({"__dummy__": [0]})
        group_labels = [""]
        n_groups = 1

    if group_ncols is None or group_ncols <= 0:
        ncols = n_groups
    else:
        ncols = max(1, min(group_ncols, n_groups))

    group_rows = math.ceil(n_groups / ncols)
    rows = group_rows * len(ordered_metrics)

    subplot_titles = []
    for group_row_idx in range(group_rows):
        for metric in ordered_metrics:
            for group_col_idx in range(ncols):
                global_group_idx = group_row_idx * ncols + group_col_idx
                if global_group_idx >= n_groups:
                    subplot_titles.append("")
                else:
                    label = group_labels[global_group_idx]
                    subplot_titles.append(
                        metric if label == "" else f"{label}<br>{metric}"
                    )

    marker_color = marker_color or line_color
    mode = "lines+markers" if show_markers else "lines"
    hovertemplate = hovertemplate or "Lag=%{x}<br>Correlation=%{y}<extra></extra>"

    fig = make_subplots(
        rows=rows,
        cols=ncols,
        shared_xaxes=True,
        vertical_spacing=0.04,
        horizontal_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    for group_idx in range(n_groups):
        group_row_idx = group_idx // ncols
        group_col_idx = group_idx % ncols
        group_mask = pd.Series(True, index=diagnostics.index)
        if group_columns:
            row_values = group_frame.iloc[group_idx]
            for col in group_columns:
                group_mask &= diagnostics[col] == row_values[col]

        for metric_idx, metric in enumerate(ordered_metrics):
            row = group_row_idx * len(ordered_metrics) + metric_idx + 1
            col = group_col_idx + 1
            subset = diagnostics.loc[
                group_mask & (diagnostics["metric"] == metric)
            ].sort_values("lag")
            if subset.empty:
                continue

            x = subset["lag"]
            y = subset["value"]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    marker=dict(
                        color=marker_color,
                        size=marker_size,
                        symbol=marker_symbol,
                        line=dict(
                            color=marker_line_color or marker_color,
                            width=marker_line_width,
                        ),
                    ),
                    name=metric,
                    showlegend=show_legend and group_idx == 0 and metric_idx == 0,
                    hovertemplate=hovertemplate,
                ),
                row=row,
                col=col,
            )

            lag_min = x.min()
            lag_max = x.max()

            # Zero reference line
            fig.add_trace(
                go.Scatter(
                    x=[lag_min, lag_max],
                    y=[0, 0],
                    mode="lines",
                    line=dict(
                        color=zero_line_color,
                        width=zero_line_width,
                        dash=zero_line_dash,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            if show_white_noise:
                upper = subset["white_noise_upper"].iloc[0]
                lower = subset["white_noise_lower"].iloc[0]

                fig.add_trace(
                    go.Scatter(
                        x=[lag_min, lag_max],
                        y=[upper, upper],
                        mode="lines",
                        line=dict(
                            color=white_noise_color,
                            dash=white_noise_dash,
                            width=white_noise_width,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[lag_min, lag_max],
                        y=[lower, lower],
                        mode="lines",
                        line=dict(
                            color=white_noise_color,
                            dash=white_noise_dash,
                            width=white_noise_width,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            # Apply axis labels
            if col == 1:
                fig.update_yaxes(
                    title_text=y_lab if metric_idx == 0 else "", row=row, col=col
                )
            else:
                fig.update_yaxes(title_text="", row=row, col=col)

    # Shared x-axis labels for bottom metric block
    for group_row_idx in range(group_rows):
        base_row = (group_row_idx + 1) * len(ordered_metrics)
        for col in range(1, ncols + 1):
            fig.update_xaxes(title_text=x_lab, row=base_row, col=col)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        margin=dict(l=60, r=20, t=60, b=50),
        width=width,
        height=height,
    )

    return fig

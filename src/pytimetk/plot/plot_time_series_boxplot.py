from __future__ import annotations
import math
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
from pytimetk.utils import pandas_flavor_compat as pf

from pytimetk.utils.requirements import require_plotly

if TYPE_CHECKING:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:  # Optional dependency for seamless polars support
    import polars as pl
except ImportError:  # pragma: no cover - optional import
    pl = None

from pytimetk.plot.theme import palette_timetk
from pytimetk.utils.datetime_helpers import floor_date, parse_human_duration
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame
from pytimetk.utils.plot_helpers import hex_to_rgba, name_to_hex


def _canonical_freqstr(freqstr: str) -> str:
    freqstr = freqstr.strip()
    if not freqstr:
        raise ValueError("Unable to resolve a frequency string for `period`.")
    freq_upper = freqstr.upper()
    if freq_upper.endswith("MIN"):
        return freq_upper[:-3] + "min"
    return freq_upper


def _normalise_period_spec(
    period: Union[str, pd.DateOffset, pd.Timedelta, np.timedelta64, timedelta],
) -> str:
    """
    Convert period inputs (e.g. ``\"30 minutes\"``) to pandas frequency strings.
    """

    def _dateoffset_to_str(offset: pd.DateOffset) -> str:
        freqstr = getattr(offset, "freqstr", None)
        if freqstr and not freqstr.startswith("<"):
            return freqstr

        kwds = getattr(offset, "kwds", None) or {}
        years = kwds.get("years")
        months = kwds.get("months")
        if years:
            return f"{int(years)}Y"
        if months:
            return f"{int(months)}M"

        raise ValueError(
            "Unsupported DateOffset specification for `period`. "
            "Provide a pandas frequency string (e.g. '7D', '1M') or a human-readable duration."
        )

    if isinstance(period, str):
        text = period.strip()
        if not text:
            raise ValueError("`period` cannot be an empty string.")
        try:
            offset = pd.tseries.frequencies.to_offset(text)
            if isinstance(offset, pd.DateOffset):
                try:
                    return _canonical_freqstr(_dateoffset_to_str(offset))
                except ValueError:
                    pass
            freqstr = getattr(offset, "freqstr", None)
            if freqstr and not freqstr.startswith("<"):
                return _canonical_freqstr(freqstr)
        except (ValueError, TypeError):
            parsed = parse_human_duration(text)
            return _normalise_period_spec(parsed)

    if isinstance(period, pd.DateOffset):
        return _canonical_freqstr(_dateoffset_to_str(period))

    if isinstance(period, (pd.Timedelta, np.timedelta64, timedelta)):
        delta = pd.to_timedelta(period)
        offset = pd.tseries.frequencies.to_offset(delta)
        freqstr = getattr(offset, "freqstr", None)
        if freqstr and not freqstr.startswith("<"):
            return _canonical_freqstr(freqstr)
        raise ValueError(
            "Timedelta-based `period` values must correspond to fixed-width offsets (seconds through weeks)."
        )

    raise TypeError(
        "`period` must be a pandas frequency string, human-readable duration, "
        "Timedelta/DateOffset, or datetime.timedelta."
    )


def _resolve_single_selector(
    selector: Union[str, ColumnSelector],
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    label: str,
) -> str:
    if isinstance(selector, str):
        return selector
    resolved = resolve_column_selection(
        data, selector, allow_none=False, require_match=True
    )
    if len(resolved) != 1:
        raise ValueError(f"`{label}` must resolve to exactly one column.")
    return resolved[0]


def _build_color_mapping(
    categories: Sequence[str],
    palette: Optional[Union[Dict[str, str], Sequence[str], str]],
) -> Dict[str, str]:
    timetk_colors = list(palette_timetk().values())
    if palette is None:
        base_cycle = timetk_colors
    elif isinstance(palette, str):
        if palette.lower() == "timetk":
            base_cycle = timetk_colors
        else:
            base_cycle = [palette]
    elif isinstance(palette, dict):
        base_cycle = list(palette.values()) or timetk_colors
    else:
        base_cycle = list(palette) or timetk_colors

    if not base_cycle:
        base_cycle = timetk_colors

    mapping: Dict[str, str] = {}
    if isinstance(palette, dict):
        for idx, category in enumerate(categories):
            mapping[category] = palette.get(category, base_cycle[idx % len(base_cycle)])
        return mapping

    for idx, category in enumerate(categories):
        mapping[category] = base_cycle[idx % len(base_cycle)]
    return mapping


def _fill_with_alpha(color: str, alpha: float) -> str:
    """
    Best-effort conversion of a color string into an RGBA value with transparency.
    """
    if not color:
        color = "#2c3e50"
    try:
        return hex_to_rgba(color, alpha=alpha)
    except ValueError:
        converted = name_to_hex(color)
        if converted:
            return hex_to_rgba(converted, alpha=alpha)
        return color


def _format_facet_label(row: pd.Series, columns: Sequence[str], sep: str) -> str:
    parts: List[str] = []
    for col in columns:
        value = row[col]
        if col.startswith("__facet__"):
            continue
        if pd.isna(value):
            continue
        parts.append(f"{col}={value}")
    return sep.join(parts)


@pf.register_groupby_method
@pf.register_dataframe_method
def plot_time_series_boxplot(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector],
    period: Union[str, pd.DateOffset, pd.Timedelta, np.timedelta64, timedelta],
    color_column: Optional[Union[str, ColumnSelector]] = None,
    color_palette: Optional[Union[Dict[str, str], Sequence[str], str]] = None,
    facet_vars: Optional[Union[str, Sequence[str], ColumnSelector]] = None,
    facet_ncols: int = 1,
    facet_label_sep: str = ", ",
    box_fill_color: str = "#2c3e50",
    box_fill_alpha: float = 0.25,
    box_line_color: str = "#2c3e50",
    box_line_width: float = 1.2,
    outlier_color: str = "#2c3e50",
    boxpoints: str = "outliers",
    smooth: bool = True,
    smooth_func: Union[str, Callable[[pd.Series], float]] = "mean",
    smooth_color: str = "#3366FF",
    smooth_line_width: float = 2.0,
    smooth_line_dash: str = "solid",
    smooth_alpha: float = 0.9,
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    y_intercept_dash: str = "dash",
    legend_show: bool = True,
    color_lab: str = "Legend",
    title: str = "Time Series Box Plot",
    x_lab: str = "",
    y_lab: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    plotly_dropdown: bool = False,
    plotly_dropdown_x: float = 1.05,
    plotly_dropdown_y: float = 1.05,
    hovertemplate: Optional[str] = None,
) -> go.Figure:
    """
    Visualize rolling distributions of a time series by aggregating values into
    fixed windows (weeks, months, etc.) and rendering box plots per window.
    Supports pandas or polars inputs, tidy-style selectors, grouped data, and an
    optional Plotly dropdown for faceted series.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Long-format time series data or grouped data whose groups are treated
        as facet combinations. Polars DataFrames are converted automatically.
    date_column : str or ColumnSelector
        Datetime column to bucket by ``period``.
    value_column : str or ColumnSelector
        Numeric column plotted on the y-axis.
    period : str, pd.DateOffset, Timedelta, or timedelta
        Window size passed to :func:`pytimetk.floor_date`. Accepts pandas
        frequency strings (``"7D"``, ``"1M"``) or human-friendly durations
        (``"30 minutes"``, ``"2 weeks"``).
    color_column : str or ColumnSelector, optional
        Optional categorical column that splits the distribution/legend.
    color_palette : dict, sequence, or str, optional
        Custom palette for ``color_column``. Dicts map ``{category: "#RRGGBB"}``.
        Sequences are cycled; ``"timetk"`` reuses the package palette.
    facet_vars : str, sequence, or ColumnSelector, optional
        Additional columns used to facet the output. Combined with any pandas
        ``groupby`` columns on the input.
    facet_ncols : int, optional
        Number of subplot columns when ``plotly_dropdown`` is ``False``.
    facet_label_sep : str, optional
        Separator used when composing facet labels (default ``", "``).
    box_fill_color, box_fill_alpha : optional
        Styling for boxes when ``color_column`` is ``None``.
    box_line_color, box_line_width : optional
        Outline styling for box traces.
    outlier_color : str, optional
        Marker color for outliers.
    boxpoints : str, optional
        Plotly ``boxpoints`` argument (``"outliers"``, ``"all"``, ``False``).
    smooth : bool, optional
        Draw a smoothed summary line over the box centers.
    smooth_func : str or callable, optional
        Aggregation applied before plotting the smoothing line (default ``"mean"``).
    smooth_color, smooth_line_width, smooth_line_dash, smooth_alpha : optional
        Styling for the smoothing line.
    y_intercept : float, optional
        Optional horizontal reference line.
    legend_show : bool, optional
        Display the legend (only applies when ``color_column`` is supplied).
    color_lab : str, optional
        Legend title when ``color_column`` is provided.
    title, x_lab, y_lab : str, optional
        Layout titles and axis labels.
    width, height : int, optional
        Figure size in pixels. Height defaults to a sensible value based on the
        number of facets.
    plotly_dropdown : bool, optional
        When True and multiple facet combinations exist, render a dropdown to
        switch between them instead of drawing subplots.
    plotly_dropdown_x, plotly_dropdown_y : float, optional
        Dropdown anchor location.
    hovertemplate : str, optional
        Custom hover template for the box traces.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing one subplot per facet (or dropdown entry) with box
        plots per period bucket and optional smoothing lines.

    Examples
    --------
    ```{python}
    import pytimetk as tk

    df = tk.load_dataset("taylor_30_min", parse_dates=["date"]).assign(
        month=lambda d: d["date"].dt.month_name()
    )

    fig = tk.plot_time_series_boxplot(
        data=df,
        date_column="date",
        value_column="value",
        period="1 week",
        facet_vars="month",
        title="Weekly Revenue Distribution",
    )
    fig
    ```

    ```{python}
    # Dropdown example with tidy selectors
    from pytimetk.utils.selection import contains

    fig_dropdown = tk.plot_time_series_boxplot(
        data=df.assign(weekday=lambda d: d["date"].dt.day_name()),
        date_column="date",
        value_column=contains("value"),
        period="1 week",
        facet_vars="month",
        color_column="weekday",
        plotly_dropdown=True,
    )
    fig_dropdown
    ```
    """

    # Convert polars inputs to pandas for downstream compatibility
    if pl is not None:
        if isinstance(data, pl.DataFrame):
            data = data.to_pandas()
        elif hasattr(pl.dataframe.group_by, "GroupBy") and isinstance(
            data,
            pl.dataframe.group_by.GroupBy,  # type: ignore[arg-type]
        ):
            data = data.to_pandas()  # type: ignore[attr-defined]

    period_unit = _normalise_period_spec(period)

    base_for_selectors = (
        resolve_pandas_groupby_frame(data)
        if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy)
        else data
    )

    date_column = _resolve_single_selector(
        date_column, base_for_selectors, "date_column"
    )
    value_column = _resolve_single_selector(
        value_column, base_for_selectors, "value_column"
    )
    color_column_name: Optional[str] = None
    if color_column is not None:
        color_column_name = _resolve_single_selector(
            color_column, base_for_selectors, "color_column"
        )

    facet_var_names: List[str] = []
    if facet_vars is not None:
        facet_var_names = resolve_column_selection(
            base_for_selectors,
            facet_vars,
            allow_none=False,
            require_match=True,
            unique=True,
        )

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_columns = [col for col in data.grouper.names if col is not None]
        frame = resolve_pandas_groupby_frame(data).copy()
    else:
        group_columns = []
        frame = data.copy()

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "`data` must be a pandas DataFrame or GroupBy after conversion."
        )

    if frame.empty:
        raise ValueError("`data` contains no rows to plot.")

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.dropna(subset=[date_column, value_column])
    if frame.empty:
        raise ValueError("All rows were dropped after coercing date/value columns.")

    frame["__bucket__"] = floor_date(frame[date_column], unit=period_unit)
    frame = frame.dropna(subset=["__bucket__"]).reset_index(drop=True)

    color_label_col: Optional[str] = None
    color_categories: List[str] = []
    color_mapping: Dict[str, str] = {}
    if color_column_name:
        color_label_col = "__color_label__"
        color_labels = frame[color_column_name].astype("string")
        color_labels = color_labels.fillna("Missing")
        frame[color_label_col] = color_labels.astype(str)
        color_categories = frame[color_label_col].drop_duplicates().tolist()
        color_mapping = _build_color_mapping(color_categories, color_palette)

    facet_columns: List[str] = []
    for col in group_columns + facet_var_names:
        if col not in facet_columns:
            facet_columns.append(col)

    fallback_facet: Optional[str] = None
    if not facet_columns:
        fallback_facet = "__facet__"
        facet_columns = [fallback_facet]
        frame[fallback_facet] = ""

    facet_frame = frame[facet_columns].drop_duplicates().reset_index(drop=True)
    facet_labels = facet_frame.apply(
        lambda row: _format_facet_label(row, facet_columns, facet_label_sep), axis=1
    ).tolist()

    n_facets = len(facet_labels)
    if n_facets == 0:
        raise ValueError(
            "Unable to determine facet combinations for the supplied data."
        )

    use_dropdown = plotly_dropdown and n_facets > 1

    if use_dropdown:
        ncols = 1
        facet_rows = 1
    else:
        ncols = (
            1 if facet_ncols is None or facet_ncols <= 0 else min(facet_ncols, n_facets)
        )
        facet_rows = math.ceil(n_facets / ncols)

    if height is None:
        base_rows = 1 if use_dropdown else facet_rows
        height = max(450, base_rows * 320)

    subplot_titles: Optional[List[str]] = None
    if not use_dropdown:
        subplot_titles = []
        total_slots = facet_rows * ncols
        for idx in range(total_slots):
            label = facet_labels[idx] if idx < n_facets else ""
            subplot_titles.append(label or "")

    subplot_kwargs = dict(
        rows=(1 if use_dropdown else facet_rows),
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )
    if subplot_titles is not None:
        subplot_kwargs["subplot_titles"] = subplot_titles

    require_plotly()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(**subplot_kwargs)

    legend_visible = legend_show and bool(color_column_name)
    box_hovertemplate = (
        hovertemplate or "Bucket=%{x|%Y-%m-%d %H:%M}<br>Value=%{y:.3f}<extra></extra>"
    )

    trace_indices_by_facet: List[List[int]] = []

    for facet_idx in range(n_facets):
        mask = pd.Series(True, index=frame.index)
        for col in facet_columns:
            mask &= frame[col] == facet_frame.loc[facet_idx, col]
        facet_data = frame.loc[mask].copy()
        if facet_data.empty:
            if use_dropdown:
                trace_indices_by_facet.append([])
            continue

        if use_dropdown:
            facet_trace_indices: List[int] = []

        if use_dropdown:
            row = col = 1
        else:
            row = (facet_idx // ncols) + 1
            col = (facet_idx % ncols) + 1

        visible = (not use_dropdown) or (facet_idx == 0)

        # Box traces
        if color_label_col:
            categories_iter = (
                color_categories
                or facet_data[color_label_col].drop_duplicates().tolist()
            )
            for category in categories_iter:
                subset = facet_data[facet_data[color_label_col] == category]
                if subset.empty:
                    continue
                color_value = color_mapping.get(category, box_line_color)
                trace = go.Box(
                    x=subset["__bucket__"],
                    y=subset[value_column],
                    name=category,
                    marker=dict(color=color_value, outliercolor=outlier_color),
                    fillcolor=_fill_with_alpha(color_value, box_fill_alpha),
                    line=dict(color=color_value, width=box_line_width),
                    boxpoints=boxpoints,
                    legendgroup=category,
                    offsetgroup=category,
                    showlegend=legend_visible and facet_idx == 0,
                    hovertemplate=box_hovertemplate,
                    visible=visible,
                )
                fig.add_trace(trace, row=row, col=col)
                if use_dropdown:
                    facet_trace_indices.append(len(fig.data) - 1)
        else:
            trace = go.Box(
                x=facet_data["__bucket__"],
                y=facet_data[value_column],
                name="Distribution",
                marker=dict(color=outlier_color, outliercolor=outlier_color),
                fillcolor=_fill_with_alpha(box_fill_color, box_fill_alpha),
                line=dict(color=box_line_color, width=box_line_width),
                boxpoints=boxpoints,
                showlegend=False,
                hovertemplate=box_hovertemplate,
                visible=visible,
            )
            fig.add_trace(trace, row=row, col=col)
            if use_dropdown:
                facet_trace_indices.append(len(fig.data) - 1)

        # Smooth traces
        if smooth:
            smooth_group_cols = ["__bucket__"]
            if color_label_col:
                smooth_group_cols.append(color_label_col)
            agg = (
                facet_data.groupby(smooth_group_cols, dropna=False)[value_column]
                .agg(smooth_func if smooth_func is not None else "mean")
                .reset_index()
                .rename(columns={value_column: "__smooth__"})
                .sort_values("__bucket__")
            )
            if color_label_col:
                categories_iter = (
                    color_categories or agg[color_label_col].drop_duplicates().tolist()
                )
                for category in categories_iter:
                    subset = agg[agg[color_label_col] == category]
                    if subset.empty:
                        continue
                    trace = go.Scatter(
                        x=subset["__bucket__"],
                        y=subset["__smooth__"],
                        mode="lines",
                        line=dict(
                            color=smooth_color,
                            width=smooth_line_width,
                            dash=smooth_line_dash,
                        ),
                        opacity=smooth_alpha,
                        showlegend=False,
                        hoverinfo="skip",
                        visible=visible,
                    )
                    fig.add_trace(trace, row=row, col=col)
                    if use_dropdown:
                        facet_trace_indices.append(len(fig.data) - 1)
            else:
                if not agg.empty:
                    trace = go.Scatter(
                        x=agg["__bucket__"],
                        y=agg["__smooth__"],
                        mode="lines",
                        line=dict(
                            color=smooth_color,
                            width=smooth_line_width,
                            dash=smooth_line_dash,
                        ),
                        opacity=smooth_alpha,
                        showlegend=False,
                        hoverinfo="skip",
                        visible=visible,
                    )
                    fig.add_trace(trace, row=row, col=col)
                    if use_dropdown:
                        facet_trace_indices.append(len(fig.data) - 1)

        if y_intercept is not None:
            bucket_min = facet_data["__bucket__"].min()
            bucket_max = facet_data["__bucket__"].max()
            if pd.notna(bucket_min) and pd.notna(bucket_max):
                trace = go.Scatter(
                    x=[bucket_min, bucket_max],
                    y=[y_intercept, y_intercept],
                    mode="lines",
                    line=dict(
                        color=y_intercept_color, dash=y_intercept_dash, width=1.2
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=visible,
                )
                fig.add_trace(trace, row=row, col=col)
                if use_dropdown:
                    facet_trace_indices.append(len(fig.data) - 1)

        if use_dropdown:
            trace_indices_by_facet.append(facet_trace_indices)

    # Axis labels
    if use_dropdown:
        fig.update_xaxes(title_text=x_lab, row=1, col=1)
        fig.update_yaxes(title_text=y_lab, row=1, col=1)
    else:
        for r in range(1, facet_rows + 1):
            for c in range(1, ncols + 1):
                facet_number = (r - 1) * ncols + (c - 1)
                if facet_number >= n_facets:
                    continue
                fig.update_xaxes(
                    title_text=x_lab if r == facet_rows else "", row=r, col=c
                )
                fig.update_yaxes(title_text=y_lab if c == 1 else "", row=r, col=c)

    layout_kwargs = dict(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        plot_bgcolor="#f7f9fb",
        paper_bgcolor="white",
        margin=dict(l=80, r=30, t=90, b=60),
        width=width,
        height=height,
        boxmode="group",
        showlegend=legend_visible,
    )

    if color_column_name:
        layout_kwargs["legend"] = dict(title=dict(text=color_lab))

    if use_dropdown and trace_indices_by_facet:
        total_traces = len(fig.data)
        buttons = []
        for idx, label in enumerate(facet_labels[: len(trace_indices_by_facet)]):
            visibility = [False] * total_traces
            for trace_idx in trace_indices_by_facet[idx]:
                if trace_idx < total_traces:
                    visibility[trace_idx] = True
            button_title = f"{title}<br>{label}" if label else title
            buttons.append(
                dict(
                    label=label or f"Facet {idx + 1}",
                    method="update",
                    args=[
                        {"visible": visibility},
                        {"title": {"text": button_title}},
                    ],
                )
            )
        layout_kwargs["updatemenus"] = [
            dict(
                type="dropdown",
                x=plotly_dropdown_x,
                y=plotly_dropdown_y,
                showactive=True,
                buttons=buttons,
            )
        ]

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5ebf2", zeroline=False)
    fig.update_annotations(yshift=10)

    return fig

import math
from typing import List, Optional, Sequence, Union

import pandas as pd
import pandas_flavor as pf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:  # Optional dependency for seamless polars support
    import polars as pl
except ImportError:  # pragma: no cover - optional import
    pl = None

from pytimetk.core.stl_diagnostics import stl_diagnostics
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame


VALID_FEATURES = ["observed", "season", "trend", "remainder", "seasadj"]


def _to_pandas(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]):
    if pl is not None:
        pl_groupby_cls = getattr(pl.dataframe.group_by, "GroupBy", None)
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
        if pl_groupby_cls is not None and isinstance(
            data,
            pl_groupby_cls,  # pragma: no cover - optional path
        ):
            return data.to_pandas()  # type: ignore[attr-defined]
    return data


def _resolve_single_selector(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    selector: Union[str, ColumnSelector],
    label: str,
) -> str:
    if isinstance(selector, str):
        return selector
    resolved = resolve_column_selection(
        resolve_pandas_groupby_frame(data)
        if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy)
        else data,
        selector,
        allow_none=False,
        require_match=True,
    )
    if len(resolved) != 1:
        raise ValueError(f"`{label}` selector must resolve to exactly one column.")
    return resolved[0]


def _validate_feature_set(feature_set: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(feature_set, str):
        cleaned = [feature_set]
    else:
        cleaned = list(feature_set)

    if not cleaned:
        raise ValueError("`feature_set` must include at least one STL component.")

    bad = [feature for feature in cleaned if feature not in VALID_FEATURES]
    if bad:
        raise ValueError(
            "`feature_set` contains unsupported entries: " + ", ".join(sorted(set(bad)))
        )
    return cleaned


@pf.register_groupby_method
@pf.register_dataframe_method
def plot_stl_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector],
    feature_set: Union[str, Sequence[str]] = (
        "observed",
        "season",
        "trend",
        "remainder",
        "seasadj",
    ),
    facet_vars: Optional[Union[str, Sequence[str], ColumnSelector]] = None,
    facet_ncols: int = 1,
    frequency: Union[str, int, float] = "auto",
    trend: Union[str, int, float] = "auto",
    robust: bool = True,
    line_color: str = "#2c3e50",
    line_width: float = 2.0,
    line_dash: str = "solid",
    line_alpha: float = 1.0,
    title: str = "STL Diagnostics",
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
    Visualize STL decomposition components (observed, season, trend, remainder,
    seasonally adjusted) for one or more time series using Plotly. Supports
    tidy selectors, pandas GroupBy objects, and polars inputs.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Time series data in long format, optionally grouped.
    date_column : str or ColumnSelector
        Datetime column plotted on the x-axis.
    value_column : str or ColumnSelector
        Numeric column that is decomposed.
    feature_set : str or sequence, optional
        Subset (or single value) of ``{"observed", "season", "trend",
        "remainder", "seasadj"}`` to plot. Defaults to all components.
    facet_vars : str, sequence, or ColumnSelector, optional
        Additional categorical columns used to facet the output.
    facet_ncols : int, optional
        Number of facet columns when ``plotly_dropdown`` is ``False``.
    frequency : str, int, float, optional
        Seasonal period forwarded to :func:`pytimetk.core.stl_diagnostics.stl_diagnostics`.
    trend : str, int, float, optional
        STL trend window specification forwarded to :func:`stl_diagnostics`.
    robust : bool, optional
        Use robust STL fitting. Defaults to ``True``.
    line_color, line_width, line_dash, line_alpha : optional
        Styling for the component lines.
    title, x_lab, y_lab : str, optional
        Figure and axis labels.
    width, height : int, optional
        Figure dimensions in pixels.
    plotly_dropdown : bool, optional
        When ``True`` and multiple facet combinations exist, render a dropdown
        to switch between them.
    plotly_dropdown_x, plotly_dropdown_y : float, optional
        Dropdown anchor coordinates.
    hovertemplate : str, optional
        Custom Plotly hover template.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure showing STL components per facet.

    Examples
    --------
    ```{python}
    import pytimetk as tk

    df = tk.load_dataset("taylor_30_min", parse_dates=["date"])

    fig = tk.plot_stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        title="STL decomposition",
    )
    fig
    ```

    ```{python}
    # Faceted example with additional feature configuration
    df_features = df.assign(hour=lambda d: d["date"].dt.hour)

    fig_faceted = tk.plot_stl_diagnostics(
        data=df_features,
        date_column="date",
        value_column="value",
        feature_set=["observed", "trend", "remainder"],
        facet_vars="hour",
        plotly_dropdown=True,
    )
    fig_faceted
    ```
    """

    feature_list = _validate_feature_set(feature_set)
    data = _to_pandas(data)

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_columns = [col for col in data.grouper.names if col is not None]
        base_frame = resolve_pandas_groupby_frame(data).copy()
    elif isinstance(data, pd.DataFrame):
        group_columns = []
        base_frame = data.copy()
    else:
        raise TypeError("`data` must be a pandas DataFrame or GroupBy.")

    if base_frame.empty:
        raise ValueError("`data` contains no rows to decompose.")

    date_column_name = _resolve_single_selector(base_frame, date_column, "date_column")
    value_column_name = _resolve_single_selector(
        base_frame, value_column, "value_column"
    )

    base_frame[date_column_name] = pd.to_datetime(
        base_frame[date_column_name], errors="coerce"
    )
    base_frame[value_column_name] = pd.to_numeric(
        base_frame[value_column_name], errors="coerce"
    )
    base_frame = base_frame.dropna(subset=[date_column_name, value_column_name])
    if base_frame.empty:
        raise ValueError("No valid rows remain after coercing date/value columns.")

    facet_var_names: List[str] = []
    if facet_vars is not None:
        facet_var_names = resolve_column_selection(
            base_frame,
            facet_vars,
            allow_none=False,
            require_match=True,
            unique=True,
        )

    grouping_union = group_columns.copy()
    for name in facet_var_names:
        if name not in grouping_union:
            grouping_union.append(name)

    if grouping_union:
        diag_input = base_frame.groupby(grouping_union, dropna=False, sort=False)
    else:
        diag_input = base_frame

    diagnostics = stl_diagnostics(
        data=diag_input,
        date_column=date_column_name,
        value_column=value_column_name,
        frequency=frequency,
        trend=trend,
        robust=robust,
    )

    melt_cols = [
        col
        for col in diagnostics.columns
        if col not in grouping_union + [date_column_name]
    ]
    diagnostics_long = diagnostics.melt(
        id_vars=grouping_union + [date_column_name],
        value_vars=[col for col in feature_list if col in melt_cols],
        var_name="feature",
        value_name="value",
    )

    diagnostics_long = diagnostics_long.dropna(subset=["value"])
    if diagnostics_long.empty:
        raise ValueError("STL diagnostics produced no rows to plot.")

    facet_columns = grouping_union
    fallback_facet: Optional[str] = None
    if not facet_columns:
        fallback_facet = "__facet__"
        diagnostics_long[fallback_facet] = ""
        facet_columns = [fallback_facet]

    facet_frame = (
        diagnostics_long[facet_columns].drop_duplicates().reset_index(drop=True)
    )

    def _format_facet(row: pd.Series) -> str:
        parts: List[str] = []
        for col in facet_columns:
            if fallback_facet and col == fallback_facet:
                continue
            parts.append(f"{col}={row[col]}")
        return ", ".join(part for part in parts if part)

    facet_labels = facet_frame.apply(_format_facet, axis=1).tolist()
    n_facets = len(facet_labels)
    use_dropdown = plotly_dropdown and n_facets > 1

    if use_dropdown:
        ncols = 1
        facet_rows = 1
    else:
        ncols = (
            1
            if facet_ncols is None or facet_ncols <= 0
            else min(facet_ncols, max(n_facets, 1))
        )
        facet_rows = math.ceil(max(n_facets, 1) / ncols)

    ordered_features = [feature for feature in feature_list if feature in melt_cols]
    if not ordered_features:
        raise ValueError(
            "Requested `feature_set` columns are missing from diagnostics output."
        )

    rows = len(ordered_features) if use_dropdown else facet_rows * len(ordered_features)

    if height is None:
        base_rows = 1 if use_dropdown else facet_rows
        height = max(450, base_rows * len(ordered_features) * 220)

    subplot_titles: Optional[List[str]] = None
    if not use_dropdown:
        subplot_titles = []
        total_slots = facet_rows * ncols * len(ordered_features)
        for facet_row in range(facet_rows):
            for feature in ordered_features:
                for col in range(ncols):
                    facet_idx = facet_row * ncols + col
                    if facet_idx >= n_facets:
                        subplot_titles.append("")
                    else:
                        label = facet_labels[facet_idx]
                        subplot_titles.append(
                            f"{feature} — {label}" if label else feature
                        )
        while len(subplot_titles) < total_slots:
            subplot_titles.append("")

    subplot_kwargs = dict(
        rows=rows,
        cols=(1 if use_dropdown else ncols),
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )
    if subplot_titles is not None:
        subplot_kwargs["subplot_titles"] = subplot_titles

    fig = make_subplots(**subplot_kwargs)

    hovertemplate = (
        hovertemplate or "Date=%{x|%Y-%m-%d %H:%M}<br>Value=%{y:.3f}<extra></extra>"
    )
    trace_indices_by_facet: List[List[int]] = []

    for facet_idx in range(n_facets):
        facet_mask = pd.Series(True, index=diagnostics_long.index)
        for col in facet_columns:
            facet_mask &= diagnostics_long[col] == facet_frame.loc[facet_idx, col]

        facet_data = diagnostics_long.loc[facet_mask]
        if facet_data.empty and use_dropdown:
            trace_indices_by_facet.append([])
            continue

        facet_traces: List[int] = []

        for feature_idx, feature in enumerate(ordered_features):
            subset = facet_data[facet_data["feature"] == feature]
            if subset.empty:
                continue

            if use_dropdown:
                row = feature_idx + 1
                col = 1
            else:
                row = (facet_idx // ncols) * len(ordered_features) + feature_idx + 1
                col = (facet_idx % ncols) + 1

            visible = (not use_dropdown) or (facet_idx == 0)

            trace = go.Scatter(
                x=subset[date_column_name],
                y=subset["value"],
                mode="lines",
                line=dict(color=line_color, width=line_width, dash=line_dash),
                opacity=line_alpha,
                showlegend=False,
                hovertemplate=hovertemplate,
                name=feature,
                visible=visible,
            )
            fig.add_trace(trace, row=row, col=col)
            if use_dropdown:
                facet_traces.append(len(fig.data) - 1)

            # Axis titles handled globally below

        if use_dropdown:
            trace_indices_by_facet.append(facet_traces)

    if use_dropdown:
        for feature_idx in range(len(ordered_features)):
            fig.update_yaxes(
                title_text=ordered_features[feature_idx], row=feature_idx + 1, col=1
            )
            fig.update_xaxes(
                title_text=x_lab if feature_idx == len(ordered_features) - 1 else "",
                row=feature_idx + 1,
                col=1,
            )
    else:
        for facet_row in range(facet_rows):
            for feature_idx, feature in enumerate(ordered_features):
                row = facet_row * len(ordered_features) + feature_idx + 1
                for col in range(1, ncols + 1):
                    if facet_row == facet_rows - 1:
                        fig.update_xaxes(title_text=x_lab, row=row, col=col)
                    else:
                        fig.update_xaxes(title_text="", row=row, col=col)
                    fig.update_yaxes(
                        title_text=feature if col == 1 else "", row=row, col=col
                    )

    layout_kwargs = dict(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        margin=dict(l=80, r=30, t=90, b=60),
        width=width,
        height=height,
    )

    if use_dropdown and trace_indices_by_facet:
        total_traces = len(fig.data)
        buttons = []
        for idx, label in enumerate(facet_labels):
            visibility = [False] * total_traces
            for trace_idx in trace_indices_by_facet[idx]:
                visibility[trace_idx] = True
            button_title = f"{title}<br>{label}" if label else title
            buttons.append(
                dict(
                    label=label or f"Facet {idx + 1}",
                    method="update",
                    args=[{"visible": visibility}, {"title": {"text": button_title}}],
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
    fig.update_annotations(yshift=10)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5ebf2", zeroline=False)

    return fig

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

from pytimetk.core.seasonal_diagnostics import seasonal_diagnostics
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame
from pytimetk.utils.plot_helpers import hex_to_rgba


@pf.register_groupby_method
@pf.register_dataframe_method
def plot_seasonal_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector],
    feature_set: Union[str, Sequence[str], None] = "auto",
    facet_vars: Optional[Union[str, Sequence[str], ColumnSelector]] = None,
    facet_ncols: Optional[int] = 1,
    geom: str = "box",
    geom_color: str = "#2c3e50",
    geom_outlier_color: str = "#2c3e50",
    title: str = "Seasonal Diagnostics",
    x_lab: str = "",
    y_lab: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    plotly_dropdown: bool = False,
    plotly_dropdown_x: float = 1.05,
    plotly_dropdown_y: float = 1.05,
) -> go.Figure:
    """
    Visualize seasonal patterns using box or violin plots grouped by seasonality
    features (hour, weekday, month, etc.). Works with pandas or polars inputs.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Time series data (long format) or grouped data. Polars DataFrames are supported.
    date_column : str or ColumnSelector
        Datetime column used to compute the seasonal features.
    value_column : str or ColumnSelector
        Numeric column plotted on the y-axis.
    feature_set : str or sequence, optional
        One or more of ``["second", "minute", "hour", "wday.lbl", "week",
        "month.lbl", "quarter", "year"]``. ``"auto"`` selects a sensible subset.
    facet_vars : str, sequence, or ColumnSelector, optional
        Additional categorical columns to facet by. They are treated as grouping
        columns for the diagnostics.
    facet_ncols : int, optional
        Number of facet columns when ``plotly_dropdown`` is ``False``. Defaults to 2.
    geom : {"box", "violin"}, optional
        Plotting geometry for each seasonal feature. Defaults to ``"box"``.
    geom_color : str, optional
        Primary color for the box/violin geometry. Defaults to ``"#2c3e50"``.
    geom_outlier_color : str, optional
        Outlier color for box plots. Defaults to ``"#2c3e50"``.
    title : str, optional
        Plot title.
    x_lab, y_lab : str, optional
        Axis labels.
    width, height : int, optional
        Figure dimensions in pixels. Height defaults to a sensible value based on
        the number of facets.
    plotly_dropdown : bool, optional
        When ``True`` and facet combinations exist, render a dropdown to switch
        between them.
    plotly_dropdown_x, plotly_dropdown_y : float, optional
        Dropdown position (only used when ``plotly_dropdown`` is ``True``).

    Returns
    -------
    plotly.graph_objects.Figure
        Figure containing one subplot per seasonal feature for each facet.

    Examples
    --------
    ```{python}
    import pytimetk as tk

    df = tk.load_dataset("taylor_30_min", parse_dates=["date"]).assign(
        month_name=lambda d: d["date"].dt.month_name()
    )

    fig = tk.plot_seasonal_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        feature_set="auto",
        geom="box",
    )
    fig
    ```

    ```{python}
    # Dropdown example, using tidy selectors
    from pytimetk.utils.selection import contains

    fig_dropdown = tk.plot_seasonal_diagnostics(
        data=df,
        date_column="date",
        value_column=contains("value"),
        feature_set="auto",
        facet_vars="month_name",
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

    # Resolve selectors to concrete column names
    def _resolve_single_selector(selector, data_obj, label: str) -> str:
        if isinstance(selector, str):
            return selector
        resolved = resolve_column_selection(
            data_obj, selector, allow_none=False, require_match=True
        )
        if len(resolved) != 1:
            raise ValueError(
                f"The `{label}` selector must resolve to exactly one column."
            )
        return resolved[0]

    base_data_for_selectors = (
        resolve_pandas_groupby_frame(data)
        if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy)
        else data
    )

    date_column = _resolve_single_selector(
        date_column, base_data_for_selectors, "date_column"
    )
    value_column = _resolve_single_selector(
        value_column, base_data_for_selectors, "value_column"
    )

    facet_var_names: List[str] = []
    if facet_vars is not None:
        facet_var_names = resolve_column_selection(
            base_data_for_selectors,
            facet_vars,
            allow_none=False,
            require_match=True,
            unique=True,
        )

    geom = geom.lower()
    if geom not in {"box", "violin"}:
        raise ValueError("`geom` must be one of {'box', 'violin'}.")

    # Prepare grouped data for diagnostics if facet vars supplied
    data_for_diag = data
    group_column_names: List[str] = []

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_column_names.extend(list(data.grouper.names))
        base_frame = resolve_pandas_groupby_frame(data)
    else:
        base_frame = data

    if facet_var_names:
        grouping_union = group_column_names.copy()
        for name in facet_var_names:
            if name not in grouping_union:
                grouping_union.append(name)
        data_for_diag = base_frame.groupby(grouping_union)
        group_column_names = grouping_union

    diagnostics = seasonal_diagnostics(
        data=data_for_diag,
        date_column=date_column,
        value_column=value_column,
        feature_set=feature_set,
    )

    base_cols = {date_column, value_column, "seasonal_feature", "seasonal_value"}
    facet_columns = [col for col in diagnostics.columns if col not in base_cols]

    ordered_features = diagnostics["seasonal_feature"].drop_duplicates().tolist()
    if not ordered_features:
        raise ValueError("No seasonal diagnostics available for the supplied data.")

    fallback_facet_col: Optional[str] = None
    if not facet_columns:
        fallback_facet_col = "__facet__"
        diagnostics[fallback_facet_col] = ""
        facet_columns = [fallback_facet_col]

    facet_frame = diagnostics[facet_columns].drop_duplicates().reset_index(drop=True)

    def _format_facet_label(row) -> str:
        parts: List[str] = []
        for col in facet_columns:
            value = row[col]
            if fallback_facet_col and col == fallback_facet_col:
                text = str(value)
            else:
                text = f"{col}={value}"
            if text and text.strip():
                parts.append(text)
        return ", ".join(parts)

    facet_labels = facet_frame.apply(
        lambda row: _format_facet_label(row), axis=1
    ).tolist()

    n_facets = len(facet_labels)
    use_dropdown = plotly_dropdown and n_facets > 1

    if use_dropdown:
        ncols = 1
        facet_rows = 1
        rows = len(ordered_features)
    else:
        if facet_ncols is None or facet_ncols <= 0:
            ncols = 1
        else:
            ncols = max(1, min(facet_ncols, n_facets))
        facet_rows = math.ceil(n_facets / ncols)
        rows = facet_rows * len(ordered_features)

    subplot_titles: List[str] = []
    if use_dropdown:
        subplot_titles = ordered_features
    else:
        for block in range(facet_rows):
            for feature in ordered_features:
                for col in range(ncols):
                    facet_idx = block * ncols + col
                    if facet_idx >= n_facets:
                        subplot_titles.append(feature)
                    else:
                        label = facet_labels[facet_idx]
                        subplot_titles.append(
                            f"{feature} — {label}" if label else feature
                        )

    marker_color = geom_color

    if height is None:
        height = max(450, rows * 260)

    fig = make_subplots(
        rows=rows,
        cols=ncols,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        subplot_titles=subplot_titles,
    )

    trace_indices_by_facet: List[List[int]] = [] if use_dropdown else []

    for facet_idx in range(n_facets):
        facet_mask = pd.Series(True, index=diagnostics.index)
        for col in facet_columns:
            facet_mask &= diagnostics[col] == facet_frame.loc[facet_idx, col]

        if use_dropdown:
            facet_trace_indices: List[int] = []

        for feature_idx, feature in enumerate(ordered_features):
            subset = diagnostics.loc[
                facet_mask & (diagnostics["seasonal_feature"] == feature)
            ]
            if subset.empty:
                continue

            x_values = subset["seasonal_value"].astype(str)
            y_values = subset[value_column]

            if use_dropdown:
                row = feature_idx + 1
                col = 1
            else:
                block_row = facet_idx // ncols
                row = block_row * len(ordered_features) + feature_idx + 1
                col = (facet_idx % ncols) + 1

            visible = (not use_dropdown) or (facet_idx == 0)
            if geom == "violin":
                trace = go.Violin(
                    x=x_values,
                    y=y_values,
                    fillcolor=hex_to_rgba(marker_color, alpha=0.35),
                    line=dict(color=marker_color),
                    box_visible=True,
                    meanline_visible=True,
                    points=False,
                    showlegend=False,
                    visible=visible,
                    name=feature,
                )
            else:
                trace = go.Box(
                    x=x_values,
                    y=y_values,
                    marker=dict(color=geom_outlier_color),
                    fillcolor=hex_to_rgba(marker_color, alpha=0.25),
                    line=dict(color=marker_color, width=1.2),
                    boxpoints="outliers",
                    showlegend=False,
                    visible=visible,
                    name=feature,
                )

            fig.add_trace(trace, row=row, col=col)
            if use_dropdown:
                facet_trace_indices.append(len(fig.data) - 1)

            if col == 1:
                fig.update_yaxes(
                    title_text=y_lab if feature_idx == 0 else "", row=row, col=col
                )

        if use_dropdown:
            trace_indices_by_facet.append(facet_trace_indices)

    # Apply x-axis labels to bottom row(s)
    if use_dropdown:
        for r in range(1, len(ordered_features) + 1):
            fig.update_xaxes(title_text=x_lab, row=r, col=1)
    else:
        for block in range(facet_rows):
            base_row = (block + 1) * len(ordered_features)
            for col in range(1, ncols + 1):
                fig.update_xaxes(title_text=x_lab, row=base_row, col=col)

    layout_kwargs = dict(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        plot_bgcolor="#f7f9fb",
        paper_bgcolor="white",
        margin=dict(l=80, r=20, t=90, b=60),
        width=width,
        height=height,
    )

    if use_dropdown:
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
    fig.update_xaxes(title_standoff=20, showgrid=False, zeroline=False)
    fig.update_yaxes(
        title_standoff=20, showgrid=True, zeroline=False, gridcolor="#e5ebf2"
    )
    fig.update_annotations(yshift=10)

    if not use_dropdown and n_facets > 1:
        for idx, label in enumerate(facet_labels):
            if not label:
                continue
            block_mid = idx * len(ordered_features) + len(ordered_features) / 2
            y = 1 - (block_mid / rows)
            fig.add_annotation(
                text=label,
                x=-0.04,
                xref="paper",
                y=y,
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="#2c3e50"),
                align="right",
            )

    return fig

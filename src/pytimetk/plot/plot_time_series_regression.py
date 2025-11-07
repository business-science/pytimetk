import pandas as pd
import pandas_flavor as pf
import plotly.graph_objects as go
import statsmodels.formula.api as smf

from typing import Any, Dict, List, Optional, Union

try:  # Optional dependency for seamless polars support
    import polars as pl
except ImportError:  # pragma: no cover - optional import
    pl = None

from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection


SERIES_COLUMN = "__regression_series__"
VALUE_COLUMN = "__regression_value__"


def _to_pandas(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]):
    if pl is not None:
        pl_groupby_cls = getattr(pl.dataframe.group_by, "GroupBy", None)
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
        if pl_groupby_cls is not None and isinstance(data, pl_groupby_cls):  # pragma: no cover - optional path
            return data.to_pandas()  # type: ignore[attr-defined]
    return data


def _resolve_date_column(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    selector: Union[str, ColumnSelector],
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
        raise ValueError("`date_column` selector must resolve to exactly one column.")
    return resolved[0]


def _prepare_long_frame(
    df: pd.DataFrame,
    date_column: str,
    response_column: str,
    fitted_values: pd.Series,
    group_columns: List[str],
) -> pd.DataFrame:
    base_cols = [col for col in group_columns if col in df.columns]
    cols = [date_column] + base_cols

    actual = df[cols].copy()
    actual[SERIES_COLUMN] = "observed"
    actual[VALUE_COLUMN] = pd.to_numeric(df[response_column], errors="coerce")

    fitted = df[cols].copy()
    fitted[SERIES_COLUMN] = "fitted"
    fitted[VALUE_COLUMN] = pd.to_numeric(fitted_values, errors="coerce")

    combined = pd.concat([actual, fitted], ignore_index=True)
    return combined.dropna(subset=[date_column])


def _fit_group(
    df: pd.DataFrame,
    date_column: str,
    formula: str,
    show_summary: bool,
    group_label: Optional[str],
    model_kwargs: Dict[str, Any],
    group_columns: List[str],
) -> pd.DataFrame:
    model = smf.ols(formula=formula, data=df, **model_kwargs).fit()
    if show_summary:
        if group_label:
            print(f"\nSummary for Group: {group_label}\n{'-' * 40}")
        else:
            print("\nSummary\n" + "-" * 40)
        print(model.summary())
        print("-" * 40)

    fitted = model.predict(df)
    response_column = model.model.endog_names
    return _prepare_long_frame(df, date_column, response_column, fitted, group_columns)


@pf.register_groupby_method
@pf.register_dataframe_method
def plot_time_series_regression(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    formula: str,
    show_summary: bool = False,
    model_kwargs: Optional[Dict[str, Any]] = None,
    **plot_kwargs: Any,
) -> go.Figure:
    """
    Fit a linear regression using a formula and visualise observed vs fitted
    values over time using :func:`pytimetk.plot_timeseries`.

    Parameters
    ----------
    data : DataFrame or GroupBy
        Long-format time series data or grouped data via ``DataFrame.groupby``.
        Polars inputs are automatically converted to pandas.
    date_column : str or ColumnSelector
        Datetime column used on the x-axis.
    formula : str
        Patsy/Statsmodels formula passed to :func:`statsmodels.formula.api.ols`.
    show_summary : bool, optional
        Print statsmodels regression summaries (per group when grouped).
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`statsmodels.formula.api.ols`.
        Use this to pass ``eval_env`` when formulas depend on outer scope names.
    **plot_kwargs : dict, optional
        Additional keyword arguments forwarded to :func:`pytimetk.plot_timeseries`
        (faceting, dropdowns, theme overrides, etc.).

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly figure showing observed vs fitted values over time.

    Examples
    --------
    ```{python}
    import numpy as np
    import pytimetk as tk

    df = tk.load_dataset("taylor_30_min", parse_dates=["date"]).assign(
        trend=lambda d: np.arange(len(d))
    )

    fig = tk.plot_time_series_regression(
        data=df,
        date_column="date",
        formula="value ~ trend",
        title="Observed vs Fitted",
    )
    fig
    ```

    ```{python}
    # Grouped example
    df["half"] = np.where(df["trend"] < df["trend"].median(), "H1", "H2")
    fig_grouped = tk.plot_time_series_regression(
        data=df.groupby("half"),
        date_column="date",
        formula="value ~ trend",
        facet_ncol=1,
    )
    fig_grouped
    ```
    """

    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be a Patsy-style string such as 'y ~ x1 + x2'.")

    data = _to_pandas(data)

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        group_columns: List[str] = []
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_columns = [col for col in data.grouper.names if col is not None]
        frame = resolve_pandas_groupby_frame(data).copy()
    else:
        raise TypeError("`data` must be a pandas DataFrame or GroupBy.")

    if frame.empty:
        raise ValueError("`data` contains no rows.")

    date_column_name = _resolve_date_column(frame if not group_columns else frame, date_column)
    frame[date_column_name] = pd.to_datetime(frame[date_column_name], errors="coerce")
    frame = frame.dropna(subset=[date_column_name])
    if frame.empty:
        raise ValueError("No valid rows remain after coercing `date_column` to datetime.")

    model_kwargs = model_kwargs.copy() if model_kwargs is not None else {}
    model_kwargs.setdefault("eval_env", 2)

    result_frames: List[pd.DataFrame] = []

    if group_columns:
        grouped = frame.groupby(group_columns, dropna=False, sort=False)
        for key, group_df in grouped:
            if group_df.empty:
                continue
            label = ", ".join(f"{col}={val}" for col, val in zip(group_columns, key if isinstance(key, tuple) else (key,)))
            result_frames.append(
                _fit_group(
                    group_df,
                    date_column_name,
                    formula,
                    show_summary,
                    label,
                    model_kwargs,
                    group_columns,
                )
            )
    else:
        result_frames.append(
            _fit_group(
                frame,
                date_column_name,
                formula,
                show_summary,
                None,
                model_kwargs,
                [],
            )
        )

    if not result_frames:
        raise ValueError("Unable to compute regression fits for the supplied data.")

    plot_data = pd.concat(result_frames, ignore_index=True)

    plot_kwargs = plot_kwargs.copy()
    plot_kwargs.setdefault("legend_show", True)
    plot_kwargs.setdefault("smooth", False)

    reserved = {"data", "date_column", "value_column", "color_column"}
    conflicts = reserved.intersection(plot_kwargs.keys())
    if conflicts:
        conflict_list = ", ".join(sorted(conflicts))
        raise ValueError(
            f"The following plot keyword arguments are managed internally and "
            f"cannot be overridden: {conflict_list}"
        )

    plot_input = plot_data if not group_columns else plot_data.groupby(group_columns)

    from pytimetk.plot.plot_timeseries import plot_timeseries as _plot_timeseries

    fig = _plot_timeseries(
        data=plot_input,
        date_column=date_column_name,
        value_column=VALUE_COLUMN,
        color_column=SERIES_COLUMN,
        **plot_kwargs,
    )
    return fig

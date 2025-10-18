import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings
from typing import Optional, Sequence, Union

try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
    restore_output_type,
    conversion_to_pandas,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_drawdown(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    close_column: str,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate running drawdown statistics for pandas or polars data.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input time-series data. Grouped inputs are processed per group before
        the drawdown metrics are appended.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Name of the column containing the values used to compute drawdowns.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with the following columns appended:

        - ``{close_column}_peak``
        - ``{close_column}_drawdown``
        - ``{close_column}_drawdown_pct``

        The return type matches the input backend.

    Notes
    -----
    Drawdown measures the peak-to-trough decline of a series. The running peak
    is computed with a cumulative maximum per group (if present) and the
    drawdown percentage is expressed relative to that peak. When the peak is
    zero the percentage drawdown is left as ``NaN`` to avoid division by zero.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import polars as pl

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas DataFrame (engine inferred)
    dd_pd = (
        df.groupby("symbol")
        .augment_drawdown(
            date_column="date",
            close_column="close",
        )
    )

    # Polars DataFrame using the tk accessor
    dd_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_drawdown(
            date_column="date",
            close_column="close",
        )
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column, require_numeric_dtype=True)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    if engine_resolved == "cudf" and cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required for engine='cudf', but it is not installed."
        )

    conversion_engine = engine_resolved
    conversion: FrameConversion = convert_to_engine(data, conversion_engine)
    prepared_data = conversion.data

    if reduce_memory and conversion_engine == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and conversion_engine in ("polars", "cudf"):
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if conversion_engine == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_drawdown_pandas(
            data=sorted_data,
            close_column=close_column,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_drawdown. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_drawdown_pandas(
                data=pandas_input,
                close_column=close_column,
            )
        else:
            result = _augment_drawdown_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_drawdown_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_drawdown_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf drawdown backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        peak = df_sorted.groupby(list(group_columns), sort=False)[close_column].cummax()
    else:
        peak = df_sorted[close_column].cummax()

    drawdown = df_sorted[close_column] - peak
    drawdown_pct = (drawdown / peak).where(peak != 0)

    df_sorted[f"{close_column}_peak"] = peak
    df_sorted[f"{close_column}_drawdown"] = drawdown
    df_sorted[f"{close_column}_drawdown_pct"] = drawdown_pct

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_drawdown_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
) -> pd.DataFrame:
    """Pandas implementation of drawdown calculation."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        col = close_column

        # Calculate running peak, drawdown, and drawdown percentage
        df[f"{col}_peak"] = df[col].cummax()
        df[f"{col}_drawdown"] = df[col] - df[f"{col}_peak"]
        df[f"{col}_drawdown_pct"] = df[f"{col}_drawdown"] / df[f"{col}_peak"]

        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        df = resolve_pandas_groupby_frame(data).copy()
        col = close_column

        df[f"{col}_peak"] = df.groupby(group_names)[col].cummax()
        df[f"{col}_drawdown"] = df[col] - df[f"{col}_peak"]
        df[f"{col}_drawdown_pct"] = df[f"{col}_drawdown"] / df[f"{col}_peak"]

        return df

    raise TypeError("Unsupported data type passed to _augment_drawdown_pandas.")


def _augment_drawdown_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    """Polars implementation of drawdown calculation."""

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    peak_expr = (
        pl.col(close_column).cum_max().over(resolved_groups)
        if resolved_groups
        else pl.col(close_column).cum_max()
    )

    result = sorted_frame.with_columns(peak_expr.alias(f"{close_column}_peak"))
    result = result.with_columns(
        (pl.col(close_column) - pl.col(f"{close_column}_peak")).alias(
            f"{close_column}_drawdown"
        )
    )
    result = result.with_columns(
        (
            pl.col(f"{close_column}_drawdown") / pl.col(f"{close_column}_peak")
        ).alias(f"{close_column}_drawdown_pct")
    ).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result

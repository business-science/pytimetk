import pandas as pd
import polars as pl

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union

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
def augment_roc(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    start_index: int = 0,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Add rate-of-change (ROC) columns to a pandas or polars DataFrame.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are expanded per group before
        computing ROC features.
    date_column : str
        Name of the column containing date information. Used to preserve the
        original ordering.
    close_column : str
        Name of the column containing closing prices on which the ROC is
        computed.
    periods : int, tuple, or list, optional
        Lookback windows used for the denominator term. An integer adds a
        single ROC column, a tuple ``(start, end)`` expands to the inclusive
        range ``start..end``, and a list provides explicit periods. Defaults to
        ``1``.
    start_index : int, optional
        Offset applied to the numerator. When ``0`` (default) the current value
        is used; otherwise the numerator uses ``close.shift(start_index)``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If used
        with polars inputs a warning is emitted and no conversion is performed.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data type while also accepting explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with ROC columns appended. The return type matches the input
        backend (pandas or polars).

    Notes
    -----
    Rate of change is defined as ::

        ROC = (Value_t - Value_{t-period}) / Value_{t-period}

    When ``start_index`` is non-zero the numerator is taken from
    ``Value_{t-start_index}`` instead of the current value. The implementation
    safeguards against division by zero by returning ``NaN`` whenever the
    denominator is zero.

    Examples
    --------
    ```{python}
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas DataFrame input (engine inferred)
    roc_df = df.groupby("symbol").augment_roc(
        date_column="date",
        close_column="close",
        periods=[22, 63],
        start_index=5,
    )

    # Polars DataFrame input using the tk accessor
    roc_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_roc(
            date_column="date",
            close_column="close",
            periods=(5, 10),
        )
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    periods_list = _normalize_periods(periods)
    if start_index >= min(periods_list):
        raise ValueError("start_index must be less than the minimum value in periods.")

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

    close_columns = [close_column]

    if conversion_engine == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_roc_pandas(
            data=sorted_data,
            close_columns=close_columns,
            periods=periods_list,
            start_index=start_index,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_roc. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_roc_pandas(
                data=pandas_input,
                close_columns=close_columns,
                periods=periods_list,
                start_index=start_index,
            )
        else:
            result = _augment_roc_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_columns=close_columns,
                periods=periods_list,
                start_index=start_index,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_roc_polars(
            data=prepared_data,
            date_column=date_column,
            close_columns=close_columns,
            periods=periods_list,
            start_index=start_index,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_roc_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_columns: List[str],
    periods: List[int],
    start_index: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf roc backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    for col in close_columns:
        df_sorted[col] = df_sorted[col].astype("float64")

    group_list = list(group_columns) if group_columns else None

    for col in close_columns:
        if group_list:
            grouped_series = df_sorted.groupby(group_list, sort=False)[col]
        else:
            grouped_series = None

        for period in periods:
            denominator = (
                grouped_series.shift(period).reset_index(drop=True)
                if grouped_series is not None
                else df_sorted[col].shift(period)
            )
            if start_index == 0:
                numerator = df_sorted[col]
            else:
                numerator = (
                    grouped_series.shift(start_index).reset_index(drop=True)
                    if grouped_series is not None
                    else df_sorted[col].shift(start_index)
                )

            roc_series = (numerator / denominator) - 1
            roc_series = roc_series.where(denominator != 0)
            df_sorted[_roc_column(col, start_index, period)] = roc_series

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_roc_pandas(
    data,
    close_columns: List[str],
    periods: List[int],
    start_index: int,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for col in close_columns:
            for period in periods:
                df[_roc_column(col, start_index, period)] = _roc_pandas_series(
                    df[col], period, start_index
                )
        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = resolve_pandas_groupby_frame(data).copy()
        grouped = base_df.groupby(group_names)

        for col in close_columns:
            grouped_col = grouped[col]
            for period in periods:
                numerator = (
                    base_df[col] if start_index == 0 else grouped_col.shift(start_index)
                )
                denominator = grouped_col.shift(period)
                result = (numerator / denominator) - 1
                result = result.where(denominator != 0)
                base_df[_roc_column(col, start_index, period)] = result
        return base_df

    raise TypeError("Unsupported data type passed to _augment_roc_pandas.")


def _roc_pandas_series(series: pd.Series, period: int, start_index: int) -> pd.Series:
    if start_index == 0:
        return series.pct_change(period)
    numerator = series.shift(start_index)
    denominator = series.shift(period)
    return (numerator / denominator) - 1


def _augment_roc_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_columns: List[str],
    periods: List[int],
    start_index: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    roc_columns = [
        _roc_expression(col, period, start_index, resolved_groups)
        for col in close_columns
        for period in periods
    ]

    augmented = sorted_frame.with_columns(roc_columns).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _roc_expression(
    col: str,
    period: int,
    start_index: int,
    groups: Sequence[str],
) -> pl.Expr:
    numerator = pl.col(col) if start_index == 0 else pl.col(col).shift(start_index)
    denominator = pl.col(col).shift(period)

    if groups:
        numerator = numerator.over(groups)
        denominator = denominator.over(groups)

    expr = pl.when(denominator == 0).then(None).otherwise((numerator / denominator) - 1)
    return expr.alias(_roc_column(col, start_index, period))


def _roc_column(col: str, start_index: int, period: int) -> str:
    return f"{col}_roc_{start_index}_{period}"


def _normalize_periods(periods: Union[int, Tuple[int, int], List[int]]) -> List[int]:
    if isinstance(periods, int):
        return [periods]
    if isinstance(periods, tuple):
        if len(periods) != 2:
            raise ValueError("Expected tuple of length 2 for `periods`.")
        start, end = periods
        return list(range(start, end + 1))
    if isinstance(periods, list):
        return [int(p) for p in periods]
    raise TypeError(
        f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list."
    )

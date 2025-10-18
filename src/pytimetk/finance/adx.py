import numpy as np
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
from pytimetk.utils.polars_helpers import collect_lazyframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_adx(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI using pandas or polars backends.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        indicators are appended.
    date_column : str
        Name of the column containing date information.
    high_column : str
        Name of the column containing high prices.
    low_column : str
        Name of the column containing low prices.
    close_column : str
        Name of the column containing closing prices. Indicator columns are
        prefixed with this name.
    periods : int, tuple, or list, optional
        Lookback windows for smoothing. Accepts an integer, a tuple specifying
        an inclusive range, or a list of explicit periods. Defaults to ``14``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with the following columns appended for each ``period``:

        - ``{close_column}_plus_di_{period}``
        - ``{close_column}_minus_di_{period}``
        - ``{close_column}_adx_{period}``

        The return type matches the input backend.

    Notes
    -----
    The implementation follows Wilder's smoothing approach using exponential
    moving averages with ``alpha = 1 / period`` for the true range (TR) and
    directional movement (+DM, -DM) components. Division by zero is guarded by
    returning ``NaN`` when the denominator is zero.

    Examples
    --------
    ```{python}
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    adx_df = (
        df.groupby("symbol")
        .augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14, 28],
        )
    )

    adx_df.glimpse()
    ```

    ```{python}
    # Polars example (method chaining)
    import polars as pl

    pl_df = pl.from_pandas(df.query("symbol == 'AAPL'"))

    adx_pl = pl_df.tk.augment_adx(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=14,
    )

    adx_pl.glimpse()
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    periods_list = _normalize_periods(periods)

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
        result = _augment_adx_pandas(
            data=sorted_data,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=periods_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_adx. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_adx_pandas(
                data=pandas_input,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                periods=periods_list,
            )
        else:
            result = _augment_adx_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                periods=periods_list,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_adx_polars(
            data=prepared_data,
            date_column=date_column,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=periods_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_adx_pandas(
    data,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        grouped = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        grouped = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy()
    else:
        raise TypeError("Unsupported data type passed to _augment_adx_pandas.")

    col = close_column

    df["tr"] = pd.concat(
        [
            df[high_column] - df[low_column],
            (df[high_column] - df[col].shift(1)).abs(),
            (df[low_column] - df[col].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["plus_dm"] = np.where(
        (df[high_column] - df[high_column].shift(1))
        > (df[low_column].shift(1) - df[low_column]),
        np.maximum(df[high_column] - df[high_column].shift(1), 0),
        0,
    )
    df["minus_dm"] = np.where(
        (df[low_column].shift(1) - df[low_column])
        > (df[high_column] - df[high_column].shift(1)),
        np.maximum(df[low_column].shift(1) - df[low_column], 0),
        0,
    )

    if grouped is not None:
        grouped_obj = df.groupby(grouped)

    for period in periods:
        alpha = 1 / period
        if grouped is not None:
            tr_smooth = grouped_obj["tr"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
            plus_dm_smooth = grouped_obj["plus_dm"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
            minus_dm_smooth = grouped_obj["minus_dm"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
        else:
            tr_smooth = df["tr"].ewm(alpha=alpha, adjust=False).mean()
            plus_dm_smooth = df["plus_dm"].ewm(alpha=alpha, adjust=False).mean()
            minus_dm_smooth = df["minus_dm"].ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        df[f"{col}_plus_di_{period}"] = plus_di
        df[f"{col}_minus_di_{period}"] = minus_di
        df[f"{col}_adx_{period}"] = adx

    df.drop(columns=["tr", "plus_dm", "minus_dm"], inplace=True)
    return df


def _augment_adx_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf adx backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[high_column] = df_sorted[high_column].astype("float64")
    df_sorted[low_column] = df_sorted[low_column].astype("float64")
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        group_list = list(group_columns)
        prev_close = df_sorted.groupby(group_list, sort=False)[close_column].shift(1)
        prev_high = df_sorted.groupby(group_list, sort=False)[high_column].shift(1)
        prev_low = df_sorted.groupby(group_list, sort=False)[low_column].shift(1)
    else:
        group_list = None
        prev_close = df_sorted[close_column].shift(1)
        prev_high = df_sorted[high_column].shift(1)
        prev_low = df_sorted[low_column].shift(1)

    tr_candidates = cudf.DataFrame(
        {
            "a": df_sorted[high_column] - df_sorted[low_column],
            "b": (df_sorted[high_column] - prev_close).abs(),
            "c": (df_sorted[low_column] - prev_close).abs(),
        }
    )
    df_sorted["__adx_tr"] = tr_candidates.max(axis=1)

    up_move = (df_sorted[high_column] - prev_high).fillna(0)
    down_move = (prev_low - df_sorted[low_column]).fillna(0)

    df_sorted["__adx_plus_dm"] = up_move.where(
        (up_move > down_move) & (up_move > 0), 0.0
    )
    df_sorted["__adx_minus_dm"] = down_move.where(
        (down_move > up_move) & (down_move > 0), 0.0
    )

    for period in periods:
        alpha = 1.0 / float(period)

        tr_smooth = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")
        plus_dm_smooth = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")
        minus_dm_smooth = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")

        if group_list:
            for _, group_df in df_sorted.groupby(group_list, sort=False):
                idx = group_df.index
                tr_smooth.loc[idx] = group_df["__adx_tr"].ewm(
                    alpha=alpha,
                    adjust=False,
                ).mean()
                plus_dm_smooth.loc[idx] = group_df["__adx_plus_dm"].ewm(
                    alpha=alpha,
                    adjust=False,
                ).mean()
                minus_dm_smooth.loc[idx] = group_df["__adx_minus_dm"].ewm(
                    alpha=alpha,
                    adjust=False,
                ).mean()
        else:
            tr_smooth = df_sorted["__adx_tr"].ewm(alpha=alpha, adjust=False).mean()
            plus_dm_smooth = df_sorted["__adx_plus_dm"].ewm(
                alpha=alpha, adjust=False
            ).mean()
            minus_dm_smooth = df_sorted["__adx_minus_dm"].ewm(
                alpha=alpha, adjust=False
            ).mean()

        plus_di = (plus_dm_smooth / tr_smooth) * 100
        minus_di = (minus_dm_smooth / tr_smooth) * 100

        denom = plus_di + minus_di
        dx = ((plus_di - minus_di).abs() / denom) * 100
        dx = dx.where(denom != 0)

        if group_list:
            adx_series = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")
            for _, group_df in df_sorted.groupby(group_list, sort=False):
                idx = group_df.index
                adx_series.loc[idx] = dx.loc[idx].ewm(
                    alpha=alpha,
                    adjust=False,
                ).mean()
        else:
            adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

        df_sorted[f"{close_column}_plus_di_{period}"] = plus_di
        df_sorted[f"{close_column}_minus_di_{period}"] = minus_di
        df_sorted[f"{close_column}_adx_{period}"] = adx_series

    df_sorted = df_sorted.drop(columns=["__adx_tr", "__adx_plus_dm", "__adx_minus_dm"])

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_adx_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    def _maybe_over(expr: pl.Expr) -> pl.Expr:
        if resolved_groups:
            return expr.over(resolved_groups)
        return expr

    lazy_frame = sorted_frame.lazy()
    temp_columns: List[str] = []

    # Previous values per group
    prev_high_alias = "__adx_prev_high"
    prev_low_alias = "__adx_prev_low"
    prev_close_alias = "__adx_prev_close"
    temp_columns.extend([prev_high_alias, prev_low_alias, prev_close_alias])

    lazy_frame = lazy_frame.with_columns(
        _maybe_over(pl.col(high_column).shift(1)).alias(prev_high_alias),
        _maybe_over(pl.col(low_column).shift(1)).alias(prev_low_alias),
        _maybe_over(pl.col(close_column).shift(1)).alias(prev_close_alias),
    )

    delta_high_expr = pl.col(high_column) - pl.col(prev_high_alias)
    delta_low_expr = pl.col(prev_low_alias) - pl.col(low_column)

    tr_alias = "__adx_tr"
    plus_dm_alias = "__adx_plus_dm"
    minus_dm_alias = "__adx_minus_dm"
    temp_columns.extend([tr_alias, plus_dm_alias, minus_dm_alias])

    lazy_frame = lazy_frame.with_columns(
        pl.max_horizontal(
            pl.col(high_column) - pl.col(low_column),
            (pl.col(high_column) - pl.col(prev_close_alias)).abs(),
            (pl.col(low_column) - pl.col(prev_close_alias)).abs(),
        ).alias(tr_alias),
        pl.when(delta_high_expr > delta_low_expr)
        .then(
            pl.max_horizontal(delta_high_expr, pl.lit(0.0))
        )
        .otherwise(0.0)
        .alias(plus_dm_alias),
        pl.when(delta_low_expr > delta_high_expr)
        .then(
            pl.max_horizontal(delta_low_expr, pl.lit(0.0))
        )
        .otherwise(0.0)
        .alias(minus_dm_alias),
    )

    dx_aliases: List[str] = []

    for period in periods:
        alpha = 1.0 / period
        plus_sm_alias = f"__adx_plus_sm_{period}"
        minus_sm_alias = f"__adx_minus_sm_{period}"
        tr_sm_alias = f"__adx_tr_sm_{period}"
        dx_alias = f"__adx_dx_{period}"
        plus_alias = f"{close_column}_plus_di_{period}"
        minus_alias = f"{close_column}_minus_di_{period}"
        adx_alias = f"{close_column}_adx_{period}"

        temp_columns.extend(
            [plus_sm_alias, minus_sm_alias, tr_sm_alias, dx_alias]
        )

        lazy_frame = lazy_frame.with_columns(
            _maybe_over(
                pl.col(plus_dm_alias).ewm_mean(
                    alpha=alpha, adjust=False, min_periods=period
                )
            ).alias(plus_sm_alias),
            _maybe_over(
                pl.col(minus_dm_alias).ewm_mean(
                    alpha=alpha, adjust=False, min_periods=period
                )
            ).alias(minus_sm_alias),
            _maybe_over(
                pl.col(tr_alias).ewm_mean(
                    alpha=alpha, adjust=False, min_periods=period
                )
            ).alias(tr_sm_alias),
        )

        lazy_frame = lazy_frame.with_columns(
            (
                100 * (pl.col(plus_sm_alias) / pl.col(tr_sm_alias))
            ).alias(plus_alias),
            (
                100 * (pl.col(minus_sm_alias) / pl.col(tr_sm_alias))
            ).alias(minus_alias),
        )

        lazy_frame = lazy_frame.with_columns(
            (
                100
                * (pl.col(plus_alias) - pl.col(minus_alias)).abs()
                / (pl.col(plus_alias) + pl.col(minus_alias))
            ).alias(dx_alias)
        )

        lazy_frame = lazy_frame.with_columns(
            _maybe_over(
                pl.col(dx_alias).ewm_mean(
                    alpha=alpha, adjust=False, min_periods=period
                )
            ).alias(adx_alias)
        )

        dx_aliases.append(dx_alias)

    temp_columns.extend(dx_aliases)

    if temp_columns:
        lazy_frame = lazy_frame.drop(temp_columns)

    result = collect_lazyframe(lazy_frame).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result


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

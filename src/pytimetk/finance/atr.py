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

from pytimetk._polars_compat import ensure_polars_rolling_kwargs
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
def augment_atr(
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
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    normalize: bool = False,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate Average True Range (ATR) or Normalised ATR for pandas or polars data.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        indicators are appended.
    date_column : str
        Name of the column containing date information.
    high_column, low_column, close_column : str
        Column names used to compute the true range and ATR.
    periods : int, tuple, or list, optional
        Rolling window lengths. Accepts an integer, an inclusive tuple range,
        or an explicit list. Defaults to ``20``.
    normalize : bool, optional
        When ``True``, report the normalised ATR (``ATR / close * 100``). Defaults
        to ``False``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with ``{close_column}_atr_{period}`` (or ``_natr_`` when
        ``normalize=True``) columns appended for each requested period. The
        return type matches the input backend.

    Notes
    -----
    The Average True Range (ATR) follows Wilder's definition, using the maximum
    of the intra-period range, the high-to-previous-close distance, and the
    low-to-previous-close distance. When ``normalize=True`` the ATR is scaled
    by the close price and expressed as a percentage (often called NATR). Both
    pandas and polars implementations guard against division by zero by
    returning ``NaN`` when the denominator is zero.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    atr_pd = (
        df.groupby("symbol")
        .augment_atr(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14, 28],
            normalize=False,
        )
    )

    # Polars example using the tk accessor
    atr_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_atr(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=14,
            normalize=True,
        )
    )
    ```
    """

    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_date_column(data, date_column)

    period_list = _normalize_periods(periods)

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

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_atr_pandas(
            data=sorted_data,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=period_list,
            normalize=normalize,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif engine_resolved == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_atr. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_atr_pandas(
                data=pandas_input,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                periods=period_list,
                normalize=normalize,
            )
        else:
            result = _augment_atr_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                periods=period_list,
                normalize=normalize,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_atr_polars(
            data=prepared_data,
            date_column=date_column,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=period_list,
            normalize=normalize,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_atr_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    normalize: bool,
) -> pd.DataFrame:
    """Pandas implementation of ATR/NATR calculation."""
    type_str = "natr" if normalize else "atr"

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # True Range calculation as a column
        df["tr"] = pd.concat(
            [
                df[high_column] - df[low_column],
                (df[high_column] - df[close_column].shift(1)).abs(),
                (df[low_column] - df[close_column].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        for period in periods:
            atr = df["tr"].rolling(window=period, min_periods=1).mean()
            if normalize:
                atr = (atr / df[close_column] * 100).replace(
                    [float("inf"), -float("inf")], pd.NA
                )
            df[f"{close_column}_{type_str}_{period}"] = atr

        df = df.drop(columns=["tr"])

        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        df = resolve_pandas_groupby_frame(data).copy()
        # True Range calculation with group-aware shift
        prev_close = df.groupby(group_names)[close_column].shift(1)
        df["tr"] = pd.concat(
            [
                df[high_column] - df[low_column],
                (df[high_column] - prev_close).abs(),
                (df[low_column] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        for period in periods:
            atr = (
                df.groupby(group_names)["tr"]
                .rolling(window=period, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            if normalize:
                atr = (atr / df[close_column] * 100).replace(
                    [float("inf"), -float("inf")], pd.NA
                )
            df[f"{close_column}_{type_str}_{period}"] = atr

        df = df.drop(columns=["tr"])

        return df

    raise TypeError("Unsupported data type passed to _augment_atr_pandas.")


def _augment_atr_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    normalize: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    """Polars implementation of ATR/NATR calculation."""
    type_str = "natr" if normalize else "atr"

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

    prev_close_alias = "__atr_prev_close"
    tr_alias = "__atr_tr"
    temp_columns.extend([prev_close_alias, tr_alias])

    lazy_frame = lazy_frame.with_columns(
        _maybe_over(pl.col(close_column).shift(1)).alias(prev_close_alias)
    )

    lazy_frame = lazy_frame.with_columns(
        pl.max_horizontal(
            pl.col(high_column) - pl.col(low_column),
            (pl.col(high_column) - pl.col(prev_close_alias)).abs(),
            (pl.col(low_column) - pl.col(prev_close_alias)).abs(),
        ).alias(tr_alias)
    )

    for period in periods:
        rolling_kwargs = ensure_polars_rolling_kwargs(
            {"window_size": period, "min_samples": 1}
        )
        atr_expr = _maybe_over(
            pl.col(tr_alias).rolling_mean(**rolling_kwargs)
        )
        if normalize:
            atr_expr = pl.when(pl.col(close_column) == 0).then(None).otherwise(
                atr_expr / pl.col(close_column) * 100
            )
        lazy_frame = lazy_frame.with_columns(
            atr_expr.alias(f"{close_column}_{type_str}_{period}")
        )

    if temp_columns:
        lazy_frame = lazy_frame.drop(temp_columns)

    result = collect_lazyframe(lazy_frame).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result


def _augment_atr_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    normalize: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf atr backend.")

    type_str = "natr" if normalize else "atr"

    sort_cols: List[str] = [date_column]
    if group_columns:
        sort_cols = list(group_columns) + sort_cols

    df_sorted = frame.sort_values(sort_cols)
    df_sorted[high_column] = df_sorted[high_column].astype("float64")
    df_sorted[low_column] = df_sorted[low_column].astype("float64")
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        group_list = list(group_columns)
        prev_close = (
            df_sorted.groupby(group_list, sort=False)[close_column].shift(1)
        )
    else:
        group_list = None
        prev_close = df_sorted[close_column].shift(1)

    tr_candidates = cudf.DataFrame(
        {
            "a": df_sorted[high_column] - df_sorted[low_column],
            "b": (df_sorted[high_column] - prev_close).abs(),
            "c": (df_sorted[low_column] - prev_close).abs(),
        }
    )
    df_sorted["__atr_tr"] = tr_candidates.max(axis=1)

    for period in periods:
        if group_list is not None:
            atr_series = (
                df_sorted.groupby(group_list, sort=False)["__atr_tr"]
                .rolling(window=period, min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
        else:
            atr_series = df_sorted["__atr_tr"].rolling(window=period, min_periods=1).mean()

        if normalize:
            atr_series = atr_series.where(df_sorted[close_column] != 0, np.nan)
            atr_series = atr_series / df_sorted[close_column] * 100

        df_sorted[f"{close_column}_{type_str}_{period}"] = atr_series

    df_sorted = df_sorted.drop(columns=["__atr_tr"])

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


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

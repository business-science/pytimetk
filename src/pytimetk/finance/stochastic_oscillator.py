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
from pytimetk.utils.polars_helpers import collect_lazyframe
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
def augment_stochastic_oscillator(
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
    k_periods: Union[int, Tuple[int, int], List[int]] = 14,
    d_periods: Union[int, List[int]] = 3,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate Stochastic Oscillator (%K and %D) using pandas or polars backends.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before
        the indicators are appended.
    date_column : str
        Name of the column containing date information.
    high_column : str
        Column containing high prices.
    low_column : str
        Column containing low prices.
    close_column : str
        Column containing closing prices. Resulting columns are prefixed with
        this name.
    k_periods : int, tuple, or list, optional
        Lookback window(s) for the %K calculation. Accepts a single integer, an
        inclusive tuple range, or an explicit list. Defaults to ``14``.
    d_periods : int or list, optional
        Lookback window(s) for the %D smoothing calculation. Defaults to ``3``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with %K and %D columns appended for every combination of
        ``k_periods`` and ``d_periods``. The return type matches the input
        backend.

    Notes
    -----
    %K is defined as ``100 * (Close - LowestLow) / (HighestHigh - LowestLow)``
    where LowestLow/HighestHigh span the specified lookback window. %D is the
    rolling mean of %K over ``d_periods``. Division-by-zero scenarios yield
    ``NaN`` values to match the pandas behaviour.

    Examples
    --------
    ```{python}
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    stoch_df = df.groupby("symbol").augment_stochastic_oscillator(
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        k_periods=[14, 21],
        d_periods=[3, 9],
    )

    # Polars example (method chaining)
    stoch_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_stochastic_oscillator(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            k_periods=14,
            d_periods=[3],
        )
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    k_values = _normalize_periods(k_periods, label="k_periods")
    d_values = _normalize_d_periods(d_periods)

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
        result = _augment_stochastic_pandas(
            data=sorted_data,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            k_periods=k_values,
            d_periods=d_values,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_stochastic_oscillator. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_stochastic_pandas(
                data=pandas_input,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                k_periods=k_values,
                d_periods=d_values,
            )
        else:
            result = _augment_stochastic_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                high_column=high_column,
                low_column=low_column,
                close_column=close_column,
                k_periods=k_values,
                d_periods=d_values,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_stochastic_polars(
            data=prepared_data,
            date_column=date_column,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            k_periods=k_values,
            d_periods=d_values,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_stochastic_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: List[int],
    d_periods: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required to execute the cudf stochastic oscillator backend."
        )

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[high_column] = df_sorted[high_column].astype("float64")
    df_sorted[low_column] = df_sorted[low_column].astype("float64")
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    group_list = list(group_columns) if group_columns else None

    for k in k_periods:
        if group_list:
            lowest_low = (
                df_sorted.groupby(group_list, sort=False)[low_column]
                .rolling(window=k, min_periods=1)
                .min()
                .reset_index(drop=True)
            )
            highest_high = (
                df_sorted.groupby(group_list, sort=False)[high_column]
                .rolling(window=k, min_periods=1)
                .max()
                .reset_index(drop=True)
            )
        else:
            lowest_low = df_sorted[low_column].rolling(window=k, min_periods=1).min()
            highest_high = df_sorted[high_column].rolling(window=k, min_periods=1).max()

        denominator = highest_high - lowest_low
        k_alias = f"{close_column}_stoch_k_{k}"
        numerator = df_sorted[close_column] - lowest_low
        df_sorted[k_alias] = (100 * numerator / denominator).where(denominator != 0)

        for d in d_periods:
            d_alias = f"{close_column}_stoch_d_{k}_{d}"
            if group_list:
                df_sorted[d_alias] = (
                    df_sorted.groupby(group_list, sort=False)[k_alias]
                    .rolling(window=d, min_periods=1)
                    .mean()
                    .reset_index(drop=True)
                )
            else:
                df_sorted[d_alias] = df_sorted[k_alias].rolling(
                    window=d, min_periods=1
                ).mean()

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_stochastic_pandas(
    data,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: List[int],
    d_periods: List[int],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        grouped = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        grouped = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy()
    else:
        raise TypeError("Unsupported data type passed to _augment_stochastic_pandas.")

    if grouped is not None:
        group_obj = df.groupby(grouped)

    for k in k_periods:
        if grouped is not None:
            lowest_low = group_obj[low_column].transform(
                lambda s: s.rolling(window=k, min_periods=1).min()
            )
            highest_high = group_obj[high_column].transform(
                lambda s: s.rolling(window=k, min_periods=1).max()
            )
        else:
            lowest_low = df[low_column].rolling(window=k, min_periods=1).min()
            highest_high = df[high_column].rolling(window=k, min_periods=1).max()

        denominator = highest_high - lowest_low
        k_alias = f"{close_column}_stoch_k_{k}"
        df[k_alias] = (100 * (df[close_column] - lowest_low) / denominator).where(
            denominator != 0
        )

        for d in d_periods:
            d_alias = f"{close_column}_stoch_d_{k}_{d}"
            if grouped is not None:
                df[d_alias] = group_obj[k_alias].transform(
                    lambda s: s.rolling(window=d, min_periods=1).mean()
                )
            else:
                df[d_alias] = df[k_alias].rolling(window=d, min_periods=1).mean()

    return df


def _augment_stochastic_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: List[int],
    d_periods: List[int],
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

    for k in k_periods:
        band_kwargs = ensure_polars_rolling_kwargs(
            {"window_size": k, "min_samples": 1}
        )
        lowest_low_expr = _maybe_over(
            pl.col(low_column).rolling_min(**band_kwargs)
        )
        highest_high_expr = _maybe_over(
            pl.col(high_column).rolling_max(**band_kwargs)
        )
        denom_expr = highest_high_expr - lowest_low_expr

        k_alias = f"{close_column}_stoch_k_{k}"
        lazy_frame = lazy_frame.with_columns(
            pl.when(denom_expr == 0)
            .then(None)
            .otherwise(
                100
                * (pl.col(close_column) - lowest_low_expr)
                / denom_expr
            )
            .alias(k_alias)
        )

        for d in d_periods:
            d_alias = f"{close_column}_stoch_d_{k}_{d}"
            avg_kwargs = ensure_polars_rolling_kwargs(
                {"window_size": d, "min_samples": 1}
            )
            lazy_frame = lazy_frame.with_columns(
                _maybe_over(
                    pl.col(k_alias).rolling_mean(**avg_kwargs)
                ).alias(d_alias)
            )

    result = collect_lazyframe(lazy_frame).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result


def _normalize_periods(periods: Union[int, Tuple[int, int], List[int]], label: str) -> List[int]:
    if isinstance(periods, int):
        return [periods]
    if isinstance(periods, tuple):
        if len(periods) != 2:
            raise ValueError(f"Expected tuple of length 2 for `{label}`.")
        start, end = periods
        return list(range(start, end + 1))
    if isinstance(periods, list):
        return [int(p) for p in periods]
    raise TypeError(
        f"Invalid {label} specification: type: {type(periods)}. Please use int, tuple, or list."
    )


def _normalize_d_periods(d_periods: Union[int, List[int]]) -> List[int]:
    if isinstance(d_periods, int):
        return [d_periods]
    if isinstance(d_periods, list):
        return [int(d) for d in d_periods]
    raise TypeError(
        f"Invalid d_periods specification: type: {type(d_periods)}. Please use int or list."
    )

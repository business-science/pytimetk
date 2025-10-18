import pandas as pd
import polars as pl

import pandas_flavor as pf
import warnings
from typing import Optional, Sequence, Union

import numpy as np

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
def augment_ppo(
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
    fast_period: int = 12,
    slow_period: int = 26,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate the Percentage Price Oscillator (PPO) for pandas or polars data.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Financial data to augment with PPO values. Grouped inputs are processed
        per group before the indicator columns are appended.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Name of the closing price column.
    fast_period : int, optional
        Lookback window for the fast EMA.
    slow_period : int, optional
        Lookback window for the slow EMA.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. Defaults to inferring from the input data type.

    Returns
    -------
    DataFrame
        DataFrame with PPO values appended. Matches the backend of the input data.

    Notes
    -----
    The PPO is computed as the percentage difference between a fast and a slow
    exponential moving average (EMA):

        PPO = (EMA_fast - EMA_slow) / EMA_slow * 100

    The implementation follows the common convention of using ``min_periods=0``
    on the EMA calculations to accumulate values from the beginning of the
    series. Division-by-zero scenarios yield ``NaN`` to align with pandas'
    behaviour.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    ppo_pd = (
        df.groupby("symbol")
        .augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=26,
        )
    )

    # Polars example using the tk accessor
    ppo_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_ppo(
            date_column="date",
            close_column="close",
            fast_period=12,
            slow_period=26,
        )
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
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
        result = _augment_ppo_pandas(
            data=sorted_data,
            close_column=close_column,
            fast_period=fast_period,
            slow_period=slow_period,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_ppo. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_ppo_pandas(
                data=pandas_input,
                close_column=close_column,
                fast_period=fast_period,
                slow_period=slow_period,
            )
        else:
            result = _augment_ppo_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                fast_period=fast_period,
                slow_period=slow_period,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_ppo_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            fast_period=fast_period,
            slow_period=slow_period,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_ppo_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    fast_period: int,
    slow_period: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf ppo backend.")

    sort_columns: Sequence[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + list(sort_columns)

    df_sorted = frame.sort_values(list(sort_columns))
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    result_name = f"{close_column}_ppo_line_{fast_period}_{slow_period}"
    df_sorted[result_name] = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")

    if group_columns:
        group_list = list(group_columns)
        grouped_iter = df_sorted.groupby(group_list, sort=False)
        for _, group_df in grouped_iter:
            group_indices = group_df.index
            close_series = group_df[close_column]
            ema_fast = close_series.ewm(
                span=fast_period, adjust=False, min_periods=0
            ).mean()
            ema_slow = close_series.ewm(
                span=slow_period, adjust=False, min_periods=0
            ).mean()
            denom = ema_slow
            ppo_series = ((ema_fast - ema_slow) / denom) * 100
            ppo_series = ppo_series.where(denom != 0)
            df_sorted.loc[group_indices, result_name] = ppo_series
    else:
        ema_fast = df_sorted[close_column].ewm(
            span=fast_period, adjust=False, min_periods=0
        ).mean()
        ema_slow = df_sorted[close_column].ewm(
            span=slow_period, adjust=False, min_periods=0
        ).mean()
        denom = ema_slow
        ppo_series = ((ema_fast - ema_slow) / denom) * 100
        df_sorted[result_name] = ppo_series.where(denom != 0)

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_ppo_pandas(
    data,
    close_column: str,
    fast_period: int,
    slow_period: int,
) -> pd.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = resolve_pandas_groupby_frame(data).copy()

        base_df = base_df.groupby(group_names, group_keys=False).apply(
            lambda x: _calculate_ppo_pandas(x, close_column, fast_period, slow_period)
        )
        return base_df

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        return _calculate_ppo_pandas(df, close_column, fast_period, slow_period)

    raise TypeError("Unsupported data type passed to _augment_ppo_pandas.")


def _calculate_ppo_pandas(df, close_column, fast_period, slow_period):
    ema_fast = (
        df[close_column].ewm(span=fast_period, adjust=False, min_periods=0).mean()
    )
    ema_slow = (
        df[close_column].ewm(span=slow_period, adjust=False, min_periods=0).mean()
    )

    ppo_line = (ema_fast - ema_slow) / ema_slow * 100

    df[f"{close_column}_ppo_line_{fast_period}_{slow_period}"] = ppo_line
    return df


def _augment_ppo_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    fast_period: int,
    slow_period: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    fast_ema = pl.col(close_column).ewm_mean(
        span=fast_period, adjust=False, min_periods=0
    )
    slow_ema = pl.col(close_column).ewm_mean(
        span=slow_period, adjust=False, min_periods=0
    )

    if resolved_groups:
        fast_ema = fast_ema.over(resolved_groups)
        slow_ema = slow_ema.over(resolved_groups)

    ppo_expr = pl.when(slow_ema == 0)
    ppo_expr = ppo_expr.then(None).otherwise((fast_ema - slow_ema) / slow_ema * 100)

    result = sorted_frame.with_columns(
        ppo_expr.alias(f"{close_column}_ppo_line_{fast_period}_{slow_period}")
    ).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result

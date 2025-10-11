import pandas as pd
import polars as pl

import pandas_flavor as pf
import warnings
from typing import Optional, Sequence, Union

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
    resolve_polars_group_columns,
    restore_output_type,
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
    ],
    date_column: str,
    close_column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Calculate PPO using pandas or polars backends.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Financial data to augment with PPO values.
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
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if engine_resolved == "pandas":
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


def _augment_ppo_pandas(
    data,
    close_column: str,
    fast_period: int,
    slow_period: int,
) -> pd.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = data.obj.copy()

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

    result = (
        sorted_frame.with_columns(ppo_expr.alias(f"{close_column}_ppo_line_{fast_period}_{slow_period}"))
        .sort(row_col)
    )

    if generated:
        result = result.drop(row_col)

    return result

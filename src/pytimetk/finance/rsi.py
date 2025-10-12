import pandas as pd
import polars as pl

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union

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
def augment_rsi(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the Relative Strength Index (RSI) for pandas or polars data.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Financial data to augment. Grouped inputs are processed per group
        before RSI columns are appended.
    date_column : str
        Name of the column containing date information.
    close_column : str or list[str]
        Column name(s) containing the closing prices used to compute RSI. When
        a list is supplied an RSI is generated for each column.
    periods : int, tuple, or list, optional
        Lookback window(s) used when computing RSI. Accepts an integer, an
        inclusive tuple range, or a list of explicit periods. Defaults to ``14``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with ``{close_column}_rsi_{period}`` columns appended. The
        return type matches the input backend.

    Notes
    -----
    RSI follows Wilder's formulation, separating gains and losses before
    computing smoothed averages and forming the ratio. Values range from 0 to
    100, with extreme readings typically interpreted as overbought or
    oversold. Division-by-zero cases yield ``NaN`` which mirrors pandas
    behaviour.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    rsi_pd = (
        df.groupby("symbol")
        .augment_rsi(
            date_column="date",
            close_column="close",
            periods=[14, 28],
        )
    )

    # Polars example using the tk accessor
    rsi_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_rsi(
            date_column="date",
            close_column=["close"],
            periods=14,
        )
    )
    ```
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

    close_columns = (
        [close_column] if isinstance(close_column, str) else list(close_column)
    )
    periods_list = _normalize_periods(periods)

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_rsi_pandas(
            data=sorted_data,
            close_columns=close_columns,
            periods=periods_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_rsi_polars(
            data=prepared_data,
            date_column=date_column,
            close_columns=close_columns,
            periods=periods_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_rsi_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_columns: List[str],
    periods: List[int],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()

        for col in close_columns:
            for period in periods:
                df[f"{col}_rsi_{period}"] = _calculate_rsi_pandas(df[col], period)

        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = data.obj.copy()

        for col in close_columns:
            for period in periods:
                base_df[f"{col}_rsi_{period}"] = base_df.groupby(
                    group_names, group_keys=False
                )[col].apply(_calculate_rsi_pandas, period=period)

        return base_df

    raise TypeError("Unsupported data type passed to _augment_rsi_pandas.")


def _augment_rsi_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_columns: List[str],
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

    rsi_columns = [
        _rsi_expression(col=col, period=period)
        for col in close_columns
        for period in periods
    ]

    if resolved_groups:
        group_key = resolved_groups if len(resolved_groups) > 1 else resolved_groups[0]

        def apply_group(df_group: pl.DataFrame) -> pl.DataFrame:
            return df_group.with_columns(rsi_columns)

        augmented = (
            sorted_frame.group_by(group_key, maintain_order=True)
            .map_groups(apply_group)
            .sort(row_col)
        )
    else:
        augmented = sorted_frame.with_columns(rsi_columns).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _calculate_rsi_pandas(series: pd.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate the sum of gains and losses using a rolling window
    mean_gains = gains.rolling(window=period).mean()
    mean_losses = losses.rolling(window=period).mean()

    # Calculate RSI
    ret = 100 - (100 / (1 + (mean_gains / mean_losses)))
    return ret


def _rsi_expression(
    col: str,
    period: int,
) -> pl.Expr:
    delta = pl.col(col).diff()

    gains = pl.when(delta > 0).then(delta).otherwise(0.0)
    losses = pl.when(delta < 0).then(-delta).otherwise(0.0)

    avg_gain = gains.rolling_mean(window_size=period, min_periods=period)
    avg_loss = losses.rolling_mean(window_size=period, min_periods=period)

    rs = pl.when(avg_loss == 0).then(None).otherwise(avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi.alias(f"{col}_rsi_{period}")


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

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
def augment_cmo(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the Chande Momentum Oscillator (CMO) using pandas or polars backends.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        indicator columns are appended.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Name of the column containing closing prices.
    periods : int, tuple, or list, optional
        Lookback window(s) applied to the CMO calculation. Accepts a single
        integer, an inclusive tuple range, or an explicit list. Defaults to ``14``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with ``{close_column}_cmo_{period}`` columns appended for
        every requested period. The return type matches the input backend.

    Notes
    -----
    The Chande Momentum Oscillator (CMO) compares the magnitude of recent gains
    to recent losses over the supplied lookback window. Values range from -100
    (all losses) to +100 (all gains). Division-by-zero cases are guarded by
    returning ``NaN`` which matches the pandas behaviour.

    Examples
    --------
    ```python
    import pytimetk as tk
    import polars as pl

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    cmo_pd = (
        df.groupby("symbol")
        .augment_cmo(
            date_column="date",
            close_column="close",
            periods=[14, 28],
        )
    )

    # Polars example using explicit engine
    cmo_pl = tk.augment_cmo(
        data=pl.from_pandas(df.query("symbol == 'AAPL'")),
        date_column="date",
        close_column="close",
        periods=14,
        engine="polars",
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column, require_numeric_dtype=True)
    check_date_column(data, date_column)

    periods_list = _normalize_periods(periods)

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
        result = _augment_cmo_pandas(
            data=sorted_data,
            close_column=close_column,
            periods=periods_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_cmo_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            periods=periods_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_cmo_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
    periods: List[int],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for period in periods:
            df[f"{close_column}_cmo_{period}"] = _calculate_cmo_pandas(
                df[close_column], period=period
            )
        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        base_df = data.obj.copy()
        for period in periods:
            base_df[f"{close_column}_cmo_{period}"] = base_df.groupby(
                group_names, group_keys=False
            )[close_column].apply(_calculate_cmo_pandas, period=period)
        return base_df

    raise TypeError("Unsupported data type passed to _augment_cmo_pandas.")


def _calculate_cmo_pandas(series: pd.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate the sum of gains and losses using a rolling window
    sum_gains = gains.rolling(window=period).sum()
    sum_losses = losses.rolling(window=period).sum()

    # Calculate CMO
    cmo = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    return cmo


def _augment_cmo_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
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

    def compute(frame: pl.DataFrame) -> pl.DataFrame:
        df = frame.with_columns(
            (pl.col(close_column) - pl.col(close_column).shift(1)).alias("_cmo_delta")
        )
        df = df.with_columns(
            [
                pl.when(pl.col("_cmo_delta") > 0)
                .then(pl.col("_cmo_delta"))
                .otherwise(0.0)
                .alias("_cmo_gain"),
                pl.when(pl.col("_cmo_delta") < 0)
                .then(-pl.col("_cmo_delta"))
                .otherwise(0.0)
                .alias("_cmo_loss"),
            ]
        )

        for period in periods:
            gain_sum = pl.col("_cmo_gain").rolling_sum(
                window_size=period, min_samples=period
            )
            loss_sum = pl.col("_cmo_loss").rolling_sum(
                window_size=period, min_samples=period
            )
            denom = gain_sum + loss_sum
            cmo_expr = pl.when(denom == 0).then(None).otherwise(
                100 * (gain_sum - loss_sum) / denom
            )
            df = df.with_columns(cmo_expr.alias(f"{close_column}_cmo_{period}"))

        return df.drop(["_cmo_delta", "_cmo_gain", "_cmo_loss"])

    if resolved_groups:
        group_key = resolved_groups if len(resolved_groups) > 1 else resolved_groups[0]
        result = (
            sorted_frame.group_by(group_key, maintain_order=True)
            .map_groups(compute)
            .sort(row_col)
        )
    else:
        result = compute(sorted_frame).sort(row_col)

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

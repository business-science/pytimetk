import numpy as np
import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import List, Optional, Sequence, Union

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
def augment_macd(
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
    signal_period: int = 9,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate MACD for a given financial instrument using either pandas or polars engine.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Name of the column containing closing price data.
    fast_period : int, optional
        Number of periods for the fast EMA in MACD calculation.
    slow_period : int, optional
        Number of periods for the slow EMA in MACD calculation.
    signal_period : int, optional
        Number of periods for the signal line EMA in MACD calculation.
    reduce_memory : bool, optional
        Whether to reduce memory usage of the data before performing the calculation.
    engine : {"auto", "pandas", "polars", "cudf"}, optional
        Computation engine to use. Defaults to infer from the input data type.

    Returns
    -------
    DataFrame
        DataFrame with MACD line, signal line, and MACD histogram added. Matches
        the backend of the input data.

    Notes
    -----
    The MACD (Moving Average Convergence Divergence) is a
    trend-following momentum indicator that shows the relationship
    between two moving averages of a security’s price. Developed by
    Gerald Appel in the late 1970s, the MACD is one of the simplest
    and most effective momentum indicators available.

    MACD Line: The MACD line is the difference between two
    exponential moving averages (EMAs) of a security’s price,
    typically the 12-day and 26-day EMAs.

    Signal Line: This is usually a 9-day EMA of the MACD line. It
    acts as a trigger for buy and sell signals.

    Histogram: The MACD histogram plots the difference between the
    MACD line and the signal line. A histogram above zero indicates
    that the MACD line is above the signal line (bullish), and
    below zero indicates it is below the signal line (bearish).

    Crossovers: The most common MACD signals are when the MACD line
    crosses above or below the signal line. A crossover above the
    signal line is a bullish signal, indicating it might be time to
    buy, and a crossover below the signal line is bearish,
    suggesting it might be time to sell.


    Examples
    --------

    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])

    df
    ```

    ```{python}
    # MACD pandas engine
    df_macd = (
        df
            .groupby('symbol')
            .augment_macd(
                date_column = 'date',
                close_column = 'close',
                fast_period = 12,
                slow_period = 26,
                signal_period = 9,
                engine = "pandas"
            )
    )

    df_macd.glimpse()
    ```

    ```{python}
    # MACD polars engine
    pl_df = pl.from_pandas(df)
    df_macd = (
        pl_df
            .group_by('symbol')
            .tk.augment_macd(
                date_column = 'date',
                close_column = 'close',
                fast_period = 12,
                slow_period = 26,
                signal_period = 9,
            )
    )

    df_macd.glimpse()
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

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_macd_pandas(
            data=sorted_data,
            close_column=close_column,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif engine_resolved == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_macd. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_macd_pandas(
                data=pandas_input,
                close_column=close_column,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
            )
        else:
            result = _augment_macd_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                fast_period=fast_period,
                slow_period=slow_period,
                signal_period=signal_period,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_macd_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_macd_pandas(
    data,
    close_column,
    fast_period,
    slow_period,
    signal_period,
):
    """
    Internal function to calculate MACD using Pandas.
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # If data is a GroupBy object, apply MACD calculation for each group
        group_names = data.grouper.names
        data = resolve_pandas_groupby_frame(data)

        df = data.copy()

        df = df.groupby(group_names, group_keys=False).apply(
            lambda x: _calculate_macd_pandas(
                x, close_column, fast_period, slow_period, signal_period
            )
        )
    elif isinstance(data, pd.DataFrame):
        # If data is a DataFrame, apply MACD calculation directly
        df = data.copy()
        df = _calculate_macd_pandas(
            df, close_column, fast_period, slow_period, signal_period
        )
    else:
        raise ValueError("data must be a pandas DataFrame or a pandas GroupBy object")

    return df


def _calculate_macd_pandas(df, close_column, fast_period, slow_period, signal_period):
    """
    Calculate MACD, Signal Line, and MACD Histogram for a DataFrame.
    """
    # Calculate Fast and Slow EMAs
    ema_fast = (
        df[close_column].ewm(span=fast_period, adjust=False, min_periods=0).mean()
    )
    ema_slow = (
        df[close_column].ewm(span=slow_period, adjust=False, min_periods=0).mean()
    )

    # Calculate MACD Line
    macd_line = ema_fast - ema_slow

    # Calculate Signal Line
    signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=0).mean()

    # Calculate MACD Histogram
    macd_histogram = macd_line - signal_line

    # Add columns
    df[f"{close_column}_macd_line_{fast_period}_{slow_period}_{signal_period}"] = (
        macd_line
    )
    df[
        f"{close_column}_macd_signal_line_{fast_period}_{slow_period}_{signal_period}"
    ] = signal_line
    df[f"{close_column}_macd_histogram_{fast_period}_{slow_period}_{signal_period}"] = (
        macd_histogram
    )

    # # Calculate Bullish and Bearish Crossovers
    # df[f'{close_column}_macd_bullish_crossover_{fast_period}_{slow_period}_{signal_period}'] = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

    # df[f'{close_column}_macd_bearish_crossover_{fast_period}_{slow_period}_{signal_period}'] = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    return df


def _augment_macd_polars(
    data,
    date_column,
    close_column,
    fast_period,
    slow_period,
    signal_period,
    group_columns,
    row_id_column,
):
    """
    Internal function to calculate MACD using Polars.
    """
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    fast_ema_expr = pl.col(close_column).ewm_mean(
        span=fast_period, adjust=False, min_periods=0
    )
    slow_ema_expr = pl.col(close_column).ewm_mean(
        span=slow_period, adjust=False, min_periods=0
    )

    if resolved_groups:
        fast_ema_expr = fast_ema_expr.over(resolved_groups)
        slow_ema_expr = slow_ema_expr.over(resolved_groups)

    fast_alias = "_macd_fast_ema"
    slow_alias = "_macd_slow_ema"
    macd_alias = f"{close_column}_macd_line_{fast_period}_{slow_period}_{signal_period}"
    signal_alias = (
        f"{close_column}_macd_signal_line_{fast_period}_{slow_period}_{signal_period}"
    )
    hist_alias = (
        f"{close_column}_macd_histogram_{fast_period}_{slow_period}_{signal_period}"
    )

    augmented = sorted_frame.with_columns(
        [
            fast_ema_expr.alias(fast_alias),
            slow_ema_expr.alias(slow_alias),
        ]
    ).with_columns(
        [
            (pl.col(fast_alias) - pl.col(slow_alias)).alias(macd_alias),
        ]
    )

    signal_expr = pl.col(macd_alias).ewm_mean(
        span=signal_period, adjust=False, min_periods=0
    )
    if resolved_groups:
        signal_expr = signal_expr.over(resolved_groups)

    augmented = augmented.with_columns(signal_expr.alias(signal_alias))
    augmented = augmented.with_columns(
        (pl.col(macd_alias) - pl.col(signal_alias)).alias(hist_alias)
    ).drop([fast_alias, slow_alias])

    augmented = augmented.sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _augment_macd_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf macd backend.")

    sort_cols: List[str] = [date_column]
    if group_columns:
        sort_cols = list(group_columns) + sort_cols

    df_sorted = frame.sort_values(sort_cols)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    macd_alias = f"{close_column}_macd_line_{fast_period}_{slow_period}_{signal_period}"
    signal_alias = f"{close_column}_macd_signal_line_{fast_period}_{slow_period}_{signal_period}"
    hist_alias = f"{close_column}_macd_histogram_{fast_period}_{slow_period}_{signal_period}"

    df_sorted[macd_alias] = np.nan
    df_sorted[signal_alias] = np.nan
    df_sorted[hist_alias] = np.nan

    if group_columns:
        group_iter = df_sorted.groupby(list(group_columns), sort=False)
    else:
        group_iter = [(None, df_sorted)]

    for _, subdf in group_iter:
        idx = subdf.index
        close = subdf[close_column]

        ema_fast = close.ewm(span=fast_period, adjust=False, min_periods=0).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False, min_periods=0).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False, min_periods=0).mean()
        hist = macd_line - signal_line

        df_sorted.loc[idx, macd_alias] = macd_line
        df_sorted.loc[idx, signal_alias] = signal_line
        df_sorted.loc[idx, hist_alias] = hist

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted

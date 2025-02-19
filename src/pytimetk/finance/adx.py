import pandas as pd
import polars as pl
import numpy as np

import pandas_flavor as pf
from typing import Union, List, Tuple, Optional

from pytimetk.utils.parallel_helpers import progress_apply
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_adx(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''Calculate Average Directional Index (ADX), +DI, and -DI for a financial time series to determine strength of trend.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or GroupBy object with time series data.
    date_column : str
        Column name containing dates or timestamps.
    high_column : str
        Column name with high prices.
    low_column : str
        Column name with low prices.
    close_column : str
        Column name with closing prices.
    periods : Union[int, Tuple[int, int], List[int]], optional
        Number of periods for ADX calculation. Accepts int, tuple (start, end), or list. Default is 14.
    reduce_memory : bool, optional
        If True, reduces memory usage before calculation. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - {close_column}_plus_di_{period}: Positive Directional Indicator (+DI)
        - {close_column}_minus_di_{period}: Negative Directional Indicator (-DI)
        - {close_column}_adx_{period}: Average Directional Index (ADX)
        
    Notes
    -----
    - The ADX is a trend strength indicator that ranges from 0 to 100.
    - A high ADX value indicates a strong trend, while a low ADX value indicates a weak trend.
    - The +DI and -DI values range from 0 to 100.
    - The ADX is calculated as the average of the DX values over the specified period.
    - The DX value is calculated as 100 * |(+DI - -DI)| / (+DI + -DI).
    - The True Range (TR) is the maximum of the following:
        - High - Low
        - High - Previous Close
        - Low - Previous Close
    - The +DM is calculated as follows:
        - If High - Previous High > Previous Low - Low, then +DM = max(High - Previous High, 0)
        - Otherwise, +DM = 0
    - The -DM is calculated as follows:
        - If Previous Low - Low > High - Previous High, then -DM = max(Previous Low - Low, 0)
        - Otherwise, -DM = 0
    
    References: 
    
    - https://www.investopedia.com/terms/a/adx.asp
        
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock ADX with pandas engine
    adx_df = (
        df.query("symbol == 'AAPL'")
        .augment_adx(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            periods=[14, 28]
        )
    )
    adx_df.head()
    ```
    
    ```{python}
    # Example 2 - Multiple stocks with groupby using pandas engine
    adx_df = (
        df.groupby('symbol')
        .augment_adx(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            periods=14
        )
    )
    adx_df.groupby('symbol').tail(1)
    ```
    
    ```{python}
    # Example 3 - Single stock ADX with polars engine
    adx_df = (
        df.query("symbol == 'AAPL'")
        .augment_adx(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            periods=[14, 28],
            engine='polars'
        )
    )
    adx_df.head()
    ```
    
    ```{python}
    # Example 4 - Multiple stocks with groupby using polars engine
    adx_df = (
        df.groupby('symbol')
        .augment_adx(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            periods=14,
            engine='polars'
        )
    )
    adx_df.groupby('symbol').tail(1)
    ```
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_adx_pandas(data, date_column, high_column, low_column, close_column, periods)
    elif engine == 'polars':
        ret = _augment_adx_polars(data, date_column, high_column, low_column, close_column, periods)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    ret = ret.sort_index()
    
    return ret

# Monkey patch to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_adx = augment_adx

def _augment_adx_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int]
) -> pd.DataFrame:
    """Pandas implementation of ADX calculation."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    
    # Calculate True Range (TR) and Directional Movement (+DM, -DM)
    df['tr'] = pd.concat([
        df[high_column] - df[low_column],
        (df[high_column] - df[close_column].shift(1)).abs(),
        (df[low_column] - df[close_column].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    df['plus_dm'] = np.where(
        (df[high_column] - df[high_column].shift(1)) > (df[low_column].shift(1) - df[low_column]),
        np.maximum(df[high_column] - df[high_column].shift(1), 0),
        0
    )
    df['minus_dm'] = np.where(
        (df[low_column].shift(1) - df[low_column]) > (df[high_column] - df[high_column].shift(1)),
        np.maximum(df[low_column].shift(1) - df[low_column], 0),
        0
    )
    
    for period in periods:
        if group_names:
            tr_smooth = df.groupby(group_names)['tr'].rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
            plus_dm_smooth = df.groupby(group_names)['plus_dm'].rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
            minus_dm_smooth = df.groupby(group_names)['minus_dm'].rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
        else:
            tr_smooth = df['tr'].rolling(window=period, min_periods=1).mean()
            plus_dm_smooth = df['plus_dm'].rolling(window=period, min_periods=1).mean()
            minus_dm_smooth = df['minus_dm'].rolling(window=period, min_periods=1).mean()
        
        df[f'{col}_plus_di_{period}'] = 100 * plus_dm_smooth / tr_smooth
        df[f'{col}_minus_di_{period}'] = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate DX and ADX with proper warmup
        dx = 100 * np.abs(plus_dm_smooth - minus_dm_smooth) / (plus_dm_smooth + minus_dm_smooth)
        df[f'dx_{period}'] = dx  # Temporary column
        if group_names:
            df[f'{col}_adx_{period}'] = df.groupby(group_names)[f'dx_{period}'].rolling(window=period, min_periods=period).mean().reset_index(level=0, drop=True)
        else:
            df[f'{col}_adx_{period}'] = df[f'dx_{period}'].rolling(window=period, min_periods=period).mean()

    # Drop temporary columns
    df = df.drop(columns=['tr', 'plus_dm', 'minus_dm'] + [f'dx_{p}' for p in periods])
    
    return df


def _augment_adx_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int]
) -> pd.DataFrame:
    """Polars implementation of ADX calculation."""
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
    else:
        pandas_df = data.copy()
        group_names = None
    
    df = pl.from_pandas(pandas_df)
    col = close_column
    
    # Calculate True Range (TR) and Directional Movement (+DM, -DM)
    df = df.with_columns([
        pl.max_horizontal([
            (pl.col(high_column) - pl.col(low_column)),
            (pl.col(high_column) - pl.col(close_column).shift(1)).abs(),
            (pl.col(low_column) - pl.col(close_column).shift(1)).abs()
        ]).alias('tr'),
        pl.when((pl.col(high_column) - pl.col(high_column).shift(1)) > (pl.col(low_column).shift(1) - pl.col(low_column)))
              .then(pl.col(high_column) - pl.col(high_column).shift(1))
              .otherwise(0)
              .clip(lower_bound=0)  # Fixed to use lower_bound
              .alias('plus_dm'),
        pl.when((pl.col(low_column).shift(1) - pl.col(low_column)) > (pl.col(high_column) - pl.col(high_column).shift(1)))
              .then(pl.col(low_column).shift(1) - pl.col(low_column))
              .otherwise(0)
              .clip(lower_bound=0)  # Fixed to use lower_bound
              .alias('minus_dm')
    ])
    
    for period in periods:
        if group_names:
            df = df.with_columns([
                pl.col('tr').rolling_mean(window_size=period, min_periods=1).over(group_names).alias('tr_smooth'),
                pl.col('plus_dm').rolling_mean(window_size=period, min_periods=1).over(group_names).alias('plus_dm_smooth'),
                pl.col('minus_dm').rolling_mean(window_size=period, min_periods=1).over(group_names).alias('minus_dm_smooth')
            ])
        else:
            df = df.with_columns([
                pl.col('tr').rolling_mean(window_size=period, min_periods=1).alias('tr_smooth'),
                pl.col('plus_dm').rolling_mean(window_size=period, min_periods=1).alias('plus_dm_smooth'),
                pl.col('minus_dm').rolling_mean(window_size=period, min_periods=1).alias('minus_dm_smooth')
            ])
        
        df = df.with_columns([
            (100 * pl.col('plus_dm_smooth') / pl.col('tr_smooth')).alias(f'{col}_plus_di_{period}'),
            (100 * pl.col('minus_dm_smooth') / pl.col('tr_smooth')).alias(f'{col}_minus_di_{period}'),
            (100 * (pl.col('plus_dm_smooth') - pl.col('minus_dm_smooth')).abs() / 
             (pl.col('plus_dm_smooth') + pl.col('minus_dm_smooth'))).alias(f'dx_{period}')
        ])
        
        if group_names:
            df = df.with_columns(
                pl.col(f'dx_{period}').rolling_mean(window_size=period, min_periods=period).over(group_names).alias(f'{col}_adx_{period}')
            )
        else:
            df = df.with_columns(
                pl.col(f'dx_{period}').rolling_mean(window_size=period, min_periods=period).alias(f'{col}_adx_{period}')
            )
    
    # Drop temporary columns
    df = df.drop(['tr', 'plus_dm', 'minus_dm', 'tr_smooth', 'plus_dm_smooth', 'minus_dm_smooth'] + [f'dx_{p}' for p in periods])
    
    return df.to_pandas()
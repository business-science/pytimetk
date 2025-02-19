import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Tuple, Optional

import pandas_flavor as pf
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_hurst_exponent(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    window: Union[int, Tuple[int, int], List[int]] = 100,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''Calculate the Hurst Exponent on a rolling window for a financial time series. Used for detecting trends and mean-reversion.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or GroupBy object with time series data.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices to calculate the Hurst Exponent.
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window for Hurst Exponent calculation. Accepts int, tuple (start, end), or list. Default is 100.
    reduce_memory : bool, optional
        If True, reduces memory usage before calculation. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - {close_column}_hurst_{window}: Hurst Exponent for each window size
    
    Notes
    -----
    The Hurst Exponent measures the long-term memory of a time series:
    
    - H < 0.5: Mean-reverting behavior
    - H ≈ 0.5: Random walk (no persistence)
    - H > 0.5: Trending or persistent behavior
    Computed using a simplified R/S analysis over rolling windows.
    
    References:
    
    - https://en.wikipedia.org/wiki/Hurst_exponent
    
    Examples:
    ---------
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock Hurst Exponent with pandas engine
    hurst_df = (
        df.query("symbol == 'AAPL'")
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=[100, 200]
        )
    )
    hurst_df.glimpse()
    ```
    
    ``` {python}
    # Example 2 - Multiple stocks with groupby using pandas engine
    hurst_df = (
        df.groupby('symbol')
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=100
        )
    )
    hurst_df.glimpse()
    ```
    
    ``` {python}
    # Example 3 - Single stock Hurst Exponent with polars engine
    hurst_df = (
        df.query("symbol == 'AAPL'")
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=[100, 200],
            engine='polars'
        )
    )
    hurst_df.glimpse()
    ```
    
    ``` {python}
    # Example 4 - Multiple stocks with groupby using polars engine
    hurst_df = (
        df.groupby('symbol')
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=100,
            engine='polars'
        )
    )
    hurst_df.glimpse()
    ```
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    # Convert window to a list of windows
    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, tuple):
        windows = list(range(window[0], window[1] + 1))
    elif isinstance(window, list):
        windows = window
    else:
        raise TypeError(f"Invalid window specification: type: {type(window)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_hurst_exponent_pandas(data, date_column, close_column, windows)
    elif engine == 'polars':
        ret = _augment_hurst_exponent_polars(data, date_column, close_column, windows)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    ret = ret.sort_index()
    
    return ret


# Monkey patch to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_hurst_exponent = augment_hurst_exponent


def _augment_hurst_exponent_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int]
) -> pd.DataFrame:
    """Pandas implementation of Hurst Exponent calculation."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    
    def calculate_hurst(series, min_size=8):
        """Simplified R/S analysis for Hurst Exponent."""
        n = len(series)
        if n < min_size or np.all(series == series[0]):  # Too short or constant
            return np.nan
        
        # Mean-adjusted series
        mean = np.mean(series)
        y = series - mean
        z = np.cumsum(y)
        
        # Range (R) and Standard Deviation (S)
        r = np.max(z) - np.min(z)
        s = np.std(series)
        
        if s == 0 or r == 0:  # Avoid division by zero
            return np.nan
        
        # R/S ratio
        rs = r / s
        
        # Simplified H: log(R/S) / log(n)
        h = np.log(rs) / np.log(n)
        return h if 0 <= h <= 1 else np.nan  # Clamp to valid range
    
    for window in windows:
        if group_names:
            df[f'{col}_hurst_{window}'] = (
                df.groupby(group_names)[col]
                .rolling(window=window, min_periods=window)
                .apply(calculate_hurst, raw=True)
                .reset_index(level=0, drop=True)
            )
        else:
            df[f'{col}_hurst_{window}'] = (
                df[col]
                .rolling(window=window, min_periods=window)
                .apply(calculate_hurst, raw=True)
            )
    
    return df


def _augment_hurst_exponent_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int]
) -> pd.DataFrame:
    """Polars implementation of Hurst Exponent calculation."""
    
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
    
    def hurst_udf(series):
        """User-defined function for Hurst Exponent in Polars."""
        series = series.to_numpy()
        n = len(series)
        if n < 8 or np.all(series == series[0]):  # Minimum size or constant
            return np.nan
        
        mean = np.mean(series)
        y = series - mean
        z = np.cumsum(y)
        r = np.max(z) - np.min(z)
        s = np.std(series)
        
        if s == 0 or r == 0:
            return np.nan
        
        rs = r / s
        h = np.log(rs) / np.log(n)
        return h if 0 <= h <= 1 else np.nan
    
    for window in windows:
        if group_names:
            df = df.with_columns(
                pl.col(col)
                .rolling_map(
                    hurst_udf,
                    window_size=window,
                    min_periods=window
                )
                .over(group_names)
                .alias(f'{col}_hurst_{window}')
            )
        else:
            df = df.with_columns(
                pl.col(col)
                .rolling_map(
                    hurst_udf,
                    window_size=window,
                    min_periods=window
                )
                .alias(f'{col}_hurst_{window}')
            )
    
    return df.to_pandas()

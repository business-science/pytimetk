import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple, Optional

from pytimetk.utils.parallel_helpers import progress_apply
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_stochastic_oscillator(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: Union[int, Tuple[int, int], List[int]] = 14,
    d_periods: Union[int, List[int]] = 3,  # Updated to support list
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The `augment_stochastic_oscillator` function calculates the Stochastic Oscillator (%K and %D)
    for a financial instrument using either pandas or polars engine, and returns the augmented DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input data can be a pandas DataFrame or a pandas DataFrameGroupBy object containing
        the time series data for Stochastic Oscillator calculations.
    date_column : str
        The name of the column containing dates or timestamps.
    high_column : str
        The column containing high prices for the financial instrument.
    low_column : str
        The column containing low prices for the financial instrument.
    close_column : str
        The column containing closing prices for the financial instrument.
    k_periods : Union[int, Tuple[int, int], List[int]], optional
        The number of periods for calculating %K (fast stochastic). Can be an integer, a tuple of
        two integers (start and end periods), or a list of integers. Default is 14.
    d_periods : int, optional
        The number of periods for calculating %D (slow stochastic), typically a moving average of %K.
        Default is 3.
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame before calculation. Default is False.
    engine : str, optional
        The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame augmented with columns:
        - {close_column}_stoch_k_{k_period}: Stochastic Oscillator %K for each k_period
        - {close_column}_stoch_d_{k_period}_{d_period}: Stochastic Oscillator %D for each k_period
    
    Notes
    -----
    The Stochastic Oscillator is a momentum indicator that compares a security's closing price to its
    price range over a specific period, developed by George Lane. It consists of two lines:
    
    - %K: Measures the current close relative to the high-low range over k_periods.
    - %D: A moving average of %K over d_periods, smoothing the %K line.
    
    Key interpretations:
    
    - Values above 80 indicate overbought conditions, suggesting a potential price reversal downward.
    - Values below 20 indicate oversold conditions, suggesting a potential price reversal upward.
    - Crossovers of %K and %D can signal buy/sell opportunities.
    - Divergences between price and the oscillator can indicate trend reversals.
    
    Formula:
    
    - %K = 100 * (Close - Lowest Low in k_periods) / (Highest High in k_periods - Lowest Low in k_periods)
    - %D = Moving average of %K over d_periods
    
    References:
    
    - https://www.investopedia.com/terms/s/stochasticoscillator.asp
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock stochastic oscillator
    stoch_df = (
        df.query("symbol == 'AAPL'")
        .augment_stochastic_oscillator(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            k_periods=[14, 28],
            d_periods=3
        )
    )
    stoch_df.head()
    ```
    
    ``` {python}
    # Example 2 - Multiple stocks with groupby
    stoch_df = (
        df.groupby('symbol')
        .augment_stochastic_oscillator(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            k_periods=14,
            d_periods=3
        )
    )
    stoch_df.groupby('symbol').tail(1)
    ```
    
    ``` {python}
    # Example 3 - Polars engine for single stock
    stoch_df = (
        df.query("symbol == 'AAPL'")
        .augment_stochastic_oscillator(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            k_periods=[14, 28],
            d_periods=3,
            engine='polars'
        )
    )
    stoch_df.head()
    ```
    
    ``` {python}
    # Example 4 - Polars engine with groupby
    stoch_df = (
        df.groupby('symbol')
        .augment_stochastic_oscillator(
            date_column='date',
            high_column='high',
            low_column='low',
            close_column='close',
            k_periods=14,
            d_periods=3,
            engine='polars'
        )
    )
    stoch_df.groupby('symbol').tail(1)
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    # Handle k_periods
    if isinstance(k_periods, int):
        k_periods = [k_periods]
    elif isinstance(k_periods, tuple):
        k_periods = list(range(k_periods[0], k_periods[1] + 1))
    elif not isinstance(k_periods, list):
        raise TypeError(f"Invalid k_periods specification: type: {type(k_periods)}. Please use int, tuple, or list.")
    
    # Handle d_periods
    if isinstance(d_periods, int):
        d_periods = [d_periods]  # Convert to list for consistency
    elif not isinstance(d_periods, list):
        raise TypeError(f"Invalid d_periods specification: type: {type(d_periods)}. Please use int or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_stochastic_oscillator_pandas(data, date_column, high_column, low_column, close_column, k_periods, d_periods)
    elif engine == 'polars':
        ret = _augment_stochastic_oscillator_polars(data, date_column, high_column, low_column, close_column, k_periods, d_periods)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_stochastic_oscillator = augment_stochastic_oscillator

def _augment_stochastic_oscillator_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: List[int],
    d_periods: List[int]  # Updated to List[int]
) -> pd.DataFrame:
    """Pandas implementation of Stochastic Oscillator calculation."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    
    for k_period in k_periods:
        if group_names:
            # Grouped calculation for %K
            lowest_low = df.groupby(group_names)[low_column].rolling(window=k_period, min_periods=1).min().reset_index(level=0, drop=True)
            highest_high = df.groupby(group_names)[high_column].rolling(window=k_period, min_periods=1).max().reset_index(level=0, drop=True)
            df[f'{col}_stoch_k_{k_period}'] = 100 * (df[col] - lowest_low) / (highest_high - lowest_low)
            # Calculate %D for each d_period
            for d_period in d_periods:
                df[f'{col}_stoch_d_{k_period}_{d_period}'] = df.groupby(group_names)[f'{col}_stoch_k_{k_period}'].rolling(
                    window=d_period, min_periods=1
                ).mean().reset_index(level=0, drop=True)
        else:
            # Non-grouped calculation for %K
            lowest_low = df[low_column].rolling(window=k_period, min_periods=1).min()
            highest_high = df[high_column].rolling(window=k_period, min_periods=1).max()
            df[f'{col}_stoch_k_{k_period}'] = 100 * (df[col] - lowest_low) / (highest_high - lowest_low)
            # Calculate %D for each d_period
            for d_period in d_periods:
                df[f'{col}_stoch_d_{k_period}_{d_period}'] = df[f'{col}_stoch_k_{k_period}'].rolling(
                    window=d_period, min_periods=1
                ).mean()
    
    return df


def _augment_stochastic_oscillator_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    k_periods: List[int],
    d_periods: List[int]  # Updated to List[int]
) -> pd.DataFrame:
    """Polars implementation of Stochastic Oscillator calculation."""
    
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
    
    if group_names:
        # Step 1: Calculate all %K columns
        k_exprs = []
        for k_period in k_periods:
            lowest_low = pl.col(low_column).rolling_min(window_size=k_period).over(group_names)
            highest_high = pl.col(high_column).rolling_max(window_size=k_period).over(group_names)
            k_expr = (100 * (pl.col(col) - lowest_low) / (highest_high - lowest_low)).alias(f'{col}_stoch_k_{k_period}')
            k_exprs.append(k_expr)
        df = df.with_columns(k_exprs)
        
        # Step 2: Calculate all %D columns for each %K and d_period
        d_exprs = []
        for k_period in k_periods:
            for d_period in d_periods:
                d_expr = pl.col(f'{col}_stoch_k_{k_period}').rolling_mean(window_size=d_period).over(group_names).alias(f'{col}_stoch_d_{k_period}_{d_period}')
                d_exprs.append(d_expr)
        df = df.with_columns(d_exprs)
    else:
        # Step 1: Calculate all %K columns
        k_exprs = []
        for k_period in k_periods:
            lowest_low = pl.col(low_column).rolling_min(window_size=k_period)
            highest_high = pl.col(high_column).rolling_max(window_size=k_period)
            k_expr = (100 * (pl.col(col) - lowest_low) / (highest_high - lowest_low)).alias(f'{col}_stoch_k_{k_period}')
            k_exprs.append(k_expr)
        df = df.with_columns(k_exprs)
        
        # Step 2: Calculate all %D columns for each %K and d_period
        d_exprs = []
        for k_period in k_periods:
            for d_period in d_periods:
                d_expr = pl.col(f'{col}_stoch_k_{k_period}').rolling_mean(window_size=d_period).alias(f'{col}_stoch_d_{k_period}_{d_period}')
                d_exprs.append(d_expr)
        df = df.with_columns(d_exprs)
    
    return df.to_pandas()
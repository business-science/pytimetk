
import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_rsi(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The `augment_rsi` function calculates the Relative Strength Index (RSI) for a given financial
    instrument using either pandas or polars engine, and returns the augmented DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
        DataFrameGroupBy object. It contains the data on which the RSI will be
        calculated.
    date_column : str
        The name of the column in the data that contains the dates or timestamps.
    close_column : str
        The `close_column` parameter is used to specify the column(s) in the input data that contain the
        values on which the RSI will be calculated. It can be either a single column name (string) or a list
        of column names (if you want to calculate RSI on multiple columns).
    periods : Union[int, Tuple[int, int], List[int]], optional
        The `periods` parameter in the `augment_rsi` function specifies the number of rolling periods over which
        the RSI is calculated. It can be provided as an integer, a tuple of two
        integers (start and end periods), or a list of integers.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
        usage of the data before performing the RSI calculation. If set to `True`, the function will attempt
        to reduce the memory usage of the input data. If set to `False`, the function will not attempt to reduce the memory usage of the input data.
    engine : str, optional
        The `engine` parameter specifies the computation engine to use for calculating the RSI. It can take two values: 'pandas' or 'polars'.
    
    Returns
    -------
    pd.DataFrame
        The function `augment_rsi` returns a pandas DataFrame that contains the augmented data with the
        Relative Strength Index (RSI) values added.
        
    Notes
    -----
    The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. Developed by J. Welles Wilder Jr. and introduced in his 1978 book "New Concepts in Technical Trading Systems", the RSI is one of the most well-known and widely used technical analysis indicators.
    
    - Range: The RSI oscillates between 0 and 100.
    - Overbought and Oversold Levels: Traditionally, the RSI is 
    considered overbought when above 70 and oversold when below
    30. These thresholds can indicate potential reversal points 
    where a security is overvalued or undervalued.
    - Divergence: RSI can also be used to identify potential
    reversals by looking for bearish and bullish divergences.  
    
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])
    df
    
    # Example 1 - Calculate RSI for a single column
    rsi_df = (
        df
            .query("symbol == 'AAPL'")
            .augment_rsi(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28]
            )
    )
    rsi_df
    ```
    
    ``` {python}
    # Example 2 - Calculate RSI for multiple groups
    rsi_df = (
        df
            .groupby('symbol')
            .augment_rsi(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28]
            )
    )
    rsi_df.groupby('symbol').tail(1)
    
    ```
    
    ```{python}
    # Example 3 - Calculate RSI for polars engine
    rsi_df = (
        df
            .query("symbol == 'AAPL'")
            .augment_rsi(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28],
                engine='polars'
            )
    )
    rsi_df
    ```
    
    ```{python}
    # Example 4 - Calculate RSI for polars engine and groups
    rsi_df = (
        df
            .groupby('symbol')
            .augment_rsi(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28],
                engine='polars'
            )
    )
    rsi_df.groupby('symbol').tail(1)
    ```
    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    if isinstance(close_column, str):
        close_column = [close_column]
        
    if isinstance(periods, int):
        periods = [periods]  # Convert to a list with a single value
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_rsi_pandas(data, date_column, close_column, periods)
    elif engine == 'polars':
        ret = _augment_rsi_polars(data, date_column, close_column, periods)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rsi = augment_rsi


def _augment_rsi_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14
) -> pd.DataFrame:

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, apply RSI function
    if isinstance(data, pd.DataFrame):

        df = data.copy()
        
        for col in close_column:
            for period in periods:
                df[f'{col}_rsi_{period}'] = _calculate_rsi_pandas(df[col], period=period)
    
    # GROUPBY EXTENSION - If data is a Pandas GroupBy object, apply RSI function BY GROUP
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()
        
        for col in close_column:   
            for period in periods:
                df[f'{col}_rsi_{period}'] = df.groupby(group_names, group_keys=False)[col].apply(_calculate_rsi_pandas, period=period)
    
    return df


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


def _augment_rsi_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14
) -> pd.DataFrame:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data.copy()
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        def apply_rsi(pl_df):
            for col in close_column:
                for period in periods:
                    pl_df = pl_df.with_columns(
                        _calculate_rsi_polars(pl_df[col], period=period).alias(f'{col}_rsi_{period}')
                    )
            return pl_df
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.group_by(
            *group_names, maintain_order=True
        ).map_groups(apply_rsi)
        
        df = out_df

    else:
        
        _exprs = []
        
        for col in close_column:
            for period in periods:
                _expr = _calculate_rsi_polars(pl.col(col), period=period).alias(f'{col}_rsi_{period}')
                _exprs.append(_expr)
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.select(_exprs)
        
        df = pl.concat([df, out_df], how="horizontal")
    
    return df.to_pandas()
    
    

def _calculate_rsi_polars(series: pl.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()
    
    # Separate gains and losses        
    gains = pl.when(delta > 0).then(delta).otherwise(0)
    losses = pl.when(delta <= 0).then(-delta).otherwise(0)

    # Calculate the sum of gains and losses using a rolling window
    mean_gains = gains.rolling_mean(window_size=period)
    mean_losses = losses.rolling_mean(window_size=period)

    # Calculate RSI
    ret = 100 - (100 / (1 + (mean_gains / mean_losses)))
    
    return ret


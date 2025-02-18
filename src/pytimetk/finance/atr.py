import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe



@pf.register_dataframe_method
def augment_atr(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    normalize: bool = False,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The `augment_atr` function is used to calculate Average True Range (ATR) and 
    Normalized Average True Range (NATR) for a given dataset and return
    the augmented dataset. 
    Set the `normalize` parameter to `True` to calculate NATR.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
        DataFrameGroupBy object. It contains the data on which the Bollinger Bands will be calculated.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the dates.
    high_column : str
        The `high_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the high prices of the asset.
    low_column : str
        The `low_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the low prices of the asset.
    close_column : str
        The `close_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the closing prices of the asset.
    periods : Union[int, Tuple[int, int], List[int]], optional
        The `periods` parameter in the `augment_atr` function can be specified as an integer, a tuple,
        or a list. This parameter specifies the number of rolling periods to use when calculating the ATR.
    normalize : bool, optional
        The `normalize` parameter is a boolean flag that indicates whether or not to normalize the ATR
        values. If set to `True`, the function will normalize the ATR values to express this volatility as a percentage of 
        the closing price.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
        usage of the input data before performing the calculation. If set to `True`, the function will
        attempt to reduce the memory usage of the input data using techniques such as downcasting numeric
        columns and converting object columns
    engine : str, optional
        The `engine` parameter specifies the computation engine to use for calculating the Bollinger Bands.
        It can take two values: 'pandas' or 'polars'. If 'pandas' is selected, the function will use the
        pandas library for computation. If 'polars' is selected,
    
    Returns
    -------
    pd.DataFrame
        The function `augment_atr` returns a pandas DataFrame.
        
    Notes
    -----
    
    ## ATR (Average True Range)
    
    The Average True Range (ATR) is a technical analysis indicator used to measure market volatility. It was introduced by J. Welles Wilder Jr. in his 1978 book "New Concepts in Technical Trading Systems."  
    
    The ATR is calculated as follows:

    1. True Range: For each period (typically a day), the True Range is the greatest of the following:

        - The current high minus the current low.
        - The absolute value of the current high minus the previous close.
        - The absolute value of the current low minus the previous close.
        
    2. Average True Range: The ATR is an average of the True Range over a specified number of periods (commonly 14 days).
    
    ## NATR (Normalized Average True Range)
    
    The NATR (Normalized Average True Range) is a variation of the ATR that normalizes the ATR values to express this volatility as a percentage of the closing price.
    
    The NATR (`normalize = True`) is calculated as follows:
    NATR = (ATR / Close) * 100


    Examples
    --------
    
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
    
    df
    ```
    
    ``` {python}
    # ATR pandas engine
    df_atr = (
        df
            .groupby('symbol')
            .augment_atr(
                date_column = 'date', 
                high_column='high',
                low_column='low',
                close_column='close', 
                periods = [14, 28],
                normalize = False, # True for NATR
                engine = "pandas"
            )
    )
    
    df_atr.glimpse()
    ```
    
    ``` {python}
    # ATR polars engine
    df_atr = (
        df
            .groupby('symbol')
            .augment_atr(
                date_column = 'date', 
                high_column='high',
                low_column='low',
                close_column='close', 
                periods = [14, 28],
                normalize = False, # True for NATR
                engine = "polars"
            )
    )
    
    df_atr.glimpse()
    ```
    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_date_column(data, date_column)
    
    # Handle periods
    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    # Sort data and preserve index
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_atr_pandas(data, date_column, high_column, low_column, close_column, periods, normalize)
    elif engine == 'polars':
        ret = _augment_atr_polars(data, date_column, high_column, low_column, close_column, periods, normalize)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    return ret.sort_index()

# Monkey patch
pd.core.groupby.generic.DataFrameGroupBy.augment_atr = augment_atr
def _augment_atr_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str, 
    periods: List[int],
    normalize: bool
) -> pd.DataFrame:
    """Pandas implementation of ATR/NATR calculation."""
    type_str = 'natr' if normalize else 'atr'
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        # True Range calculation as a column
        df['tr'] = pd.concat([
            df[high_column] - df[low_column],
            (df[high_column] - df[close_column].shift(1)).abs(),
            (df[low_column] - df[close_column].shift(1)).abs()
        ], axis=1).max(axis=1)
        
        for period in periods:
            atr = df['tr'].rolling(window=period, min_periods=1).mean()
            if normalize:
                atr = (atr / df[close_column] * 100).replace([float('inf'), -float('inf')], pd.NA)
            df[f'{close_column}_{type_str}_{period}'] = atr
        
        df = df.drop(columns=['tr'])
            
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
        # True Range calculation with group-aware shift
        prev_close = df.groupby(group_names)[close_column].shift(1)
        df['tr'] = pd.concat([
            df[high_column] - df[low_column],
            (df[high_column] - prev_close).abs(),
            (df[low_column] - prev_close).abs()
        ], axis=1).max(axis=1)
        
        for period in periods:
            atr = df.groupby(group_names)['tr'].rolling(window=period, min_periods=1).mean().reset_index(level=0, drop=True)
            if normalize:
                atr = (atr / df[close_column] * 100).replace([float('inf'), -float('inf')], pd.NA)
            df[f'{close_column}_{type_str}_{period}'] = atr
        
        df = df.drop(columns=['tr'])
    
    return df

def _augment_atr_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str, 
    periods: List[int],
    normalize: bool
) -> pd.DataFrame:
    """Polars implementation of ATR/NATR calculation."""
    type_str = 'natr' if normalize else 'atr'
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        df = pl.from_pandas(data.obj)
        group_names = data.grouper.names if isinstance(data.grouper.names, list) else [data.grouper.names]
    else:
        df = pl.from_pandas(data.copy())
        group_names = None
    
    # True Range calculation
    tr = pl.max_horizontal([
        pl.col(high_column) - pl.col(low_column),
        (pl.col(high_column) - pl.col(close_column).shift(1)).abs(),
        (pl.col(low_column) - pl.col(close_column).shift(1)).abs()
    ])
    
    # Add ATR/NATR columns
    for period in periods:
        atr = tr.rolling_mean(window_size=period, min_periods=1).over(group_names if group_names else None)
        if normalize:
            atr = (atr / pl.col(close_column) * 100).replace([float('inf'), -float('inf')], None)
        df = df.with_columns(atr.alias(f'{close_column}_{type_str}_{period}'))
    
    return df.to_pandas()

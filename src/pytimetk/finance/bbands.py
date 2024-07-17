import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe



@pf.register_dataframe_method
def augment_bbands(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    std_dev: Union[int, float, List[float]] = 2,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The `augment_bbands` function is used to calculate Bollinger Bands for a given dataset and return
    the augmented dataset.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
        DataFrameGroupBy object. It contains the data on which the Bollinger Bands will be calculated.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the dates.
    close_column : str
        The `close_column` parameter is a string that specifies the name of the column in the `data`
        DataFrame that contains the closing prices of the asset.
    periods : Union[int, Tuple[int, int], List[int]], optional
        The `periods` parameter in the `augment_bbands` function can be specified as an integer, a tuple,
        or a list. This parameter specifies the number of rolling periods to use when calculating the Bollinger Bands.
    std_dev : float, optional
        The `std_dev` parameter is a float that represents the number of standard deviations to use
        when calculating the Bollinger Bands. Bollinger Bands are a technical analysis tool that consists of
        a middle band (usually a simple moving average) and an upper and lower band that are typically two
        standard deviations away from the middle band. The `std_dev` parameter specifies the number of standard deviations. `std_dev` can be a list of floats as well.
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
        The function `augment_bbands` returns a pandas DataFrame.
        
    Notes
    -----
    
    Bollinger Bands are a technical analysis tool developed by John
    Bollinger in the 1980s. They are used to measure the 
    'volatility' of a stock price or other financial instrument. 
    This indicator consists of three lines which are plotted in 
    relation to an asset's price:
    
    1. The Middle Band: This is typically a simple moving average 
    (SMA) of the closing prices over a certain number of days 
    (commonly 20 days).

    2. The Upper Band: This is set a specified number of standard
    deviations (usually two) above the middle band.
    
    3. The Lower Band: This is set the same number of standard 
    deviations (again, usually two) below the middle band.
    
    Volatility Indicator: The width of the bands is a measure of 
    volatility. When the bands widen, it indicates increased 
    volatility, and when they contract, it suggests decreased 
    volatility.
    
    Overbought and Oversold Conditions: Prices are considered 
    overbought near the upper band and oversold near the lower
    band. However, these conditions do not necessarily signal a 
    reversal; prices can remain overbought or oversold for extended
    periods during strong trends.
    
    
    Examples
    --------
    
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
    
    df
    ```
    
    ``` {python}
    # BBANDS pandas engine
    df_bbands = (
        df
            .groupby('symbol')
            .augment_bbands(
                date_column = 'date', 
                close_column='close', 
                periods = [20, 40],
                std_dev = 2, 
                engine = "pandas"
            )
    )
    
    df_bbands.glimpse()
    ```
    
    ``` {python}
    # BBANDS polars engine
    df_bbands = (
        df
            .groupby('symbol')
            .augment_bbands(
                date_column = 'date', 
                close_column='close', 
                periods = [20, 40],
                std_dev = 2, 
                engine = "polars"
            )
    )
    
    df_bbands.glimpse()
    ```
    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)

    if isinstance(periods, int):
        periods = [periods]
        
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
        
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
      
    if isinstance(std_dev, float):
        std_dev = [std_dev]
    elif isinstance(std_dev, int):
        std_dev = [float(std_dev)]
    elif isinstance(std_dev, list):
        std_dev = [float(i) for i in std_dev]
    elif not isinstance(std_dev, list):
        raise TypeError(f"Invalid std_dev specification: type: {type(std_dev)}. Please use float or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_bbands_pandas(data, date_column, close_column, periods, std_dev)
    elif engine == 'polars':
        ret = _augment_bbands_polars(data, date_column, close_column, periods, std_dev)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_bbands = augment_bbands


def _augment_bbands_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    std_dev: Union[float, List[float]] = 2.0
) -> pd.DataFrame:
    """
    Internal function to calculate BBANDS using Pandas.
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        data = data.obj
        df = data.copy()
        
        for period in periods:
            
            for sd in std_dev:
            
                ma = df.groupby(group_names)[close_column].rolling(period).mean().reset_index(level = 0, drop = True)
                
                std = df.groupby(group_names)[close_column].rolling(period).std().reset_index(level = 0, drop = True)
                
                # Add upper and lower bband columns
                df[f'{close_column}_bband_middle_{period}_{sd}'] = ma
                
                df[f'{close_column}_bband_upper_{period}_{sd}'] = ma + (std * sd)
                
                df[f'{close_column}_bband_lower_{period}_{sd}'] = ma - (std * sd)
        

    elif isinstance(data, pd.DataFrame):
        
        df = data.copy()
        
        for period in periods:
            
            for sd in std_dev:
            
                ma = df[close_column].rolling(period).mean()
                
                std = df[close_column].rolling(period).std()
                
                # Add upper and lower bband columns
                df[f'{close_column}_bband_middle_{period}_{sd}'] = ma
                
                df[f'{close_column}_bband_upper_{period}_{sd}'] = ma + (std * sd)
                
                df[f'{close_column}_bband_lower_{period}_{sd}'] = ma - (std * sd)
            
    else:
        raise ValueError("data must be a pandas DataFrame or a pandas GroupBy object")

    return df

def _augment_bbands_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    std_dev: Union[float, List[float]] = 2.0
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
                
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        
        pl_df = pl.from_pandas(pandas_df)
        
        for period in periods:
            for sd in std_dev:
            
                ma = pl.col(close_column).rolling_mean(window_size=period).over(group_names).alias(f'{close_column}_bband_middle_{period}_{sd}')
                
                std = pl.col(close_column).rolling_std(window_size=period).over(group_names).alias('std')
                
                # Add upper and lower bband columns     
                upper_band = (ma + std * sd).alias(f'{close_column}_bband_upper_{period}_{sd}')
                
                lower_band = (ma - std * sd).alias(f'{close_column}_bband_lower_{period}_{sd}')   
                
                pl_df = pl_df.with_columns([ma, upper_band, lower_band]) 
        
    else:
        
        pl_df = pl.from_pandas(pandas_df)
        
        for period in periods:
            for sd in std_dev:
            
                ma = pl.col(close_column).rolling_mean(window_size=period).alias(f'{close_column}_bband_middle_{period}_{sd}')
                
                std = pl.col(close_column).rolling_std(window_size=period).alias('std')
            
                # Add upper and lower bband columns     
                upper_band = (ma + std * sd).alias(f'{close_column}_bband_upper_{period}_{sd}')
                
                lower_band = (ma - std * sd).alias(f'{close_column}_bband_lower_{period}_{sd}')   
                
                pl_df = pl_df.with_columns([ma, upper_band, lower_band])   
    
    return pl_df.to_pandas()
    
    




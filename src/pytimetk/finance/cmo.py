
import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.parallel_helpers import progress_apply

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_cmo(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The `augment_cmo` function calculates the Chande Momentum Oscillator (CMO) for a given financial
    instrument using either pandas or polars engine, and returns the augmented DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
        DataFrameGroupBy object. It contains the data on which the Chande Momentum  Oscillator (CMO) will be
        calculated.
    date_column : str
        The name of the column in the data that contains the dates or timestamps.
    close_column : str
        The `close_column` parameter is used to specify the column in the input data that contain the
        values on which the CMO will be calculated. 
    periods : Union[int, Tuple[int, int], List[int]], optional
        The `periods` parameter in the `augment_cmo` function specifies the number of rolling periods over which
        the Chande Momentum Oscillator (CMO) is calculated. It can be provided as an integer, a tuple of two
        integers (start and end periods), or a list of integers.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
        usage of the data before performing the CMO calculation. If set to `True`, the function will attempt
        to reduce the memory usage of the input data. If set to `False`, the function will not attempt to reduce the memory usage of the input data.
    engine : str, optional
        The `engine` parameter specifies the computation engine to use for calculating the Chande Momentum
        Oscillator (CMO). It can take two values: 'pandas' or 'polars'.
    
    Returns
    -------
    pd.DataFrame
        The function `augment_cmo` returns a pandas DataFrame that contains the augmented data with the
        Chande Momentum Oscillator (CMO) values added.
        
    Notes
    -----
    The Chande Momentum Oscillator (CMO), developed by Tushar Chande, is a technical analysis tool used to gauge the momentum of a financial instrument. It is similar to other momentum indicators like the Relative Strength Index (RSI), but with some distinct characteristics. Here's what the CMO tells us:

    Momentum of Price Movements:

    The CMO measures the strength of trends in price movements. It calculates the difference between the sum of gains and losses over a specified period, normalized to oscillate between -100 and +100.
    Overbought and Oversold Conditions:

    Values close to +100 suggest overbought conditions, indicating that the price might be too high and could reverse.
    Conversely, values near -100 suggest oversold conditions, implying that the price might be too low and could rebound.
    Trend Strength:

    High absolute values (either positive or negative) indicate strong trends, while values near zero suggest a lack of trend or a weak trend.
    Divergences:

    Divergences between the CMO and price movements can be significant. For example, if the price is making new highs but the CMO is declining, it may indicate weakening momentum and a potential trend reversal.
    Crossing the Zero Line:

    When the CMO crosses above zero, it can be seen as a bullish signal, whereas a cross below zero can be interpreted as bearish.
    Customization:

    The period over which the CMO is calculated can be adjusted. A shorter period makes the oscillator more sensitive to price changes, suitable for short-term trading. A longer period smooths out the oscillator for a longer-term perspective.
    It's important to note that while the CMO can provide valuable insights into market momentum and potential price reversals, it is most effective when used in conjunction with other indicators and analysis methods. Like all technical indicators, the CMO should not be used in isolation but rather as part of a comprehensive trading strategy.

    References:
    1. https://www.fmlabs.com/reference/default.htm?url=CMO.htm
    
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])
    df
    
    # Example 1 - Calculate CMO for a single column
    cmo_df = (
        df
            .query("symbol == 'AAPL'")
            .augment_cmo(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28]
            )
    )
    cmo_df
    ```
    
    ``` {python}
    # Example 2 - Calculate CMO for multiple groups
    cmo_df = (
        df
            .groupby('symbol')
            .augment_cmo(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28]
            )
    )
    cmo_df.groupby('symbol').tail(1)
    
    ```
    
    ```{python}
    # Example 3 - Calculate CMO for polars engine
    cmo_df = (
        df
            .query("symbol == 'AAPL'")
            .augment_cmo(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28],
                engine='polars'
            )
    )
    cmo_df
    ```
    
    ```{python}
    # Example 4 - Calculate CMO for polars engine and groups
    cmo_df = (
        df
            .groupby('symbol')
            .augment_cmo(
                date_column='date',
                close_column='adjusted',
                periods=[14, 28],
                engine='polars'
            )
    )
    cmo_df.groupby('symbol').tail(1)
    ```
    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
        
    if isinstance(periods, int):
        periods = [periods]  # Convert to a list with a single value
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_cmo_pandas(data, date_column, close_column, periods)
    elif engine == 'polars':
        ret = _augment_cmo_polars(data, date_column, close_column, periods)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_cmo = augment_cmo


def _augment_cmo_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14
) -> pd.DataFrame:

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, apply cmo function
    if isinstance(data, pd.DataFrame):

        df = data.copy()
        col = close_column

        for period in periods:
            df[f'{col}_cmo_{period}'] = _calculate_cmo_pandas(df[col], period=period)
    
    # GROUPBY EXTENSION - If data is a Pandas GroupBy object, apply cmo function BY GROUP
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()
   
        col = close_column
        
        for period in periods:
            df[f'{col}_cmo_{period}'] = df.groupby(group_names, group_keys=False)[col].apply(_calculate_cmo_pandas, period=period)

    return df


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
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 14
) -> pd.DataFrame:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data.copy()
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        def apply_cmo(pl_df):
            
            col = close_column
            
            for period in periods:
                pl_df = pl_df.with_columns(
                    _calculate_cmo_polars(pl_df[col], period=period).alias(f'{col}_cmo_{period}')
                )
                
            return pl_df
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.group_by(
            *group_names, maintain_order=True
        ).map_groups(apply_cmo)
        
        df = out_df

    else:
        
        _exprs = []
        
        col = close_column
        
        for period in periods:
            _expr = _calculate_cmo_polars(pl.col(col), period=period).alias(f'{col}_cmo_{period}')
            _exprs.append(_expr)
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.select(_exprs)
        
        df = pl.concat([df, out_df], how="horizontal")
    
    return df.to_pandas()
    
    

def _calculate_cmo_polars(series: pl.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()
    
    # Separate gains and losses    
    gains = pl.when(delta > 0).then(delta).otherwise(0)
    losses = pl.when(delta <= 0).then(-delta).otherwise(0)

    # Calculate the sum of gains and losses using a rolling window
    sum_gains = gains.rolling_sum(window_size=period)
    sum_losses = losses.rolling_sum(window_size=period)

    # Calculate CMO
    total_movement = sum_gains + sum_losses
    cmo = 100 * (sum_gains - sum_losses) / total_movement
    
    return cmo


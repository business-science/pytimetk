import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_ppo(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    fast_period: int = 12,
    slow_period: int = 26,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    """
    Calculate PPO for a given financial instrument using either pandas or polars engine.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Pandas DataFrame or GroupBy object containing financial data.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Name of the column containing closing price data.
    fast_period : int, optional
        Number of periods for the fast EMA in PPO calculation.
    slow_period : int, optional
        Number of periods for the slow EMA in PPO calculation.
    reduce_memory : bool, optional
        Whether to reduce memory usage of the data before performing the calculation.
    engine : str, optional
        Computation engine to use ('pandas' or 'polars').
        
        
    Returns
    -------
    pd.DataFrame
        DataFrame with PPO values added.
         
    Notes
    -----
    
    The Percentage Price Oscillator (PPO) is a momentum oscillator 
    that measures the difference between two moving averages as a 
    percentage of the larger moving average. The PPO is best used
    to confirm the direction of the price trend and gauge its 
    momentum. 
    
    The PPO is calculated by subtracting a long-term EMA from a 
    short-term EMA, then dividing the result by the long-term EMA,
    and finally multiplying by 100.
    
    Advantages Over MACD: The PPO's percentage-based calculation 
    allows for easier comparisons between different securities, 
    regardless of their price levels. This is a distinct advantage
    over the MACD, which provides absolute values and can be less
    meaningful when comparing stocks with significantly different 
    prices.

        
    Examples
    --------
    
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
    
    df
    ```
    
    ``` {python}
    # PPO pandas engine
    df_ppo = (
        df
            .groupby('symbol')
            .augment_ppo(
                date_column = 'date', 
                close_column = 'close', 
                fast_period = 12, 
                slow_period = 26, 
                engine = "pandas"
            )
    )
    
    df_ppo.glimpse()
    ```
    
    ``` {python}
    # PPO polars engine
    df_ppo = (
        df
            .groupby('symbol')
            .augment_ppo(
                date_column = 'date', 
                close_column = 'close', 
                fast_period = 12, 
                slow_period = 26, 
                engine = "polars"
            )
    )
    
    df_ppo.glimpse()
    ```
    
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)

    if engine == 'pandas':
        ret = _augment_ppo_pandas(data, date_column, close_column, fast_period, slow_period)
    elif engine == 'polars':
        ret = _augment_ppo_polars(data, date_column, close_column, fast_period, slow_period)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()

    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_ppo = augment_ppo

def _augment_ppo_pandas(data, date_column, close_column, fast_period, slow_period):
    """
    Internal function to calculate PPO using Pandas.
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # If data is a GroupBy object, apply MACD calculation for each group
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()
        
        df = data.groupby(group_names, group_keys=False).apply(lambda x: _calculate_ppo_pandas(x, close_column, fast_period, slow_period))
    elif isinstance(data, pd.DataFrame):
        # If data is a DataFrame, apply PPO calculation directly
        df = data.copy().sort_values(by=date_column)
        df = _calculate_ppo_pandas(df, close_column, fast_period, slow_period)
    else:
        raise ValueError("data must be a pandas DataFrame or a pandas GroupBy object")

    return df

def _calculate_ppo_pandas(df, close_column, fast_period, slow_period):
    """
    Calculate PPO for a DataFrame.
    """
    # Calculate Fast and Slow EMAs
    ema_fast = df[close_column].ewm(span=fast_period, adjust=False, min_periods = 0).mean()
    ema_slow = df[close_column].ewm(span=slow_period, adjust=False, min_periods = 0).mean()

    # Calculate PPO Line
    ppo_line = (ema_fast - ema_slow) / ema_slow * 100

    # Add columns
    df[f'{close_column}_ppo_line_{fast_period}_{slow_period}'] = ppo_line

    return df

def _augment_ppo_polars(data, date_column, close_column, fast_period, slow_period):
    """
    Internal function to calculate PPO using Polars.
    """
    # Convert to Polars DataFrame if input is a pandas DataFrame or GroupBy object
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
        pandas_df = data.obj
        pl_df = pl.from_pandas(pandas_df)
    elif isinstance(data, pd.DataFrame):
        pl_df = pl.from_pandas(data.copy())
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    # Define the function to calculate PPO for a single DataFrame
    def calculate_ppo_single(pl_df):
        # Calculate Fast and Slow EMAs
        fast_ema = pl_df[close_column].ewm_mean(span=fast_period, adjust=False, min_periods=0)
        slow_ema = pl_df[close_column].ewm_mean(span=slow_period, adjust=False, min_periods=0)

        # Calculate PPO Line
        ppo_line = (fast_ema - slow_ema) / slow_ema * 100

        return pl_df.with_columns([
            ppo_line.alias(
                f'{close_column}_ppo_line_{fast_period}_{slow_period}'
            ),
        ])

    # Apply the calculation to each group if data is grouped, otherwise apply directly
    if 'groupby' in str(type(data)):
        result_df = pl_df.group_by(
            *group_names, maintain_order=True
        ).map_groups(calculate_ppo_single).to_pandas()
        
    else:
        result_df = calculate_ppo_single(pl_df).to_pandas()

    return result_df



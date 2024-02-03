import pandas as pd
import polars as pl

import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage



@pf.register_dataframe_method
def augment_bbands(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    num_std_dev: float = 2,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    if isinstance(periods, int):
        periods = [periods]
        
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
        
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_bbands_pandas(data, date_column, close_column, periods, num_std_dev)
    elif engine == 'polars':
        ret = _augment_bbands_polars(data, date_column, close_column, periods, num_std_dev)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_bbands = augment_bbands


def _augment_bbands_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: Union[str, List[str]], 
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    num_std_dev: float = 2
) -> pd.DataFrame:
    """
    Internal function to calculate BBANDS using Pandas.
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        data = data.obj
        df = data.copy()
        
        df.sort_values(by=[*group_names, date_column], inplace=True)
        
        for period in periods:
            
            ma = df.groupby(group_names)[close_column].rolling(period).mean().reset_index(drop = True).set_axis(df.index)
            
            df[f'{close_column}_bband_middle_{period}'] = ma
            
            std = df.groupby(group_names)[close_column].rolling(period).std().reset_index(drop = True).set_axis(df.index)
            
            df[f'{close_column}_bband_upper_{period}'] = ma + (std * num_std_dev)
            
            df[f'{close_column}_bband_lower_{period}'] = ma - (std * num_std_dev)
        

    elif isinstance(data, pd.DataFrame):
        
        df = data.copy().sort_values(by=date_column)
        
        for period in periods:
            
            ma = df[close_column].rolling(period).mean().reset_index(drop = True).set_axis(df.index)
            
            std = df[close_column].rolling(period).std().reset_index(drop = True).set_axis(df.index)
            
            df[f'{close_column}_bband_middle_{period}'] = ma
            
            df[f'{close_column}_bband_upper_{period}'] = ma + (std * num_std_dev)
            
            df[f'{close_column}_bband_lower_{period}'] = ma - (std * num_std_dev)
            
    else:
        raise ValueError("data must be a pandas DataFrame or a pandas GroupBy object")

    return df




def _augment_bbands_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: Union[str, List[str]], 
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
        
        def apply_cmo(pl_df):
            for col in close_column:
                for period in periods:
                    pl_df = pl_df.with_columns(
                        _calculate_cmo_polars(pl_df[col], period=period).alias(f'{col}_cmo_{period}')
                    )
                    # pl_df[f'{col}_cmo_{period}'] = _calculate_cmo_polars(pl_df[col], period=period)
            return pl_df
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names

        pandas_df = pandas_df.sort_values(by=[*group_names, date_column])
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.group_by(
            data.grouper.names, maintain_order=True
        ).apply(apply_cmo)
        
        df = out_df

    else:
        
        _exprs = []
        
        for col in close_column:
            for period in periods:
                _expr = _calculate_cmo_polars(pl.col(col), period=period).alias(f'{col}_cmo_{period}')
                _exprs.append(_expr)
        
        pandas_df = pandas_df.sort_values(by=[date_column])
        
        df = pl.from_pandas(pandas_df)
        
        out_df = df.select(_exprs)
        
        df = pl.concat([df, out_df], how="horizontal")
    
    return df.to_pandas()
    
    

def _calculate_cmo_polars(series: pl.Series, period=14):
    # Calculate the difference in closing prices
    delta = series.diff()
    
    # Separate gains and losses    
    # gains = delta.apply(lambda x: x if x > 0 else 0)
    # losses = delta.apply(lambda x: -x if x < 0 else 0)
    
    gains = pl.when(delta > 0).then(delta).otherwise(0)
    losses = pl.when(delta <= 0).then(-delta).otherwise(0)

    # Calculate the sum of gains and losses using a rolling window
    sum_gains = gains.rolling_sum(window_size=period)
    sum_losses = losses.rolling_sum(window_size=period)

    # Calculate CMO
    total_movement = sum_gains + sum_losses
    cmo = 100 * (sum_gains - sum_losses) / total_movement
    
    return cmo


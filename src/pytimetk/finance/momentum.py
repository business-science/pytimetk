


import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage



@pf.register_dataframe_method
def augment_momentum(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    fast_period: Union[int, Tuple[int, int], List[int]] = 21,
    slow_period: Union[int, Tuple[int, int], List[int]] = 252,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    # Check if fast_period and slow_period are lists, tuples or integers
    if isinstance(fast_period, int):
        fast_period = [fast_period]
    elif isinstance(fast_period, tuple):
        fast_period = list(range(fast_period[0], fast_period[1] + 1))
    elif not isinstance(fast_period, list):
        raise ValueError("fast_period must be an int, tuple or list")
    
    if isinstance(slow_period, int):
        slow_period = [slow_period]
    elif isinstance(slow_period, tuple):
        slow_period = list(range(slow_period[0], slow_period[1] + 1))
    elif not isinstance(slow_period, list):
        raise ValueError("slow_period must be an int, tuple or list")
    
    # Reduce memory usage
    if reduce_memory:
        data = reduce_memory_usage(data)        

    # CALCULATE MOMENTUM:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        ret = data.obj.copy()
        
        for fp in fast_period:
            for sp in slow_period:
                if fp < sp:
                    
                    normalization_period = int(sp / 2)
                    
                    ret = _calculate_grouped_momentum(
                        data = ret,
                        group_names = group_names,
                        date_column = date_column,
                        close_column = close_column,
                        fast_period = fp,
                        slow_period = sp,
                        normalization_period = normalization_period,
                        engine = engine
                    )
        
    elif isinstance(data, pd.DataFrame):
        
        ret = data.copy()
        
        for fp in fast_period:
            for sp in slow_period:
                if fp < sp:
                    
                    normalization_period = int(sp / 2)
                    
                    ret = _calculate_momentum(
                        data = ret,
                        date_column = date_column,
                        close_column = close_column,
                        fast_period = fp,
                        slow_period = sp,
                        normalization_period = normalization_period,
                        engine = engine
                    )
                    
        
    else:
        raise ValueError("data must be a pandas DataFrame or a pandas GroupBy object")


    
    if reduce_memory:
        ret = reduce_memory_usage(ret)

    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_momentum = augment_momentum

def _calculate_grouped_momentum(data, group_names, date_column, close_column, fast_period, slow_period, normalization_period, engine):
    
    if not isinstance(group_names, list):
        group_names = [group_names]
    df = data.copy()
    
    cols_to_keep = df.columns.tolist()
    
    # Get Lags
    df = df \
        .groupby(group_names) \
        .augment_lags(
            date_column = date_column, 
            value_column = close_column, 
            lags = [1, slow_period, normalization_period, fast_period],
            engine = engine
        )
    
    # Get Returns
    if engine == 'pandas':
        df[f'{close_column}_returns'] = df.groupby(group_names)[close_column].pct_change()
    
    else:
        # Polars
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns([
            pl.col(close_column).pct_change().over(*group_names).alias(f'{close_column}_returns'),
        ])
        df = pl_df.to_pandas()
    
    # Get rolling standard deviation
    df = df \
        .groupby(group_names) \
        .augment_rolling(
            date_column = date_column, 
            value_column = f'{close_column}_returns', 
            window = normalization_period,
            window_func = 'std',
            engine = engine,
            show_progress = False,
        )
        
    # Calculate Momentum
    df[f'{close_column}_momentum_{fast_period}_{slow_period}'] = (
        (df[f'{close_column}_lag_{fast_period}'] - df[f'{close_column}_lag_{slow_period}']) / df[f'{close_column}_lag_{slow_period}'] - (df[f'{close_column}_lag_1'] - df[f'{close_column}_lag_{fast_period}']) / df[f'{close_column}_lag_{fast_period}']
        
    ) / df[f'{close_column}_returns_rolling_std_win_{normalization_period}']
    
    # Drop intermediate columns
    df = df[cols_to_keep + [f'{close_column}_momentum_{fast_period}_{slow_period}']]
    
    return df

def _calculate_momentum(data, date_column, close_column, fast_period, slow_period, normalization_period, engine):
    
    df = data.copy()
    
    cols_to_keep = df.columns.tolist()
    
    # Get Lags
    df = df \
        .augment_lags(
            date_column = date_column, 
            value_column = close_column, 
            lags = [1, slow_period, normalization_period, fast_period],
            engine = engine
        )
    
    # Get Returns
    if engine == 'pandas':
        df[f'{close_column}_returns'] = df[close_column].pct_change()
    
    else:
        # Polars
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.with_columns([
            pl.col(close_column).pct_change().alias(f'{close_column}_returns'),
        ])
        df = pl_df.to_pandas()
    
    # Get rolling standard deviation
    df = df \
        .augment_rolling(
            date_column = date_column, 
            value_column = f'{close_column}_returns', 
            window = normalization_period,
            window_func = 'std',
            engine = engine,
            show_progress = False,
        )
        
    # Calculate Momentum
    df[f'{close_column}_momentum_{fast_period}_{slow_period}'] = (
        (df[f'{close_column}_lag_{fast_period}'] - df[f'{close_column}_lag_{slow_period}']) / df[f'{close_column}_lag_{slow_period}'] - (df[f'{close_column}_lag_1'] - df[f'{close_column}_lag_{fast_period}']) / df[f'{close_column}_lag_{fast_period}']
        
    ) / df[f'{close_column}_returns_rolling_std_win_{normalization_period}']
    
    # Drop intermediate columns
    df = df[cols_to_keep + [f'{close_column}_momentum_{fast_period}_{slow_period}']]
    
    return df
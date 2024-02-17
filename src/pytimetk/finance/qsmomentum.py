import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf
from typing import Union, List, Tuple
from functools import partial

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage



@pf.register_dataframe_method
def augment_qsmomentum(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    
    roc_fast_period: Union[int, Tuple[int, int], List[int]] = 21,
    roc_slow_period: Union[int, Tuple[int, int], List[int]] = 252,
    returns_period: Union[int, Tuple[int, int], List[int]] = 126,
    
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    # Check if roc_fast_period lists, tuples or integers
    if isinstance(roc_fast_period, int):
        roc_fast_period = [roc_fast_period]
    elif isinstance(roc_fast_period, tuple):
        roc_fast_period = list(range(roc_fast_period[0], roc_fast_period[1] + 1))
    elif not isinstance(roc_fast_period, list):
        raise ValueError("roc_fast_period must be an int, tuple or list")
    
    if isinstance(roc_slow_period, int):
        roc_slow_period = [roc_slow_period]
    elif isinstance(roc_slow_period, tuple):
        roc_slow_period = list(range(roc_slow_period[0], roc_slow_period[1] + 1))
    elif not isinstance(roc_slow_period, list):
        raise ValueError("roc_slow_period must be an int, tuple or list")
    
    if isinstance(returns_period, int):
        returns_period = [returns_period]
    elif isinstance(returns_period, tuple):
        returns_period = list(range(returns_period[0], returns_period[1] + 1))
    elif not isinstance(returns_period, list):
        raise ValueError("returns_period must be an int, tuple or list")
    
    # Reduce memory usage
    if reduce_memory:
        data = reduce_memory_usage(data)        

    # CALCULATE MOMENTUM:
    
    if engine == "pandas":
        func = _calculate_qsmomentum_pandas
    elif engine == "polars":
        func = _calculate_qsmomentum_polars 
    
    ret = data
    
    for fp in roc_fast_period:
        for sp in roc_slow_period:
            for np in returns_period:
                if fp < sp:
                    if np < sp:
                        
                        def f(close):
                            return func(close, fp, sp, np)
                    
                        ret = ret.augment_rolling(
                            date_column = date_column,
                            value_column = close_column,
                            window = sp,
                            window_func = ("f", f),
                            engine = engine
                        )
                        
                        ret.rename(columns={ret.columns[-1]: f'{close_column}_qsmom_{fp}_{sp}_{np}'}, inplace=True)
                

    
    if reduce_memory:
        ret = reduce_memory_usage(ret)

    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_qsmomentum = augment_qsmomentum

def _calculate_qsmomentum_pandas(close, roc_fast_period, roc_slow_period, returns_period):
    
    close = pd.Series(close)
    
    returns = close.pct_change().iloc[-returns_period:]
    
    roc_slow_calc = (close.iloc[-roc_fast_period] - close.iloc[-roc_slow_period]) / (close.iloc[-roc_slow_period] + 0.0000000001)
    
    roc_fast_calc = (close.iloc[-1] - close.iloc[-roc_fast_period]) / (close.iloc[-roc_fast_period] + 0.0000000001)
    
    mom = (roc_slow_calc - roc_fast_calc) / np.std(returns)
    
    return mom

def _calculate_qsmomentum_polars(close, roc_fast_period, roc_slow_period, returns_period):
    
    close = pl.Series(close)
    
    returns = close.pct_change()
    
    returns_last_returns_period = returns.slice(-returns_period, returns_period)
    
    roc_slow_calc = (close[-roc_fast_period] - close[-roc_slow_period]) / (close[-roc_slow_period] + 0.0000000001)
    
    roc_fast_calc = (close[-1] - close[-roc_fast_period]) / (close[-roc_fast_period] + 0.0000000001)
    
    mom = (roc_slow_calc - roc_fast_calc) / returns_last_returns_period.std()
    
    return mom
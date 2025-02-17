import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf
from typing import Union, List, Tuple
from functools import partial

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


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
    '''The function `augment_qsmomentum` calculates Quant Science Momentum for financial data.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter in the `augment_qsmomentum` function is expected to be a pandas DataFrame or a
        pandas DataFrameGroupBy object. This parameter represents the input data on which the momentum
        calculations will be performed.
    date_column : str
        The `date_column` parameter in the `augment_qsmomentum` function refers to the column in your input
        data that contains the dates associated with the financial data. This column is used for time-based
        operations and calculations within the function.
    close_column : str
        The `close_column` parameter in the `augment_qsmomentum` function refers to the column in the input
        DataFrame that contains the closing prices of the financial instrument or asset for which you want
        to calculate the momentum. 
    roc_fast_period : Union[int, Tuple[int, int], List[int]], optional
        The `roc_fast_period` parameter in the `augment_qsmomentum` function determines the period used for
        calculating the fast Rate of Change (ROC) momentum indicator. 
    roc_slow_period : Union[int, Tuple[int, int], List[int]], optional
        The `roc_slow_period` parameter in the `augment_qsmomentum` function represents the period used for
        calculating the slow rate of change (ROC) in momentum analysis. 
    returns_period : Union[int, Tuple[int, int], List[int]], optional
        The `returns_period` parameter in the `augment_qsmomentum` function determines the period over
        which the returns are calculated. 
    reduce_memory : bool, optional
        The `reduce_memory` parameter in the `augment_qsmomentum` function is a boolean flag that indicates
        whether memory reduction techniques should be applied to the input data before and after the
        momentum calculation process. If set to `True`, memory reduction methods will be used to optimize
        memory usage, potentially reducing
    engine : str, optional
        The `engine` parameter in the `augment_qsmomentum` function specifies the computation engine to be
        sed for calculating momentum. It can have two possible values: "pandas" or "polars". 
    
    Returns
    -------
        The function `augment_qsmomentum` returns a pandas DataFrame that has been augmented with columns
        representing the Quant Science Momentum (QSM) calculated based on the specified parameters
        such as roc_fast_period, roc_slow_period, and returns_period. 
        
    Notes
    -----
    
    The Quant Science Momentum (QSM) is a momentum indicator that is calculated based on the Slow Rate of Change (ROC) usually over a 252-day period and the Fast Rate of Change (ROC) usually over a 21-day period. 
    
    The QSM is calculated as the difference between the slow and fast ROCs divided by the standard deviation of the returns over a specified period.
    
    This provides a measure of momentum that is normalized by the rolling volatility of the returns.
    
    Examples
    --------
    ``` {python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])

    df.glimpse()
    ```
    
    ``` {python}
    # PANDAS QS MOMENTUM CALCULATION
    df_qsmom = (
        df 
            .query('symbol == "GOOG"') 
            .augment_qsmomentum(
                date_column = 'date', 
                close_column = 'close', 
                roc_fast_period = [1, 5, 21], 
                roc_slow_period = 252, 
                returns_period = 126, 
                engine = "pandas"
            ) 
    )
    
    df_qsmom.dropna().glimpse()
    ```
    
    ``` {python}
    # POLARS QS MOMENTUM CALCULATION
    df_qsmom = (
        df 
            .query('symbol == "GOOG"') 
            .augment_qsmomentum(
                date_column = 'date', 
                close_column = 'close', 
                roc_fast_period = [1, 5, 21], 
                roc_slow_period = 252, 
                returns_period = 126, 
                engine = "polars"
            ) 
    )
    
    df_qsmom.dropna().glimpse()
    ```
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)

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
    
    # if engine == 'polars':
    #     # Polars Index to Match Pandas
    #     ret.index = idx_unsorted     

    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()

    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_qsmomentum = augment_qsmomentum

def _calculate_qsmomentum_pandas(close, roc_fast_period, roc_slow_period, returns_period):
    close = pd.Series(close)
    returns = close.pct_change().iloc[-returns_period:]
    std_returns = np.std(returns)
    
    # Check if the standard deviation is too small:
    if np.abs(std_returns) < 1e-10:
        return np.nan

    # Calculate the rates of change with a small epsilon added to the denominator
    roc_slow_calc = (close.iloc[-roc_fast_period] - close.iloc[-roc_slow_period]) / (close.iloc[-roc_slow_period] + 1e-10)
    roc_fast_calc = (close.iloc[-1] - close.iloc[-roc_fast_period]) / (close.iloc[-roc_fast_period] + 1e-10)
    
    mom = (roc_slow_calc - roc_fast_calc) / std_returns
    return mom


def _calculate_qsmomentum_polars(close, roc_fast_period, roc_slow_period, returns_period):
    close = pl.Series(close)
    returns = close.pct_change()
    returns_last_returns_period = returns.slice(-returns_period, returns_period)
    std_returns = returns_last_returns_period.std()
    
    # Check if the standard deviation is too small (or undefined)
    if std_returns is None or std_returns < 1e-10:
        return np.nan
    
    roc_slow_calc = (close[-roc_fast_period] - close[-roc_slow_period]) / (close[-roc_slow_period] + 1e-10)
    roc_fast_calc = (close[-1] - close[-roc_fast_period]) / (close[-roc_fast_period] + 1e-10)
    
    mom = (roc_slow_calc - roc_fast_calc) / std_returns
    return mom

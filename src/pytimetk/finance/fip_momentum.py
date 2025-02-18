import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple
from pytimetk.utils.parallel_helpers import progress_apply
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_fip_momentum(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    window: Union[int, List[int]] = 252,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''
    Calculate the "Frog In The Pan" (FIP) momentum metric over one or more rolling windows 
    using either pandas or polars engine, augmenting the DataFrame with FIP columns.
    
    The FIP momentum is defined as:
    FIP = Total Return * (percent of negative returns - percent of positive returns)
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or grouped DataFrame containing time series data.
    date_column : str
        Name of the column with dates or timestamps.
    close_column : str
        Name of the column with closing prices to calculate returns.
    window : Union[int, List[int]], optional
        Size of the rolling window(s) as an integer or list of integers (default is 252).
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame augmented with FIP momentum columns:
        - {close_column}_fip_momentum_{w}: Rolling FIP momentum for each window w
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])
    
    # Single window
    fip_df = (
        df.query("symbol == 'AAPL'")
        .augment_fip_momentum(
            date_column='date',
            close_column='close',
            window=252
        )
    )
    fip_df.tail()
    ```
    
    ```{python}    
    # Multiple windows
    fip_df = (
        df.groupby('symbol')
        .augment_fip_momentum(
            date_column='date',
            close_column='close',
            window=[63, 252],
            engine='polars'
        )
    )
    fip_df.tail()
    ```
    '''
    
    # Convert single window to list for consistency
    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, (list, tuple)):
        windows = window
    else:
        raise ValueError("`window` must be an integer or list/tuple of integers")
    
    # Validate windows
    if not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError("All window values must be positive integers")
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_fip_momentum_pandas(data, date_column, close_column, windows)
    elif engine == 'polars':
        ret = _augment_fip_momentum_polars(data, date_column, close_column, windows)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_fip_momentum = augment_fip_momentum

def _augment_fip_momentum_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str,
    windows: List[int]
) -> pd.DataFrame:
    """Pandas implementation of FIP momentum calculation for multiple windows."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    # Calculate daily returns once
    df[f'{col}_returns'] = df[col].pct_change()
    
    # Define FIP calculation function
    def calc_fip(ser, window):
        returns = ser.dropna()
        if len(returns) < window // 2:  # Require at least half the window
            return np.nan
        
        # Total return: cumulative product of (1 + daily returns) - 1
        total_return = np.prod(1 + returns) - 1
        
        # Percent positive and negative returns
        pct_positive = (returns > 0).mean()
        pct_negative = (returns < 0).mean()
        
        # FIP = Total Return * (pct_negative - pct_positive)
        fip = total_return * (pct_negative - pct_positive)
        return fip
    
    # Apply rolling calculation for each window
    if group_names:
        for w in windows:
            fip_series = df.groupby(group_names)[f'{col}_returns'].rolling(w).apply(lambda x: calc_fip(x, w), raw=False)
            df[f'{col}_fip_momentum_{w}'] = fip_series.reset_index(level=0, drop=True)
    else:
        for w in windows:
            df[f'{col}_fip_momentum_{w}'] = df[f'{col}_returns'].rolling(w).apply(lambda x: calc_fip(x, w), raw=False)
    
    # Drop temporary returns column
    df.drop(columns=[f'{col}_returns'], inplace=True)
    
    return df





def _augment_fip_momentum_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str,
    windows: List[int]
) -> pd.DataFrame:
    """Polars implementation of FIP momentum calculation for multiple windows."""
    
    def fip_calc(x, w):
        # Ensure we are working with a NumPy array.
        arr = x.to_numpy() if hasattr(x, "to_numpy") else x
        # Check if we have enough non-NaN values.
        if np.sum(~np.isnan(arr)) < w // 2:
            return np.nan
        # Calculate total return.
        total_return = np.prod(1 + arr[~np.isnan(arr)]) - 1
        # Calculate percentage of positive and negative returns.
        pct_positive = np.sum(arr > 0) / np.sum(~np.isnan(arr))
        pct_negative = np.sum(arr < 0) / np.sum(~np.isnan(arr))
        return total_return * (pct_negative - pct_positive)

    
    if isinstance(data, pd.DataFrame):
        pandas_df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
    else:
        pandas_df = data.copy()
        group_names = None
    
    df = pl.from_pandas(pandas_df)
    col = close_column
    
    # Calculate daily returns once
    df = df.with_columns(
        (pl.col(col) / pl.col(col).shift(1) - 1).alias(f'{col}_returns')
    )
    
    # Define FIP expressions for each window
    fip_exprs = []
    for w in windows:
        fip_expr = (
            pl.col(f'{col}_returns')
            .rolling_map(lambda x, w=w: fip_calc(x, w), window_size=w, min_periods=w//2)
            .alias(f'{col}_fip_momentum_{w}')
        )
        fip_exprs.append(fip_expr)
        
    
    # Apply expressions, grouped or ungrouped
    if group_names:
        df = df.with_columns([expr.over(group_names) for expr in fip_exprs])
    else:
        df = df.with_columns(fip_exprs)
    
    # Drop temporary returns column
    df = df.drop(f'{col}_returns')
    
    return df.to_pandas()

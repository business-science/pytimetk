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
    engine: str = 'pandas',
    fip_method: str = 'original',
    skip_window: int = 0  # new parameter to skip the first n periods
) -> pd.DataFrame:
    '''
    Calculate the "Frog In The Pan" (FIP) momentum metric over one or more rolling windows 
    using either the pandas or polars engine, augmenting the DataFrame with FIP columns.
    
    The FIP momentum is defined as:
    - For 'original': FIP = Total Return * (percent of negative returns - percent of positive returns)
    - For 'modified': FIP = sign(Total Return) * (percent of positive returns - percent of negative returns)
    
    An optional parameter, skip_window, allows you to skip the first n periods (e.g., one month) 
    to mitigate the effects of mean reversion.
    
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
    fip_method : str, optional
        Type of FIP calculation:
        - 'original': Original FIP calculation (default) where negative FIP indicates greater momentum.
        - 'modified': Modified FIP where positive FIP indicates greater momentum.
    skip_window : int, optional
        Number of initial periods to skip (set to NA) for each rolling calculation. Default is 0.
    
    Returns
    -------
    pd.DataFrame
        DataFrame augmented with FIP momentum columns:
        - {close_column}_fip_momentum_{w}: Rolling FIP momentum for each window w
    
    
    Notes
    -----
    - For 'original', a positive FIP may indicate inconsistency in the trend.
    - For 'modified', a positive FIP indicates stronger momentum in the direction of the trend (upward or downward).
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])
    
    # Single window with original FIP
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
    
    
    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, (list, tuple)):
        windows = window
    else:
        raise ValueError("`window` must be an integer or list/tuple of integers")
    
    if not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError("All window values must be positive integers")
    
    if fip_method not in ['original', 'modified']:
        raise ValueError("`fip_method` must be 'original' or 'modified'")
    
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_fip_momentum_pandas(data, date_column, close_column, windows, fip_method, skip_window)
    elif engine == 'polars':
        ret = _augment_fip_momentum_polars(data, date_column, close_column, windows, fip_method, skip_window)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

pd.core.groupby.generic.DataFrameGroupBy.augment_fip_momentum = augment_fip_momentum

def _augment_fip_momentum_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    fip_method: str,
    skip_window: int
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    df[f'{col}_returns'] = df[col].pct_change()
    
    def calc_fip(ser, window, fip_method):
        returns = ser.dropna()
        if len(returns) < window // 2:
            return np.nan
        
        total_return = np.prod(1 + returns) - 1
        pct_positive = (returns > 0).mean()
        pct_negative = (returns < 0).mean()
        if fip_method == 'original':
            return total_return * (pct_negative - pct_positive)
        elif fip_method == 'modified':
            return np.sign(total_return) * (pct_positive - pct_negative)
    
    if group_names:
        for w in windows:
            out_series = pd.Series(index=df.index, dtype=float)
            # Process each group separately to preserve original index types
            for name, group_df in df.groupby(group_names):
                roll = group_df[f'{col}_returns'].rolling(w).apply(
                    lambda x: calc_fip(x, w, fip_method), raw=False
                )
                if skip_window > 0:
                    roll.iloc[:skip_window] = np.nan
                out_series.loc[roll.index] = roll
            df[f'{col}_fip_momentum_{w}'] = out_series
    else:
        for w in windows:
            roll = df[f'{col}_returns'].rolling(w).apply(
                lambda x: calc_fip(x, w, fip_method), raw=False
            )
            if skip_window > 0:
                roll.iloc[:skip_window] = np.nan
            df[f'{col}_fip_momentum_{w}'] = roll
    
    df.drop(columns=[f'{col}_returns'], inplace=True)
    return df


def _augment_fip_momentum_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str,
    windows: List[int],
    fip_method: str,
    skip_window: int
) -> pd.DataFrame:
    def fip_calc(x, w, fip_method):
        # Convert input to a NumPy array for consistency.
        arr = np.array(x)
        valid = ~np.isnan(arr)
        if np.sum(valid) < w // 2:
            return np.nan
        total_return = np.prod(1 + arr[valid]) - 1
        pct_positive = np.sum(arr[valid] > 0) / np.sum(valid)
        pct_negative = np.sum(arr[valid] < 0) / np.sum(valid)
        if fip_method == 'original':
            return total_return * (pct_negative - pct_positive)
        elif fip_method == 'modified':
            return np.sign(total_return) * (pct_positive - pct_negative)
    
    if isinstance(data, pd.DataFrame):
        pandas_df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj.copy()
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
    else:
        pandas_df = data.copy()
        group_names = None
    
    # Convert to Polars DataFrame
    df = pl.from_pandas(pandas_df)
    col = close_column
    
    # Compute returns column
    df = df.with_columns(
        (pl.col(col) / pl.col(col).shift(1) - 1).alias(f'{col}_returns')
    )
    
    # For each window, compute the rolling FIP momentum and then apply the skip mask.
    for w in windows:
        expr = pl.col(f'{col}_returns').rolling_map(
            lambda x: fip_calc(x, w, fip_method),
            window_size=w,
            min_periods=w // 2
        ).alias(f'{col}_fip_momentum_{w}')
        
        if group_names:
            # Apply the rolling map within groups.
            df = df.with_columns(expr.over(group_names))
            # Instead of pl.cumcount(), add a global row count and subtract the group minimum.
            df = df.with_row_count("global_row_nr")
            df = df.with_columns(
                (pl.col("global_row_nr") - pl.col("global_row_nr").min().over(group_names)).alias("row_nr")
            )
            df = df.with_columns(
                pl.when(pl.col("row_nr") < skip_window)
                  .then(None)
                  .otherwise(pl.col(f'{col}_fip_momentum_{w}'))
                  .alias(f'{col}_fip_momentum_{w}')
            ).drop("row_nr", "global_row_nr")
        else:
            df = df.with_columns(expr)
            # For ungrouped data, use with_row_count() to add a row counter.
            df = df.with_row_count("row_nr")
            df = df.with_columns(
                pl.when(pl.col("row_nr") < skip_window)
                  .then(None)
                  .otherwise(pl.col(f'{col}_fip_momentum_{w}'))
                  .alias(f'{col}_fip_momentum_{w}')
            ).drop("row_nr")
    
    df = df.drop(f'{col}_returns')
    return df.to_pandas()




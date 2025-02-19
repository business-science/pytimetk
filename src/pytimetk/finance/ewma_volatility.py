import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Tuple

import pandas_flavor as pf
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_ewma_volatility(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    decay_factor: float = 0.94,
    window: Union[int, Tuple[int, int], List[int]] = 20,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''Calculate Exponentially Weighted Moving Average (EWMA) volatility for a financial time series.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or GroupBy object with time series data.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices to calculate volatility.
    decay_factor : float, optional
        Smoothing factor (lambda) for EWMA, between 0 and 1. Higher values give more weight to past data. Default is 0.94 (RiskMetrics standard).
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window to initialize EWMA calculation. For each window value the EWMA volatility is only computed when at least that many observations are available. 
        You may provide a single integer or multiple values (via tuple or list). Default is 20.
    reduce_memory : bool, optional
        If True, reduces memory usage before calculation. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - {close_column}_ewma_vol_{window}_{decay_factor}: EWMA volatility calculated using a minimum number of periods equal to each specified window.
    
    Notes
    -----
    EWMA volatility emphasizes recent price movements and is computed recursively as:
    
        σ²_t = (1 - λ) * r²_t + λ * σ²_{t-1}
    
    where r_t is the log return. By using the `min_periods` (set to the provided window value) we ensure that the EWMA is only calculated after enough observations have accumulated.
    
    References:
    
    - https://www.investopedia.com/articles/07/ewma.asp
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
    
    # Example 1 - Calculate EWMA volatility for a single stock
    
    df.query("symbol == 'AAPL'").augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```
    
    ```{python}
    # Example 2 - Calculate EWMA volatility for multiple stocks
    df.groupby('symbol').augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```
    
    ```{python}
    # Example 3 - Calculate EWMA volatility using Polars engine
    df.query("symbol == 'AAPL'").augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50],
        engine='polars'
    ).glimpse()
    ```
    
    ```{python}
    # Example 4 - Calculate EWMA volatility for multiple stocks using Polars engine
    
    df.groupby('symbol').augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50],
        engine='polars'
    ).glimpse()
    ```    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    if not 0 < decay_factor < 1:
        raise ValueError("decay_factor must be between 0 and 1.")
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, tuple):
        windows = list(range(window[0], window[1] + 1))
    elif isinstance(window, list):
        windows = window
    else:
        raise TypeError(f"Invalid window specification: type: {type(window)}. Please use int, tuple, or list.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_ewma_volatility_pandas(data, date_column, close_column, decay_factor, windows)
    elif engine == 'polars':
        ret = _augment_ewma_volatility_polars(data, date_column, close_column, decay_factor, windows)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    ret = ret.sort_index()
    
    return ret


pd.core.groupby.generic.DataFrameGroupBy.augment_ewma_volatility = augment_ewma_volatility


def _augment_ewma_volatility_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    decay_factor: float,
    windows: List[int]
) -> pd.DataFrame:
    """Pandas implementation of EWMA volatility calculation with varying minimum periods."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    
    # Calculate log returns and squared returns
    df['log_returns'] = np.log(df[col] / df[col].shift(1))
    df['squared_returns'] = df['log_returns'] ** 2
    
    # For each specified window (i.e. min_periods), compute EWMA volatility
    for win in windows:
        col_name = f'{col}_ewma_vol_{win}_{decay_factor:.2f}'
        if group_names:
            # Compute groupwise EWMA with a minimum number of periods equal to win
            ewma_variance = (
                df.groupby(group_names)['squared_returns']
                .ewm(alpha=1 - decay_factor, adjust=False, min_periods=win)
                .mean()
            )
            df[col_name] = (
                ewma_variance
                .apply(np.sqrt)
                .reset_index(level=0, drop=True)
            )
        else:
            ewma_variance = df['squared_returns'].ewm(alpha=1 - decay_factor, adjust=False, min_periods=win).mean()
            df[col_name] = np.sqrt(ewma_variance)
    
    df = df.drop(columns=['log_returns', 'squared_returns'])
    return df


def _augment_ewma_volatility_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    decay_factor: float,
    windows: List[int]
) -> pd.DataFrame:
    """Polars implementation of EWMA volatility calculation with varying minimum periods."""
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
    else:
        pandas_df = data.copy()
        group_names = None
    
    df = pl.from_pandas(pandas_df)
    col = close_column
    
    # Calculate log returns and squared returns
    df = df.with_columns(
        (pl.col(col).log() - pl.col(col).shift(1).log()).alias('log_returns')
    ).with_columns(
        (pl.col('log_returns') ** 2).alias('squared_returns')
    )
    
    for win in windows:
        col_name = f'{col}_ewma_vol_{win}_{decay_factor:.2f}'
        if group_names:
            df = df.with_columns(
                pl.col('squared_returns')
                .ewm_mean(alpha=1 - decay_factor, adjust=False, min_periods=win)
                .over(group_names)
                .sqrt()
                .alias(col_name)
            )
        else:
            df = df.with_columns(
                pl.col('squared_returns')
                .ewm_mean(alpha=1 - decay_factor, adjust=False, min_periods=win)
                .sqrt()
                .alias(col_name)
            )
    
    df = df.drop(['log_returns', 'squared_returns'])
    return df.to_pandas()


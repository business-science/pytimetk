import pandas as pd
import polars as pl
import numpy as np
from typing import Union, List, Tuple, Optional
from joblib import Parallel, delayed

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

import pandas_flavor as pf
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_regime_detection(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    window: Union[int, Tuple[int, int], List[int]] = 252,
    n_regimes: int = 2,
    method: str = 'hmm',
    step_size: int = 1,
    n_iter: int = 100,
    n_jobs: int = -1,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''Detect regimes in a financial time series using a specified method (e.g., HMM).
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or GroupBy object with time series data.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices for regime detection.
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window to fit the regime detection model. Default is 252.
    n_regimes : int, optional
        Number of regimes to detect (e.g., 2 for bull/bear). Default is 2.
    method : str, optional
        Method for regime detection. Currently supports 'hmm'. Default is 'hmm'.
    step_size : int, optional
        Step size between HMM fits (e.g., 10 fits every 10 rows). Default is 1.
    n_iter : int, optional
        Number of iterations for HMM fitting. Default is 100.
    n_jobs : int, optional
        Number of parallel jobs for group processing (-1 uses all cores). Default is -1.
    reduce_memory : bool, optional
        If True, reduces memory usage. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added columns:
        - {close_column}_regime_{window}: Integer labels for detected regimes (e.g., 0, 1).
    
    Notes
    -----
    - Uses Hidden Markov Model (HMM) to identify latent regimes based on log returns.
    - Regimes reflect distinct statistical states (e.g., high/low volatility, trending).
    - Requires 'hmmlearn' package. Install with `pip install hmmlearn`.    
    
    Examples
    --------
    ```python
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock regime detection with pandas engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.query("symbol == 'AAPL'")
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=252,
            n_regimes=2
        )
    )
    regime_df.head().glimpse()
    ```
    
    ```python
    # Example 2 - Multiple stocks with groupby using pandas engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.groupby('symbol')
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=[252, 504],  # One year and two years
            n_regimes=3
        )
    )
    regime_df.groupby('symbol').tail(1).glimpse()
    ```
    
    ```python
    # Example 3 - Single stock regime detection with polars engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.query("symbol == 'AAPL'")
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=252,
            n_regimes=2,
            engine='polars'
        )
    )
    regime_df.glimpse()
    ```
    
    ```python
    # Example 4 - Multiple stocks with groupby using polars engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.groupby('symbol')
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=504,
            n_regimes=3,
            engine='polars'
        )
    )
    regime_df.groupby('symbol').tail(1).glimpse()
    ```
    '''
    
    # Check for hmmlearn availability
    if method.lower() == 'hmm' and GaussianHMM is None:
        raise ImportError(
            "The 'hmm' method requires the 'hmmlearn' package, which is not installed. "
            "Please install it using: `pip install hmmlearn`"
        )
    
    if method.lower() == 'hmm' and GaussianHMM is None:
        raise ImportError(
            "The 'hmm' method requires the 'hmmlearn' package, which is not installed. "
            "Please install it using: `pip install hmmlearn`"
        )
    
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if method.lower() != 'hmm':
        raise ValueError("Only 'hmm' method is currently supported.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")
    
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
        ret = _augment_regime_detection_pandas(data, date_column, close_column, windows, n_regimes, step_size, n_iter, n_jobs)
    elif engine == 'polars':
        ret = _augment_regime_detection_polars(data, date_column, close_column, windows, n_regimes, step_size, n_iter, n_jobs)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    ret = ret.sort_index()
    
    return ret


pd.core.groupby.generic.DataFrameGroupBy.augment_regime_detection = augment_regime_detection


def _augment_regime_detection_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    n_regimes: int,
    step_size: int,
    n_iter: int,
    n_jobs: int
) -> pd.DataFrame:
    """Pandas implementation of regime detection using HMM."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    
    df['log_returns'] = np.log(df[col] / df[col].shift(1))
    df['log_returns'] = df['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    def detect_regimes(series, window, n_regimes, step_size, n_iter):
        n = len(series)
        regimes = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n, step_size):
            window_data = series[max(0, i - window + 1):i + 1].dropna()
            if len(window_data) < max(window // 2, n_regimes * 10):
                continue
            window_data = window_data.values.reshape(-1, 1)
            if not np.all(np.isfinite(window_data)):
                continue
            try:
                model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=n_iter)
                model.fit(window_data)
                predicted = model.predict(window_data)
                # Fill regimes from i-step_size+1 to i with the last predicted value
                start_idx = max(0, i - step_size + 1)
                regimes[start_idx:i + 1] = predicted[-step_size if i > window - 1 else -start_idx:]
            except ValueError as e:
                print(f"Warning: HMM fit failed at index {i} with error: {e}")
                continue
        return pd.Series(regimes, index=series.index)
    
    for window in windows:
        if group_names:
            # Parallelize across groups
            results = Parallel(n_jobs=n_jobs)(
                delayed(detect_regimes)(group['log_returns'], window, n_regimes, step_size, n_iter)
                for _, group in df.groupby(group_names)
            )
            df[f'{col}_regime_{window}'] = pd.concat(results).reindex(df.index)
        else:
            df[f'{col}_regime_{window}'] = detect_regimes(df['log_returns'], window, n_regimes, step_size, n_iter)
    
    df = df.drop(columns=['log_returns'])
    return df


def _augment_regime_detection_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    n_regimes: int,
    step_size: int,
    n_iter: int,
    n_jobs: int
) -> pd.DataFrame:
    """Polars implementation of regime detection using HMM (via pandas)."""
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj
        group_names = data.grouper.names
        if not isinstance(group_names, list):
            group_names = [group_names]
    else:
        pandas_df = data.copy()
        group_names = None
    
    df = pandas_df.copy()
    col = close_column
    
    df['log_returns'] = np.log(df[col] / df[col].shift(1))
    df['log_returns'] = df['log_returns'].replace([np.inf, -np.inf], np.nan)
    
    def detect_regimes(series, window, n_regimes, step_size, n_iter):
        n = len(series)
        regimes = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n, step_size):
            window_data = series[max(0, i - window + 1):i + 1].dropna()
            if len(window_data) < max(window // 2, n_regimes * 10):
                continue
            window_data = window_data.values.reshape(-1, 1)
            if not np.all(np.isfinite(window_data)):
                continue
            try:
                model = GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=n_iter)
                model.fit(window_data)
                predicted = model.predict(window_data)
                start_idx = max(0, i - step_size + 1)
                regimes[start_idx:i + 1] = predicted[-step_size if i > window - 1 else -start_idx:]
            except ValueError as e:
                print(f"Warning: HMM fit failed at index {i} with error: {e}")
                continue
        return pd.Series(regimes, index=series.index)
    
    for window in windows:
        if group_names:
            results = Parallel(n_jobs=n_jobs)(
                delayed(detect_regimes)(group['log_returns'], window, n_regimes, step_size, n_iter)
                for _, group in df.groupby(group_names)
            )
            df[f'{col}_regime_{window}'] = pd.concat(results).reindex(df.index)
        else:
            df[f'{col}_regime_{window}'] = detect_regimes(df['log_returns'], window, n_regimes, step_size, n_iter)
    
    df = df.drop(columns=['log_returns'])
    return df
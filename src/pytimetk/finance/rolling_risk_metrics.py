import pandas as pd
import polars as pl
import pandas_flavor as pf
import numpy as np
from typing import Union, List, Tuple, Optional
from pytimetk.utils.parallel_helpers import progress_apply
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from scipy import stats  # For skewness and kurtosis

@pf.register_dataframe_method
def augment_rolling_risk_metrics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    window: int = 252,
    risk_free_rate: float = 0.0,
    benchmark_column: Optional[str] = None,
    annualization_factor: int = 252,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The augment_rolling_risk_metrics function calculates rolling risk-adjusted performance
    metrics for a financial time series using either pandas or polars engine, and returns
    the augmented DataFrame with columns for Sharpe Ratio, Sortino Ratio, and other metrics.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input data can be a pandas DataFrame or a pandas DataFrameGroupBy object
        containing the time series data for risk metric calculations.
    date_column : str
        The name of the column containing dates or timestamps.
    close_column : str
        The column containing closing prices to calculate returns and risk metrics from.
    window : int, optional
        The rolling window size for calculations (e.g., 252 for annual). Default is 252.
    risk_free_rate : float, optional
        The assumed risk-free rate (e.g., 0.0 for 0%). Default is 0.0.
    benchmark_column : str or None, optional
        The column containing benchmark returns (e.g., market index) for Treynor and Information Ratios.
        Default is None.
    annualization_factor : int, optional
        The factor to annualize returns and volatility (e.g., 252 for daily data). Default is 252.
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame before calculation. Default is False.
    engine : str, optional
        The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame augmented with columns:
        - {close_column}_sharpe_ratio_{window}: Rolling Sharpe Ratio
        - {close_column}_sortino_ratio_{window}: Rolling Sortino Ratio
        - {close_column}_treynor_ratio_{window}: Rolling Treynor Ratio (if benchmark provided)
        - {close_column}_information_ratio_{window}: Rolling Information Ratio (if benchmark provided)
        - {close_column}_omega_ratio_{window}: Rolling Omega Ratio
        - {close_column}_volatility_annualized_{window}: Rolling annualized volatility
        - {close_column}_skewness_{window}: Rolling skewness of returns
        - {close_column}_kurtosis_{window}: Rolling kurtosis of returns
    
    Notes
    -----
    This function computes returns from closing prices and calculates rolling risk metrics:
    
    - Sharpe Ratio: Excess return over risk-free rate divided by volatility
    - Sortino Ratio: Excess return over risk-free rate divided by downside deviation
    - Treynor Ratio: Excess return over risk-free rate divided by beta (requires benchmark)
    - Information Ratio: Excess return over benchmark divided by tracking error (requires benchmark)
    - Omega Ratio: Ratio of gains to losses above/below a threshold
    - Volatility: Annualized standard deviation of returns
    - Skewness: Asymmetry of return distribution
    - Kurtosis: Fat-tailedness of return distribution
    
    Examples
    --------
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Single stock risk metrics
    risk_df = (
        df.query("symbol == 'AAPL'")
        .augment_rolling_risk_metrics(
            date_column='date',
            close_column='adjusted',
            window=252
        )
    )
    risk_df.head()
    ```
    
    ``` {python}
    # Multiple stocks with groupby and benchmark
    risk_df = (
        df.groupby('symbol')
        .augment_rolling_risk_metrics(
            date_column='date',
            close_column='adjusted',
            # benchmark_column='market_adjusted_returns',  # Use if a benchmark returns column exists
            window=60,
            engine='polars'
        )
    )
    risk_df.head()
    ```
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    if benchmark_column is not None:
        check_value_column(data, benchmark_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_rolling_risk_metrics_pandas(
            data, date_column, close_column, window, risk_free_rate,
            benchmark_column, annualization_factor
        )
    elif engine == 'polars':
        ret = _augment_rolling_risk_metrics_polars(
            data, date_column, close_column, window, risk_free_rate,
            benchmark_column, annualization_factor
        )
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling_risk_metrics = augment_rolling_risk_metrics



def _augment_rolling_risk_metrics_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    window: int,
    risk_free_rate: float,
    benchmark_column: Optional[str],
    annualization_factor: int
) -> pd.DataFrame:
    """Pandas implementation of rolling risk metrics calculation."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    
    col = close_column
    # Calculate log returns
    df[f'{col}_returns'] = np.log(df[col] / df[col].shift(1))
    
    if benchmark_column:
        df[f'{benchmark_column}_returns'] = np.log(df[benchmark_column] / df[benchmark_column].shift(1))
    
    # Define helper functions for rolling metrics with **kwargs to handle unexpected arguments
    def roll_downside_std(ser, **kwargs):
        # Apply rolling window first, then compute std of negative returns within each window
        return ser.rolling(window, min_periods=window//2).apply(
            lambda x: np.std(x[x < 0]) if np.any(x < 0) else np.nan, raw=True
        )
    
    def roll_omega(ser, **kwargs):
        # Apply rolling window first, then compute omega within each window
        return ser.rolling(window, min_periods=window//2).apply(
            lambda x: (
                np.sum(x[x > 0]) / np.abs(np.sum(x[x < 0]))
                if np.sum(x[x < 0]) != 0 else np.inf
            ) if np.sum(~np.isnan(x)) >= window//2 else np.nan,
            raw=True
        ).replace([np.inf, -np.inf], np.nan)
    
    def roll_beta(ser, bench_ser, **kwargs):
        cov = ser.rolling(window, min_periods=window//2).cov(bench_ser)
        var = bench_ser.rolling(window, min_periods=window//2).var()
        return cov / var.where(var != 0, np.nan)
    
    # Apply rolling calculations
    if group_names:
        grouped = df.groupby(group_names)
        
        # Precompute rolling metrics for all groups
        mean_ret = grouped[f'{col}_returns'].rolling(window, min_periods=window//2).mean()
        std_ret = grouped[f'{col}_returns'].rolling(window, min_periods=window//2).std()
        downside_std = grouped[f'{col}_returns'].apply(roll_downside_std, raw=False)
        omega = grouped[f'{col}_returns'].apply(roll_omega, raw=False)
        skew = grouped[f'{col}_returns'].rolling(window, min_periods=window//2).apply(lambda x: stats.skew(x, nan_policy='omit'), raw=True)
        kurt = grouped[f'{col}_returns'].rolling(window, min_periods=window//2).apply(lambda x: stats.kurtosis(x, nan_policy='omit'), raw=True)
        
        # Assign metrics directly using the grouped rolling results
        # Reset index to align with original df
        df[f'{col}_sharpe_ratio_{window}'] = ((mean_ret - risk_free_rate) / std_ret * np.sqrt(annualization_factor)).reset_index(level=0, drop=True)
        df[f'{col}_sortino_ratio_{window}'] = ((mean_ret - risk_free_rate) / downside_std * np.sqrt(annualization_factor)).reset_index(level=0, drop=True)
        df[f'{col}_volatility_annualized_{window}'] = (std_ret * np.sqrt(annualization_factor)).reset_index(level=0, drop=True)
        df[f'{col}_omega_ratio_{window}'] = omega.reset_index(level=0, drop=True)
        df[f'{col}_skewness_{window}'] = skew.reset_index(level=0, drop=True)
        df[f'{col}_kurtosis_{window}'] = kurt.reset_index(level=0, drop=True)
        
        if benchmark_column:
            bench_mean = grouped[f'{benchmark_column}_returns'].rolling(window, min_periods=window//2).mean()
            beta = grouped[f'{col}_returns'].apply(lambda x: roll_beta(x, df.loc[x.index, f'{benchmark_column}_returns']), raw=False)
            tracking_error = grouped[f'{col}_returns'].apply(lambda x: (x - df.loc[x.index, f'{benchmark_column}_returns']).rolling(window, min_periods=window//2).std(), raw=False)
            
            df[f'{col}_treynor_ratio_{window}'] = ((mean_ret - risk_free_rate) / beta * np.sqrt(annualization_factor)).reset_index(level=0, drop=True)
            df[f'{col}_information_ratio_{window}'] = ((mean_ret - bench_mean) / tracking_error).reset_index(level=0, drop=True)
    else:
        # Base rolling metrics for non-grouped data
        mean_ret = df[f'{col}_returns'].rolling(window, min_periods=window//2).mean()
        std_ret = df[f'{col}_returns'].rolling(window, min_periods=window//2).std()
        downside_std = roll_downside_std(df[f'{col}_returns'])
        
        # Assign metrics
        df[f'{col}_sharpe_ratio_{window}'] = (mean_ret - risk_free_rate) / std_ret * np.sqrt(annualization_factor)
        df[f'{col}_sortino_ratio_{window}'] = (mean_ret - risk_free_rate) / downside_std * np.sqrt(annualization_factor)
        df[f'{col}_volatility_annualized_{window}'] = std_ret * np.sqrt(annualization_factor)
        df[f'{col}_omega_ratio_{window}'] = roll_omega(df[f'{col}_returns'])
        df[f'{col}_skewness_{window}'] = df[f'{col}_returns'].rolling(window, min_periods=window//2).apply(lambda x: stats.skew(x, nan_policy='omit'), raw=True)
        df[f'{col}_kurtosis_{window}'] = df[f'{col}_returns'].rolling(window, min_periods=window//2).apply(lambda x: stats.kurtosis(x, nan_policy='omit'), raw=True)
        
        if benchmark_column:
            bench_mean = df[f'{benchmark_column}_returns'].rolling(window, min_periods=window//2).mean()
            beta = roll_beta(df[f'{col}_returns'], df[f'{benchmark_column}_returns'])
            tracking_error = (df[f'{col}_returns'] - df[f'{benchmark_column}_returns']).rolling(window, min_periods=window//2).std()
            
            df[f'{col}_treynor_ratio_{window}'] = (mean_ret - risk_free_rate) / beta * np.sqrt(annualization_factor)
            df[f'{col}_information_ratio_{window}'] = (mean_ret - bench_mean) / tracking_error
    
    # Drop temporary returns columns
    df.drop(columns=[f'{col}_returns'], inplace=True)
    if benchmark_column:
        df.drop(columns=[f'{benchmark_column}_returns'], inplace=True)
    
    return df


def _augment_rolling_risk_metrics_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str,
    window: int,
    risk_free_rate: float,
    benchmark_column: Optional[str],
    annualization_factor: int
) -> pd.DataFrame:
    """Optimized Polars implementation of rolling risk metrics calculation with flexible column ordering."""
    
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
    
    # Calculate returns and precompute masks
    returns_col = f'{col}_returns'
    df = df.with_columns(
        (pl.col(col).log() - pl.col(col).log().shift(1)).alias(returns_col),
        (pl.col(col).log() - pl.col(col).log().shift(1) > 0).cast(pl.Float64).alias('pos_mask'),
        (pl.col(col).log() - pl.col(col).log().shift(1) < 0).cast(pl.Float64).alias('neg_mask')
    )
    if benchmark_column:
        df = df.with_columns(
            (pl.col(benchmark_column).log() - pl.col(benchmark_column).log().shift(1)).alias(f'{benchmark_column}_returns')
        )
    
    # Define rolling metrics as expressions
    mean_ret = pl.col(returns_col).rolling_mean(window, min_periods=window//2)
    std_ret = pl.col(returns_col).rolling_std(window, min_periods=window//2)
    
    # Optimized expressions
    exprs = [
        # Sharpe
        ((mean_ret - risk_free_rate) / std_ret * pl.lit(np.sqrt(annualization_factor))).alias(f'{col}_sharpe_ratio_{window}'),
        # Volatility
        (std_ret * pl.lit(np.sqrt(annualization_factor))).alias(f'{col}_volatility_annualized_{window}'),
    ]
    
    # Downside std: use masked returns
    downside_ret = (pl.col(returns_col) * pl.col('neg_mask')).alias('downside_ret')
    downside_std_expr = (
        downside_ret.rolling_std(window, min_periods=window//2)
        .alias('downside_std_temp')
    )
    
    # Omega: use masked sums
    pos_sum = (pl.col(returns_col) * pl.col('pos_mask')).rolling_sum(window, min_periods=window//2)
    neg_sum = (pl.col(returns_col) * pl.col('neg_mask')).rolling_sum(window, min_periods=window//2).abs()
    omega_expr = (
        (pos_sum / neg_sum)
        .replace([np.inf, -np.inf], np.nan)
        .alias(f'{col}_omega_ratio_{window}')
    )
    
    # Skew and Kurtosis
    skew_expr = (
        pl.col(returns_col)
        .rolling_map(
            lambda x: stats.skew(x, nan_policy='omit') if x.is_not_null().sum() >= window//2 else np.nan,
            window_size=window,
            min_periods=window//2
        )
        .alias(f'{col}_skewness_{window}')
    )
    kurt_expr = (
        pl.col(returns_col)
        .rolling_map(
            lambda x: stats.kurtosis(x, nan_policy='omit') if x.is_not_null().sum() >= window//2 else np.nan,
            window_size=window,
            min_periods=window//2
        )
        .alias(f'{col}_kurtosis_{window}')
    )
    
    if group_names:
        # Apply grouped rolling calculations
        df = df.with_columns(
            downside_ret  # Precompute downside returns
        ).with_columns(
            downside_std_expr.over(group_names),
            omega_expr.over(group_names),
            skew_expr.over(group_names),
            kurt_expr.over(group_names)
        ).with_columns(
            [e.over(group_names) for e in exprs] + [
                ((mean_ret - risk_free_rate) / pl.col('downside_std_temp') * pl.lit(np.sqrt(annualization_factor)))
                .over(group_names)
                .alias(f'{col}_sortino_ratio_{window}')
            ]
        ).drop('downside_std_temp')
        
        if benchmark_column:
            bench_col = f'{benchmark_column}_returns'
            beta_expr = (
                pl.col(returns_col).rolling_cov(pl.col(bench_col), window, min_periods=window//2) /
                pl.col(bench_col).rolling_var(window, min_periods=window//2)
            ).alias('beta_temp')
            tracking_error_expr = (
                (pl.col(returns_col) - pl.col(bench_col))
                .rolling_std(window, min_periods=window//2)
                .alias('tracking_error_temp')
            )
            df = df.with_columns(
                beta_expr.over(group_names),
                tracking_error_expr.over(group_names)
            ).with_columns([
                ((mean_ret - risk_free_rate) / pl.col('beta_temp') * pl.lit(np.sqrt(annualization_factor)))
                .over(group_names)
                .alias(f'{col}_treynor_ratio_{window}'),
                ((mean_ret - pl.col(bench_col).rolling_mean(window, min_periods=window//2)) / 
                 pl.col('tracking_error_temp'))
                .over(group_names)
                .alias(f'{col}_information_ratio_{window}')
            ]).drop(['beta_temp', 'tracking_error_temp'])
    else:
        # Non-grouped rolling calculations
        df = df.with_columns(
            downside_ret  # Precompute downside returns
        ).with_columns(
            downside_std_expr,
            omega_expr,
            skew_expr,
            kurt_expr
        ).with_columns(
            exprs + [
                ((mean_ret - risk_free_rate) / pl.col('downside_std_temp') * pl.lit(np.sqrt(annualization_factor)))
                .alias(f'{col}_sortino_ratio_{window}')
            ]
        ).drop('downside_std_temp')
        
        if benchmark_column:
            bench_col = f'{benchmark_column}_returns'
            beta_expr = (
                pl.col(returns_col).rolling_cov(pl.col(bench_col), window, min_periods=window//2) /
                pl.col(bench_col).rolling_var(window, min_periods=window//2)
            ).alias('beta_temp')
            tracking_error_expr = (
                (pl.col(returns_col) - pl.col(bench_col))
                .rolling_std(window, min_periods=window//2)
                .alias('tracking_error_temp')
            )
            df = df.with_columns(
                beta_expr,
                tracking_error_expr
            ).with_columns([
                ((mean_ret - risk_free_rate) / pl.col('beta_temp') * pl.lit(np.sqrt(annualization_factor)))
                .alias(f'{col}_treynor_ratio_{window}'),
                ((mean_ret - pl.col(bench_col).rolling_mean(window, min_periods=window//2)) / 
                 pl.col('tracking_error_temp'))
                .alias(f'{col}_information_ratio_{window}')
            ]).drop(['beta_temp', 'tracking_error_temp'])
    
    # Drop temporary columns
    drop_cols = [f'{col}_returns', 'pos_mask', 'neg_mask', 'downside_ret']
    if benchmark_column:
        drop_cols.append(f'{benchmark_column}_returns')
    df = df.drop(drop_cols)
    
    # Dynamically build column order: original columns + metrics in Pandas order
    original_cols = [c for c in pandas_df.columns if c not in drop_cols]  # Keep input columns
    metric_cols = [
        f'{close_column}_sharpe_ratio_{window}',
        f'{close_column}_sortino_ratio_{window}',
        f'{close_column}_volatility_annualized_{window}',
        f'{close_column}_omega_ratio_{window}',
        f'{close_column}_skewness_{window}',
        f'{close_column}_kurtosis_{window}'
    ]
    final_columns = original_cols + [c for c in metric_cols if c in df.columns]
    df = df.select(final_columns)
    
    return df.to_pandas()

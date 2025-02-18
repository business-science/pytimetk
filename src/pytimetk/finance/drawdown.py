import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union, List, Tuple
from pytimetk.utils.parallel_helpers import progress_apply
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_drawdown(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''The augment_drawdown function calculates the drawdown metrics for a financial time series
    using either pandas or polars engine, and returns the augmented DataFrame with peak value,
    drawdown, and drawdown percentage columns.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input data can be either a pandas DataFrame or a pandas DataFrameGroupBy object
        containing the time series data for drawdown calculation.
    date_column : str
        The name of the column containing dates or timestamps.
    close_column : str
        The column containing the values (e.g., price) to calculate drawdowns from.
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame before calculation. Default is False.
    engine : str, optional
        The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame augmented with three columns:
        - {close_column}_peak: Running maximum value up to each point
        - {close_column}_drawdown: Absolute difference from peak to current value
        - {close_column}_drawdown_pct: Percentage decline from peak to current value
        
    Notes
    -----
    Drawdown is a measure of peak-to-trough decline in a time series, typically used to assess
    the risk of a financial instrument:
    
    - Peak Value: The highest value observed up to each point in time
    - Drawdown: The absolute difference between the peak and current value
    - Drawdown Percentage: The percentage decline from the peak value
    
    Examples
    --------
    ``` {python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Single stock drawdown
    dd_df = (
        df.query("symbol == 'AAPL'")
        .augment_drawdown(
            date_column='date',
            close_column='close',
        )
    )
    dd_df.head()
    ```
    
    ``` {python}
    dd_df.groupby('symbol').plot_timeseries('date', 'close_drawdown_pct')
    ```
    
    ``` {python}
    # Multiple stocks with groupby
    dd_df = (
        df.groupby('symbol')
        .augment_drawdown(
            date_column='date',
            close_column='close',
            engine='polars'
        )
    )
    dd_df.head()
    ```
    
    ``` {python}
    dd_df.groupby('symbol').plot_timeseries('date', 'close_drawdown_pct')
    ```    
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_drawdown_pandas(data, date_column, close_column)
    elif engine == 'polars':
        ret = _augment_drawdown_polars(data, date_column, close_column)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_drawdown = augment_drawdown

def _augment_drawdown_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str
) -> pd.DataFrame:
    """Pandas implementation of drawdown calculation."""
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        col = close_column
        
        # Calculate running peak, drawdown, and drawdown percentage
        df[f'{col}_peak'] = df[col].cummax()
        df[f'{col}_drawdown'] =  df[col] - df[f'{col}_peak']
        df[f'{col}_drawdown_pct'] = df[f'{col}_drawdown'] / df[f'{col}_peak']
    
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
        col = close_column
        
        # Groupby calculations
        df[f'{col}_peak'] = df.groupby(group_names)[col].cummax()
        df[f'{col}_drawdown'] = df[col] - df[f'{col}_peak']
        df[f'{col}_drawdown_pct'] = df[f'{col}_drawdown'] / df[f'{col}_peak']
    
    return df


def _augment_drawdown_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str
) -> pd.DataFrame:
    """Polars implementation of drawdown calculation."""
    
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
    
    if group_names:
        # Grouped calculation
        # Step 1: Calculate peak
        peak_expr = pl.col(col).cum_max().over(group_names).alias(f'{col}_peak')
        df = df.with_columns(peak_expr)
        
        # Step 2: Calculate drawdown using the peak
        drawdown_expr = (pl.col(col) - pl.col(f'{col}_peak')).alias(f'{col}_drawdown')
        df = df.with_columns(drawdown_expr)
        
        # Step 3: Calculate drawdown percentage
        drawdown_pct_expr = (pl.col(f'{col}_drawdown') / pl.col(f'{col}_peak')).alias(f'{col}_drawdown_pct')
        df = df.with_columns(drawdown_pct_expr)
        
    else:
        # Single series calculation
        # Step 1: Calculate peak
        peak_expr = pl.col(col).cum_max().alias(f'{col}_peak')
        df = df.with_columns(peak_expr)
        
        # Step 2: Calculate drawdown using the peak
        drawdown_expr = (pl.col(col) - pl.col(f'{col}_peak')).alias(f'{col}_drawdown')
        df = df.with_columns(drawdown_expr)
        
        # Step 3: Calculate drawdown percentage
        drawdown_pct_expr = (pl.col(f'{col}_drawdown') / pl.col(f'{col}_peak')).alias(f'{col}_drawdown_pct')
        df = df.with_columns(drawdown_pct_expr)
    
    return df.to_pandas()

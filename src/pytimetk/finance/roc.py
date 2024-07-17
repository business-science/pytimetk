import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_roc(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    close_column: str, 
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    start_index: int = 0,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    """
    Adds rate of change (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to add percentage differenced columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to sort the data before adding the percentage differenced values.
    close_column : str
        The `close_column` parameter in the `augment_qsmomentum` function refers to the column in the input
        DataFrame that contains the closing prices of the financial instrument or asset for which you want
        to calculate the momentum. 
    periods : int or tuple or list, optional
        The `periods` parameter is an integer, tuple, or list that specifies the 
        periods to shift values when percentage differencing. 
        
        - If it is an integer, the function will add that number of percentage differences 
          values for each column specified in the `value_column` parameter. 
        
        - If it is a tuple, it will generate percentage differences from the first to the second 
          value (inclusive). 
        
        - If it is a list, it will generate percentage differences based on the values in the list.
    start_index : int, optional
        The `start_index` parameter is an integer that specifies the starting index for the percentage difference calculation. 
        Default is 0 which is the last element in the group.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting percentage differences. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
        for augmenting percentage diffs. This can be faster than using "pandas" for large 
        datasets. 

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with percentage differenced columns added to it.
        
    Notes
    -----
    The rate of change (ROC) calculation is a momentum indicator that measures the percentage change in price between the current price and the price a certain number of periods ago. The ROC indicator is used to identify the speed and direction of price movements. It is calculated as follows:
    
    ROC = [(Close - Close n periods ago) / (Close n periods ago)] 
    
    When `start_index` is used, the formula becomes:
    
    ROC = [(Close start_index periods ago - Close n periods ago) / (Close n periods ago)] 
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])

    df.glimpse()
    ```
    
    ```{python}
    # Example 1 - Add 7 roc values for a single DataFrame object, pandas engine
    roc_df = (
        df 
            .query('symbol == "GOOG"') 
            .augment_roc(
                date_column='date',
                close_column='close',
                periods=(1, 7),
                engine='pandas'
            )
    )
    roc_df.glimpse()
    ```
    
    ```{python}
    # Example 2 - Add 2 ROC with start index 21 using GroupBy object, polars engine
    roc_df = (
        df 
            .groupby('symbol')
            .augment_roc(
                date_column='date',
                close_column='close',
                periods=[63, 252],
                start_index=21,
                engine='polars'
            )
    )
    roc_df
    ```
    """
    
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    # Check start_index > periods
    if start_index >= min(periods):
        raise ValueError("start_index must be less than the minimum value in periods.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    # Make close column iterable
    if isinstance(close_column, str):
        close_column = [close_column]
    
    # Make periods iterable
    if isinstance(periods, int):
        periods = [periods]  # Convert to a list with a single value
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")
    
    # Augment the data
    if engine == 'pandas':
        ret = _augment_roc_pandas(data, date_column, close_column, periods, start_index=start_index)
    elif engine == 'polars':
        ret = _augment_roc_polars(data, date_column, close_column, periods, start_index=start_index)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_roc = augment_roc


def _augment_roc_pandas(
    data, date_column, close_column, periods, start_index
) -> pd.DataFrame:

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    if isinstance(data, pd.DataFrame):

        df = data.copy()

        df.sort_values(by=[date_column], inplace=True)

        for col in close_column:
            for period in periods:
                if start_index == 0:
                    df[f'{col}_roc_{start_index}_{period}'] = df[col].pct_change(period)
                else:
                    df[f'{col}_roc_{start_index}_{period}'] = (df[col].shift(start_index) / df[col].shift(period)) - 1
        

    # GROUPED EXTENSION - If data is a GroupBy object, add differences by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        df.sort_values(by=[*group_names, date_column], inplace=True)

        
        for col in close_column:
            for period in periods:
                if start_index == 0:
                    df[f'{col}_roc_{start_index}_{period}'] = df.groupby(group_names)[col].pct_change(period)
               
                else: 
                    df[f'{col}_roc_{start_index}_{period}'] = (df.groupby(group_names)[col].shift(start_index) / df.groupby(group_names)[col].shift(period)) - 1
                    

    return df

def _augment_roc_polars(
    data, date_column, close_column, periods, start_index
) -> pl.DataFrame:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    roc_foo = pl.col(date_column).shift(1).alias("_diff_1")

    period_exprs = []

    
    for col in close_column:
        for period in periods:
            if start_index == 0:
                period_expr = (
                    (pl.col(col) / pl.col(col).shift(period)) - 1
                ).alias(f"{col}_roc_{start_index}_{period}")
                period_exprs.append(period_expr)
            else:
                period_expr = (
                    (pl.col(col).shift(start_index) / pl.col(col).shift(period)) - 1
                ).alias(f"{col}_roc_{start_index}_{period}")
                period_exprs.append(period_expr)

    # Select columns
    selected_columns = [roc_foo] + period_exprs

    # Drop the first column by position (index)
    selected_columns = selected_columns[1:]

    # Select the columns
    df = pl.DataFrame(pandas_df)
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        out_df = df.group_by(data.grouper.names, maintain_order=True).agg(selected_columns)
        out_df = out_df.explode(out_df.columns[len(data.grouper.names):])
        out_df = out_df.drop(data.grouper.names)
    else: # a dataframe
        out_df = df.select(selected_columns)

    # Concatenate the DataFrames horizontally
    df = pl.concat([df, out_df], how="horizontal").to_pandas()

    return df

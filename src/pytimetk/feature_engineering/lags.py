import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_lags(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    lags: Union[int, Tuple[int, int], List[int]] = 1,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    """
    Adds lags to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_lags` function takes a Pandas DataFrame or GroupBy object, a 
    date column, a value column or list of value columns, and a lag or list of 
    lags, and adds lagged versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to add lagged columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to sort the data before adding the lagged values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you 
        want to add lagged values for. It can be either a single column name 
        (string) or a list of column names.
    lags : int or tuple or list, optional
        The `lags` parameter is an integer, tuple, or list that specifies the 
        number of lagged values to add to the DataFrame. 
        
        - If it is an integer, the function will add that number of lagged 
          values for each column specified in the `value_column` parameter. 
        
        - If it is a tuple, it will generate lags from the first to the second 
          value (inclusive). 
        
        - If it is a list, it will generate lags based on the values in the list.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting lags. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for augmenting lags. This can be faster than using "pandas" for large 
          datasets. 

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with lagged columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```
    
    ```{python}
    # Example 1 - Add 7 lagged values for a single DataFrame object, pandas engine
    lagged_df_single = (
        df 
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=(1, 7),
                engine='pandas'
            )
    )
    lagged_df_single
    ```
    ```{python}
    # Example 2 - Add a single lagged value of 2 for each GroupBy object, polars engine
    lagged_df = (
        df 
            .groupby('id')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=(1, 3),
                engine='polars'
            )
    )
    lagged_df
    ```

    ```{python}
    # Example 3 add 2 lagged values, 2 and 4, for a single DataFrame object, pandas engine
    lagged_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=[2, 4],
                engine='pandas'
            )
    )
    lagged_df_single_two
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column, require_numeric_dtype = False)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    if engine == 'pandas':
        ret = _augment_lags_pandas(data, date_column, value_column, lags)
    elif engine == 'polars':
        ret = _augment_lags_polars(data, date_column, value_column, lags)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_lags = augment_lags


def _augment_lags_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    lags: Union[int, Tuple[int, int], List[int]] = 1
) -> pd.DataFrame:
    
    if isinstance(value_column, str):
        value_column = [value_column]

    if isinstance(lags, int):
        lags = [lags]
    elif isinstance(lags, tuple):
        lags = list(range(lags[0], lags[1] + 1))
    elif not isinstance(lags, list):
        raise TypeError(f"Invalid lags specification: type: {type(lags)}. Please use int, tuple, or list.")

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, apply lag function
    if isinstance(data, pd.DataFrame):

        df = data.copy()

        for col in value_column:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # GROUPED EXTENSION - If data is a GroupBy object, add lags by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        for col in value_column:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby(group_names)[col].shift(lag)

    return df

def _augment_lags_polars(
    data: Union[pl.DataFrame, pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    lags: Union[int, Tuple[int, int], List[int]] = 1
) -> pl.DataFrame:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    if isinstance(value_column, str):
        value_column = [value_column]

    lag_foo = pl.col(date_column).shift(1).alias(f"_lag_1")

    if isinstance(lags, int):
        lags = [lags]  # Convert to a list with a single value
    elif isinstance(lags, tuple):
        lags = list(range(lags[0], lags[1] + 1))
    elif not isinstance(lags, list):
        raise TypeError(f"Invalid lags specification: type: {type(lags)}. Please use int, tuple, or list.")

    lag_exprs = []

    for col in value_column:
        for lag in lags:
            lag_expr = pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            lag_exprs.append(lag_expr)

    # Select columns
    selected_columns = [lag_foo] + lag_exprs

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

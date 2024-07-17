import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_diffs(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    normalize: bool = False,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    """
    Adds differences and percentage difference (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_diffs` function takes a Pandas DataFrame or GroupBy object, a 
    date column, a value column or list of value columns, and a period or list of 
    periods, and adds differenced versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to add differenced columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to sort the data before adding the differenced values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you 
        want to add differences values for. It can be either a single column name 
        (string) or a list of column names.
    periods : int or tuple or list, optional
        The `periods` parameter is an integer, tuple, or list that specifies the 
        periods to shift values when differencing. 
        
        - If it is an integer, the function will add that number of differences 
          values for each column specified in the `value_column` parameter. 
        
        - If it is a tuple, it will generate differences from the first to the second 
          value (inclusive). 
        
        - If it is a list, it will generate differences based on the values in the list.
    normalize : bool, optional
        The `normalize` parameter is used to specify whether to normalize the 
        differenced values as a percentage difference. Default is False.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting differences. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for augmenting diffs. This can be faster than using "pandas" for large 
          datasets. 

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with differenced columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```
    
    ```{python}
    # Example 1 - Add 7 differenced values for a single DataFrame object, pandas engine
    diffed_df_single = (
        df 
            .query('id == "D10"')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=(1, 7),
                engine='pandas'
            )
    )
    diffed_df_single.glimpse()
    ```
    ```{python}
    # Example 2 - Add a single differenced value of 2 for each GroupBy object, polars engine
    diffed_df = (
        df 
            .groupby('id')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=2,
                engine='polars'
            )
    )
    diffed_df
    ```

    ```{python}
    # Example 3 add 2 differenced values, 2 and 4, for a single DataFrame object, pandas engine
    diffed_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=[2, 4],
                engine='pandas'
            )
    )
    diffed_df_single_two
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    if engine == 'pandas':
        ret = _augment_diffs_pandas(data, date_column, value_column, periods, normalize=normalize)
    elif engine == 'polars':
        ret = _augment_diffs_polars(data, date_column, value_column, periods, normalize=normalize)
        
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
        
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    # Sort index
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_diffs = augment_diffs


def _augment_diffs_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    normalize: bool = False
) -> pd.DataFrame:
    
    if isinstance(value_column, str):
        value_column = [value_column]

    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    if isinstance(data, pd.DataFrame):

        df = data.copy()

        df.sort_values(by=[date_column], inplace=True)

        if normalize:
            for col in value_column:
                for period in periods:
                    df[f'{col}_pctdiff_{period}'] = df[col].pct_change(period)
        
        else:
            for col in value_column:
                for period in periods:
                    df[f'{col}_diff_{period}'] = df[col].diff(period)

    # GROUPED EXTENSION - If data is a GroupBy object, add differences by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        df.sort_values(by=[*group_names, date_column], inplace=True)

        if normalize:
            for col in value_column:
                for period in periods:
                    df[f'{col}_pctdiff_{period}'] = df.groupby(group_names)[col].pct_change(period)
        else: 
            for col in value_column:
                for period in periods:
                    df[f'{col}_diff_{period}'] = df.groupby(group_names)[col].diff(period)

    return df

def _augment_diffs_polars(
    data: Union[pl.DataFrame, pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    normalize: bool = False
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

    if isinstance(value_column, str):
        value_column = [value_column]

    diff_foo = pl.col(date_column).shift(1).alias("_diff_1")

    if isinstance(periods, int):
        periods = [periods]  # Convert to a list with a single value
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")

    period_exprs = []

    if normalize:
        for col in value_column:
            for period in periods:
                period_expr = (
                    (pl.col(col) / pl.col(col).shift(period)) - 1
                ).alias(f"{col}_pctdiff_{period}")
                period_exprs.append(period_expr)
    else:
        for col in value_column:
            for period in periods:
                period_expr = (pl.col(col) - pl.col(col).shift(period)).alias(f"{col}_diff_{period}")
                period_exprs.append(period_expr)

    # Select columns
    selected_columns = [diff_foo] + period_exprs

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

import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_close_column
from pytimetk.utils.memory_helpers import reduce_memory_usage



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
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_close_column(data, close_column)
    check_date_column(data, date_column)
    
    # Check start_index > periods
    if start_index < max(periods):
        raise ValueError("start_index must be greater than the maximum value in periods.")
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _augment_roc_pandas(data, date_column, close_column, periods, start_index=start_index)
    elif engine == 'polars':
        ret = _augment_roc_polars(data, date_column, close_column, periods, start_index=start_index)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_roc = augment_roc


def _augment_roc_pandas(
    data, date_column, close_column, periods, start_index
) -> pd.DataFrame:
    
    if isinstance(close_column, str):
        close_column = [close_column]

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

        for col in close_column:
            for period in periods:
                if start_index == 0:
                    df[f'{col}_roc_{period}'] = df[col].pct_change(period)
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
                    df[f'{col}_roc_{period}'] = df.groupby(group_names)[col].pct_change(period)
               
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

    if isinstance(close_column, str):
        close_column = [close_column]

    diff_foo = pl.col(date_column).shift(1).suffix("_diff_1")

    if isinstance(periods, int):
        periods = [periods]  # Convert to a list with a single value
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")

    period_exprs = []

    
    for col in close_column:
        for period in periods:
            if start_index == 0:
                period_expr = (
                    (pl.col(col) / pl.col(col).shift(period)) - 1
                ).alias(f"{col}_roc_{period}")
                period_exprs.append(period_expr)
            else:
                period_expr = (pl.col(col).shift(start_index) - pl.col(col).shift(period)).alias(f"{col}_roc_{start_index}_{period}")
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

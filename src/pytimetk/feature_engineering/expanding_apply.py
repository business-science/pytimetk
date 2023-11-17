import pandas as pd
import polars as pl
import pandas_flavor as pf
import numpy as np
import inspect
import warnings

from typing import Union, Optional, Callable, Tuple, List

from pathos.multiprocessing import ProcessingPool
from functools import partial

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads
from pytimetk.utils.polars_helpers import update_dict
from pytimetk.utils.memory_helpers import reduce_memory_usage


@pf.register_dataframe_method
def augment_expanding_apply(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    window_func: Union[Tuple[str, Callable], List[Tuple[str, Callable]]], 
    min_periods: Optional[int] = None,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False, 
) -> pd.DataFrame:
    '''
    Apply one or more DataFrame-based expanding functions to one or more columns of a DataFrame.
        
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    window_func : Union[Tuple[str, Callable], List[Tuple[str, Callable]]]
        The `window_func` parameter in the `augment_expanding_apply` function 
        specifies the function(s) that operate on a expanding window with the 
        consideration of multiple columns.
    
        The specification can be:
        - A tuple where the first element is a string representing the function's name and the second element is the callable function itself.
        - A list of such tuples for multiple functions.
    
        Note: For functions targeting only a single value column without the need for 
        contextual data from other columns, consider using the `augment_expanding` 
        function in this library.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window 
        size. If set, a value will be produced even if fewer observations are 
        present than the window size.
    threads : int, optional, default 1
        Number of threads to use for parallel processing. If `threads` is set to 
        1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
    show_progress : bool, optional, default True
        If `True`, a progress bar will be displayed during parallel processing.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.

        
    Returns
    -------
    pd.DataFrame
        The `augment_expanding` function returns a DataFrame with new columns 
        for each applied function, window size, and value column.
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    ```
        
    ```{python}
    # Example showcasing the expanding correlation between two columns (`value1` and 
    # `value2`).
    # The correlation requires both columns as input.
        
    # Sample DataFrame with id, date, value1, and value2 columns.
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })
        
    # Compute the expanding correlation for each group of 'id'
    expanding_df = (
        df.groupby('id')
          .augment_expanding_apply(
            date_column='date',
            window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],  # Lambda function for correlation
            threads = 1,  # Disable parallel processing
        )
    )
    display(expanding_df)
    ```
        
    ```{python}
    # expanding Regression Example: Using `value1` as the dependent variable and 
    # `value2` and `value3` as the independent variables.
    # This example demonstrates how to perform a expanding regression using two 
    # independent variables.
    
    # Sample DataFrame with `id`, `date`, `value1`, `value2`, and `value3` columns.
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [5, 16, 24, 35, 45, 58],
        'value3': [2, 3, 6, 9, 10, 13]
    })
        
    # Define Regression Function to be applied on the expanding window.
    def regression(df):
        
        # Required module (scikit-learn) for regression.
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        X = df[['value2', 'value3']]  # Independent variables
        y = df['value1']  # Dependent variable
        model.fit(X, y)
        ret = pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])
            
        return ret # Return intercept and slope as a Series
            
    # Compute the expanding regression for each group of `id`
    result_df = (
        df.groupby('id')
        .augment_expanding_apply(
            date_column='date',
            window_func=[('regression', regression)],
            threads = 1
        )
        .dropna()
    )
    
    # Format the results to have each regression output (slope and intercept) in 
    #  separate columns.
    regression_wide_df = pd.concat(result_df['expanding_regression'].to_list(), axis=1).T
    regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)
    display(regression_wide_df)
    ```
    '''
    # Checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    if reduce_memory:
        data_copy = reduce_memory_usage(data_copy)
    
    # Get threads
    threads = get_threads(threads)
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the expanding calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
        
     # Set min_periods to 1 if not specified
    min_periods = 1 if min_periods is None else min_periods
    
    # Process each group in parallel
    if threads == 1:
        result_dfs = []
        for group in conditional_tqdm(grouped, total=len(grouped), desc="Processing rolling apply...", display= show_progress):
            args = group, window_func, min_periods 
            result_dfs.append(_process_single_expanding_apply_group(args))
    else:
        # Prepare to use pathos.multiprocessing
        pool = ProcessingPool(threads)
        args = [(group, window_func, min_periods) for group in grouped]
        result_dfs = list(conditional_tqdm(pool.map(_process_single_expanding_apply_group, args), 
                                        total=len(grouped), 
                                        desc="Processing rolling apply...", 
                                        display=show_progress))
    
    
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    if reduce_memory:
        result_df = reduce_memory_usage(result_df)
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_expanding_apply = augment_expanding_apply

def _process_single_expanding_apply_group(args):
    
    group, window_func, min_periods = args
    
    # Apply DataFrame-based expanding window functions
    name, group_df = group
    result_dfs = []
    for func in window_func:
        if isinstance(func, tuple):
            func_name, func = func
            new_column_name = f"expanding_{func_name}"
            group_df[new_column_name] = _expanding_apply(func, group_df, min_periods=min_periods)
        else:
            raise TypeError(f"Expected 'tuple', but got invalid function type: {type(func)}")     
                
        result_dfs.append(group_df)
        
    return pd.concat(result_dfs)
    
    

# Helper function to apply expanding calculations on a dataframe
def _expanding_apply(func, df, min_periods):
    num_rows = len(df)
    results = [np.nan] * num_rows
    
    for end_point in range(1, num_rows + 1):
        window_df = df.iloc[: end_point]
        if len(window_df) >= min_periods:
            results[end_point - 1] = func(window_df)

    return pd.DataFrame({'result': results}, index=df.index)


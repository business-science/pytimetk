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
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_rolling(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    window: Union[int, tuple, list] = 2,
    min_periods: Optional[int] = None,
    engine: str = 'pandas',
    center: bool = False,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''
    Apply one or more Series-based rolling functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy 
        object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each 
        group.
    value_column : Union[str, list]
        Column(s) to which the rolling window functions should be applied. Can 
        be a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_rolling` function specifies 
        the function(s) to be applied to the rolling windows of the value 
        column(s).

        1. It can be either:
            - A string representing the name of a standard function (e.g., 
              'mean', 'sum').
            
        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name 
              for the function and the function itself.
            - Each custom function should accept a Pandas Series as its input 
              and operate on that series.
              Example: ("range", lambda x: x.max() - x.min())
        
        (See more Examples below.)

        Note: If your function needs to operate on multiple columns (i.e., it 
              requires access to a DataFrame rather than just a Series), 
              consider using the `augment_rolling_apply` function in this library.   
    window : Union[int, tuple, list], optional, default 2
        Specifies the size of the rolling windows.
        - An integer applies the same window size to all columns in `value_column`.
        - A tuple generates windows from the first to the second value (inclusive).
        - A list of integers designates multiple window sizes for each respective 
          column.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the 
        window size. If set, a value will be produced even if fewer observations 
        are present than the window size.    
    center : bool, optional, default False
        If `True`, the rolling window will be centered on the current value. For 
        even-sized windows, the window will be left-biased. Otherwise, it uses a trailing window.
    threads : int, optional, default 1
        Number of threads to use for parallel processing. If `threads` is set to 
        1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
    show_progress : bool, optional, default True
        If `True`, a progress bar will be displayed during parallel processing.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : str, optional, default 'pandas'
        Specifies the backend computation library for augmenting expanding window 
        functions. 
    
        The options are:
            - "pandas" (default): Uses the `pandas` library.
            - "polars": Uses the `polars` library, which may offer performance 
               benefits for larger datasets.
    
    Returns
    -------
    pd.DataFrame
        The `augment_rolling` function returns a DataFrame with new columns for 
        each applied function, window size, and value column.
    
    Notes
    -----
    ## Performance
    
    This function uses parallel processing to speed up computation for large 
    datasets with many time series groups: 
    
    Parallel processing has overhead and may not be faster on small datasets.
    
    To use parallel processing, set `threads = -1` to use all available processors.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    ```
    
    ```{python}
    # Example 1 - Using a single window size and a single function name, pandas engine
    # This example demonstrates the use of both string-named functions and lambda 
    # functions on a rolling window. We specify a list of window sizes: [2,7]. 
    # As a result, the output will have computations for both window sizes 2 and 7.
    # Note - It's preferred to use built-in or configurable functions instead of 
    # lambda functions for performance reasons.

    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = [2,7],  # Specifying multiple window sizes
                window_func = [
                    'mean',  # Built-in mean function
                    ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                ],
                threads = 1,  # Disabling parallel processing
                engine = 'pandas'  # Using pandas engine
            )
    )
    display(rolled_df)
    ```
    
    ```{python}
    # Example 2 - Multiple groups, pandas engine
    # Example showcasing the use of string function names and lambda functions 
    # applied on rolling windows. The `window` tuple (1,3) will generate window 
    # sizes of 1, 2, and 3.
    # Note - It's preferred to use built-in or configurable functions instead of 
    # lambda functions for performance reasons.
    
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = (1,3),  # Specifying a range of window sizes
                window_func = [
                    'mean',  # Using built-in mean function
                    ('std', lambda x: x.std())  # Lambda function for standard deviation
                ],
                threads = 1,  # Disabling parallel processing
                engine = 'pandas'  # Using pandas engine
            )
    )
    display(rolled_df) 
    ```
    
    ```{python}
    # Example 3 - Multiple groups, polars engine
    
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = (1,3),  # Specifying a range of window sizes
                window_func = [
                    'mean',  # Using built-in mean function
                    'std',  # Using built-in standard deviation function
                ],
                engine = 'polars'  # Using polars engine
            )
    )
    display(rolled_df) 
    ```
    '''
    # Checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    # Convert string value column to list for consistency
    if isinstance(value_column, str):
        value_column = [value_column]
    
    # Validate window argument and convert it to a consistent list format
    if not isinstance(window, (int, tuple, list)):
        raise TypeError("`window` must be an integer, tuple, or list.")
    if isinstance(window, int):
        window = [window]
    elif isinstance(window, tuple):
        window = list(range(window[0], window[1] + 1))
    
    # Get threads
    threads = get_threads(threads)    
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Call the function to augment rolling window columns using the specified engine
    if engine == 'pandas':
        ret = _augment_rolling_pandas(
            data, 
            date_column, 
            value_column, 
            window_func, 
            window, 
            min_periods, 
            center, 
            threads, 
            show_progress,
            **kwargs
        )
    elif engine == 'polars':
        ret = _augment_rolling_polars(
            data, 
            date_column, 
            value_column, 
            window_func,
            window, 
            min_periods,
            center,
            threads,
            show_progress,
            **kwargs
        )
        
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
        
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling
    
def _augment_rolling_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    window: Union[int, tuple, list] = 2,
    min_periods: Optional[int] = None,
    center: bool = False,
    threads: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the rolling calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.groupby(group_names)
        
        # Check if the data is grouped and threads are set to 1. If true, handle it without parallel processing.
        if threads == 1:
            func = partial(_process_single_roll, 
                           value_column=value_column, 
                           window_func=window_func, 
                           window=window, 
                           min_periods=min_periods, 
                           center=center, **kwargs)

            # Use tqdm to display progress for the loop
            result_dfs = [func(group) for _, group in conditional_tqdm(grouped, total=len(grouped), desc="Calculating Rolling...", display=show_progress)]
        else:
            # Prepare to use pathos.multiprocessing
            pool = ProcessingPool(threads)

            # Use partial to "freeze" arguments for _process_single_roll
            func = partial(_process_single_roll, 
                           value_column=value_column, 
                           window_func=window_func, 
                           window=window, 
                           min_periods=min_periods, 
                           center=center, **kwargs)

            result_dfs = list(conditional_tqdm(pool.map(func, (group for _, group in grouped)), 
                                               total=len(grouped), 
                                               desc="Calculating Rolling...", 
                                               display=show_progress))
    else:
        result_dfs = [_process_single_roll(data_copy, value_column, window_func, window, min_periods, center, **kwargs)]
    
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    return result_df

def _process_single_roll(group_df, value_column, window_func, window, min_periods, center, **kwargs):
    result_dfs = []
    for value_col in value_column:
        for window_size in window:
            min_periods = window_size if min_periods is None else min_periods
            for func in window_func:
                if isinstance(func, tuple):
                    # Ensure the tuple is of length 2 and begins with a string
                    if len(func) != 2:
                        raise ValueError(f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}.")
                    if not isinstance(func[0], str):
                        raise TypeError(f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}.")
                
                    user_func_name, func = func
                    new_column_name = f"{value_col}_rolling_{user_func_name}_win_{window_size}"
                        
                    # Try handling a lambda function of the form lambda x: x
                    if inspect.isfunction(func) and len(inspect.signature(func).parameters) == 1:
                        try:
                            # Construct rolling window column
                            group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).apply(func, raw=True)
                        except Exception as e:
                            raise Exception(f"An error occurred during the operation of the `{user_func_name}` function in Pandas. Error: {e}")

                    # Try handling a configurable function (e.g. pd_quantile)
                    elif isinstance(func, tuple) and func[0] == 'configurable':
                        try:
                            # Configurable function should return 4 objects
                            _, func_name, default_kwargs, user_kwargs = func
                        except Exception as e:
                            raise ValueError(f"Unexpected function format. Expected a tuple with format ('configurable', func_name, default_kwargs, user_kwargs). Received: {func}. Original error: {e}")
                        
                        try:
                            # Define local values that may be required by configurable functions.
                            # If adding a new configurable function in utils.pandas_helpers that necessitates 
                            # additional local values, consider updating this dictionary accordingly.
                            local_values = {}
                            # Combine local values with user-provided parameters for the configurable function
                            user_kwargs.update(local_values)
                            # Update the default configurable parameters (without adding new keys)
                            default_kwargs = update_dict(default_kwargs, user_kwargs)
                        except Exception as e:
                            raise ValueError("Error encountered while updating parameters for the configurable function `{func_name}` passed to `window_func`: {e}")
                        
                        try:
                            # Get the rolling window function 
                            rolling_function = getattr(group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs), func_name, None)
                        except Exception as e:
                            raise AttributeError(f"The function `{func_name}` tried to access a non-existent attribute or method in Pandas. Error: {e}")

                        if rolling_function:
                            try:
                                # Apply rolling function to data and store in new column
                                group_df[new_column_name] = rolling_function(**default_kwargs)
                            except Exception as e:
                                raise Exception(f"Failed to construct the rolling window column using function `{user_func_name}`. Error: {e}")
                    else:
                        raise TypeError(f"Unexpected function format for `{user_func_name}`.")
            
                elif isinstance(func, str):
                    new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                    # Get the rolling function (like mean, sum, etc.) specified by `func` for the given column and window settings
                    if func == "quantile":
                        new_column_name = f"{value_col}_rolling_{func}_50_win_{window_size}"
                        group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).quantile(q=0.5)
                        warnings.warn(
                            "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                            "For more control over the quantile value, consider using the function `pd_quantile()`. "
                            "For example: ('quantile_75', pd_quantile(q=0.75))."
                        )
                    else:
                        rolling_function = getattr(group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs), func, None)
                        # Apply rolling function to data and store in new column
                        if rolling_function:
                            group_df[new_column_name] = rolling_function()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                else:
                    raise TypeError(f"Invalid function type: {type(func)}") 
          
        result_dfs.append(group_df)
    return pd.concat(result_dfs)

def _augment_rolling_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    window: Union[int, tuple, list] = 2,
    min_periods: Optional[int] = None,
    center: bool = False,
    threads: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    
    
    # Retrieve the group column names if the input data is a GroupBy object
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
    else: 
        group_names = None

    # Convert data into a Pandas DataFrame format for processing
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        pandas_df = data.copy()
    else:
        raise ValueError("Data must be a Pandas DataFrame or Pandas GroupBy object.")
    
    # Initialize lists to store rolling expressions and new column names  
    rolling_exprs = []
    new_column_names = []

    # Construct rolling expressions for each column and function combination
    for col in value_column:
        for window_size in window:
            min_periods = window_size if min_periods is None else min_periods
            for func in window_func:
                
                # Handle functions passed as tuples
                if isinstance(func, tuple):
                    # Ensure the tuple is of length 2 and begins with a string
                    if len(func) != 2:
                        raise ValueError(f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}.")
                    if not isinstance(func[0], str):
                        raise TypeError(f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}.")
                    
                    user_func_name, func = func
                    new_column_name = f"{col}_rolling_{user_func_name}_win_{window_size}"
                    
                    # Try handling a lambda function of the form lambda x: x
                    if inspect.isfunction(func) and len(inspect.signature(func).parameters) == 1:
                        try:
                            # Construct rolling window expression
                            rolling_expr = pl.col(col) \
                                .cast(pl.Float64) \
                                .rolling_map(
                                    function=func,
                                    window_size=window_size, 
                                    min_periods=min_periods
                                )
                        except Exception as e:
                            raise Exception(f"An error occurred during the operation of the `{user_func_name}` function in Polars. Error: {e}")
    
                    # Try handling a configurable function (e.g. pl_quantile) if it is not a lambda function
                    elif isinstance(func, tuple) and func[0] == 'configurable':
                        try:
                            # Configurable function should return 4 objects
                            _, func_name, default_kwargs, user_kwargs = func
                        except Exception as e:
                            raise ValueError(f"Unexpected function format. Expected a tuple with format ('configurable', func_name, default_kwargs, user_kwargs). Received: {func}. Original error: {e}")
                        
                        try:
                            # Define local values that may be required by configurable functions.
                            # If adding a new configurable function in utils.polars_helpers that necessitates 
                            # additional local values, consider updating this dictionary accordingly.
                            local_values = {
                                'window_size': window_size,
                                'min_periods': min_periods
                            }
                            # Combine local values with user-provided parameters for the configurable function
                            user_kwargs.update(local_values)
                            # Update the default configurable parameters (without adding new keys)
                            default_kwargs = update_dict(default_kwargs, user_kwargs)
                        except Exception as e:
                            raise ValueError("Error encountered while updating parameters for the configurable function `{func_name}` passed to `window_func`: {e}")
                        
                        try:
                            # Construct rolling window expression
                            rolling_expr = getattr(pl.col(col), f"rolling_{func_name}")(**default_kwargs)
                        except AttributeError as e:
                            raise AttributeError(f"The function `{user_func_name}` tried to access a non-existent attribute or method in Polars. Error: {e}")
                        except Exception as e:
                            raise Exception(f"Error during the execution of `{user_func_name}` in Polars. Error: {e}")
                    
                    else:
                        raise TypeError(f"Unexpected function format for `{user_func_name}`.")
                    
                    rolling_expr = rolling_expr.alias(new_column_name)

                # Standard Functions: "mean", "std"
                elif isinstance(func, str):
                    func_name = func
                    new_column_name = f"{col}_rolling_{func_name}_win_{window_size}"
                    if not hasattr(pl.col(col), f"{func_name}"):
                        raise ValueError(f"{func_name} is not a recognized function for Polars.")
                    
                    # Construct rolling window expression and handle specific case of 'skew'
                    if func_name == "skew":
                        rolling_expr = getattr(pl.col(col), f"rolling_{func_name}")(window_size=window_size)
                    elif func_name == "quantile":
                            new_column_name = f"{col}_rolling_{func}_50_win_{window_size}"
                            rolling_expr = getattr(pl.col(col), f"rolling_{func_name}")(quantile=0.5, window_size=window_size, min_periods=min_periods, interpolation='midpoint')
                            warnings.warn(
                                "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                                "For more control over the quantile value, consider using the function `pl_quantile()`. "
                                "For example: ('quantile_75', pl_quantile(quantile=0.75))."
                            )
                    else: 
                        rolling_expr = getattr(pl.col(col), f"rolling_{func_name}")(window_size=window_size, min_periods=min_periods)

                    rolling_expr = rolling_expr.alias(new_column_name)
                    
                else:
                    raise TypeError(f"Invalid function type: {type(func)}")
                
                # Add constructed expressions and new column names to respective lists
                rolling_exprs.append(rolling_expr)
                new_column_names.append(new_column_name)
    
    # Select the columns
    selected_columns = rolling_exprs
    
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


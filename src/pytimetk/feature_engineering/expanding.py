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
def augment_expanding(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    engine: str = 'pandas',
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''
    Apply one or more Series-based expanding functions to one or more columns of a DataFrame.
        
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    value_column : Union[str, list]
        Column(s) to which the expanding window functions should be applied. Can be 
        a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_expanding` function specifies 
        the function(s) to be applied to the expanding windows of the value column(s).
    
        1. It can be either:
            - A string representing the name of a standard function (e.g., 'mean', 'sum').
                
        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name for 
              the function and the function itself.
            - Each custom function should accept a Pandas Series as its input and 
              operate on that series. Example: ("range", lambda x: x.max() - x.min())
            
        (See more Examples below.)
    
        Note: If your function needs to operate on multiple columns (i.e., it 
              requires access to a DataFrame rather than just a Series), consider 
              using the `augment_expanding_apply` function in this library.   
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window 
        size. If set, a value will be produced even if fewer observations are 
        present than the window size.
    engine : str, optional, default 'pandas'
        Specifies the backend computation library for augmenting expanding window 
        functions. 
        
        The options are:
            - "pandas" (default): Uses the `pandas` library.
            - "polars": Uses the `polars` library, which may offer performance 
               benefits for larger datasets.
    threads : int, optional, default 1
        Number of threads to use for parallel processing. If `threads` is set to 
        1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
    show_progress : bool, optional, default True
        If `True`, a progress bar will be displayed during parallel processing.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.    
    **kwargs : additional keyword arguments
        Additional arguments passed to the `pandas.Series.expanding` method when 
        using the Pandas engine.
        
    Returns
    -------
    pd.DataFrame
        The `augment_expanding` function returns a DataFrame with new columns for 
        each applied function, window size, and value column.
    
    Notes
    -----
    
    ## Performance
    
    ### Polars Engine (3X faster than Pandas)
    
    In most cases, the `polars` engine will be faster than the `pandas` engine. Speed tests indicate 3X or more. 
    
    ### Parallel Processing (Pandas Engine Only)
    
    This function uses parallel processing to speed up computation for large 
    datasets with many time series groups: 
    
    Parallel processing has overhead and may not be faster on small datasets.
    
    To use parallel processing, set `threads = -1` to use all available processors.
    
    Examples
    --------
    
    ```{python}
    # Example 1 - Pandas Backend for Expanding Window Functions
    # This example demonstrates the use of string-named functions 
    # on an expanding window using the Pandas backend for computations.
        
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function,
                     ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function
                        
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
                threads = 1,  # Disable parallel processing
                show_progress = True,  # Display a progress bar
                )
    )
    display(expanded_df)
    ```
    
    
    ```{python}
    # Example 2 - Polars Backend for Expanding Window Functions using Built-Ins 
    #             (538X Faster than Pandas)
    #  This example demonstrates the use of string-named functions and configurable 
    #  functions using the Polars backend for computations. Configurable functions, 
    #  like pl_quantile, allow the use of specific parameters associated with their 
    #  corresponding polars.Expr.rolling_<function_name> method.
    #  For instance, pl_quantile corresponds to polars.Expr.rolling_quantile.
        
    import pytimetk as tk
    import pandas as pd
    import polars as pl
    import numpy as np
    from pytimetk.utils.polars_helpers import pl_quantile
    from pytimetk.utils.pandas_helpers import pd_quantile
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in std function
                    ('quantile_75', pl_quantile(quantile=0.75)),  # Configurable with all parameters found in polars.Expr.rolling_quantile
                ],
                min_periods = 1,
                engine = 'polars',  # Utilize Polars for the underlying computations
            )
    )
    display(expanded_df)
    ```
        
    ```{python}
    # Example 3 - Lambda Functions for Expanding Window Functions are faster in Pandas than Polars
    # This example demonstrates the use of lambda functions of the form lambda x: x
    # Identity lambda functions, while convenient, have signficantly slower performance.
    # When using lambda functions the Pandas backend will likely be faster than Polars.
    
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    
                    ('range', lambda x: x.max() - x.min()),  # Identity lambda function: can be slower, especially in Polars
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
            )
    )
    display(expanded_df)
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
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
        
    # Set min_periods to 1 if not specified
    min_periods = 1 if min_periods is None else min_periods
    
    # Call the function to augment expanding window columns using the specified engine
    if engine == 'pandas':
        
        # Get threads
        threads = get_threads(threads)  
        
        ret = _augment_expanding_pandas(
            data, 
            date_column, 
            value_column, 
            window_func, 
            min_periods, 
            threads, 
            show_progress,
            **kwargs
        )
        
    elif engine == 'polars':
        ret = _augment_expanding_polars(
            data, 
            date_column, 
            value_column, 
            window_func, 
            min_periods, 
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
pd.core.groupby.generic.DataFrameGroupBy.augment_expanding = augment_expanding


def _augment_expanding_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    threads: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Augments the given dataframe with expanding calculations using the Pandas library.
    """
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the expanding calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
    
    # Apply Series-based expanding window functions
    if threads == 1:
        func = partial(
            _process_expanding_window, 
            value_column=value_column, 
            window_func=window_func, 
            min_periods=min_periods, 
            **kwargs
        )

        # Use tqdm to display progress for the loop
        result_dfs = [func(group) for _, group in conditional_tqdm(grouped, total=len(grouped), desc="Calculating Expanding...", display=show_progress)]
    else:
        # Prepare to use pathos.multiprocessing
        pool = ProcessingPool(threads)

        # Use partial to "freeze" arguments for _process_single_roll
        func = partial(
            _process_expanding_window, 
            value_column=value_column, 
            window_func=window_func, 
            min_periods=min_periods, 
            **kwargs
        )

        result_dfs = list(conditional_tqdm(pool.map(func, (group for _, group in grouped)), 
                                            total=len(grouped), 
                                            desc="Calculating Expanding...", 
                                            display=show_progress))
    
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

def _process_expanding_window(group_df, value_column, window_func, min_periods, **kwargs):
    
    result_dfs = []
    for col in value_column:
        for func in window_func:
            if isinstance(func, tuple):
                # Ensure the tuple is of length 2 and begins with a string
                if len(func) != 2:
                    raise ValueError(f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}.")
                if not isinstance(func[0], str):
                    raise TypeError(f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}.")
            
                user_func_name, func = func
                new_column_name = f"{col}_expanding_{user_func_name}"
                    
                # Try handling a lambda function of the form lambda x: x
                if inspect.isfunction(func) and len(inspect.signature(func).parameters) == 1:
                    try:
                        # Construct expanding window column
                        group_df[new_column_name] = group_df[col].expanding(min_periods=min_periods, **kwargs).apply(func, raw=True)
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
                        # Get the expanding window function 
                        expanding_function = getattr(group_df[col].expanding(min_periods=min_periods, **kwargs), func_name, None)
                    except Exception as e:
                        raise AttributeError(f"The function `{func_name}` tried to access a non-existent attribute or method in Pandas. Error: {e}")

                    if expanding_function:
                        try:
                            # Apply expanding function to data and store in new column
                            group_df[new_column_name] = expanding_function(**default_kwargs)
                        except Exception as e:
                            raise Exception(f"Failed to construct the expanding window column using function `{user_func_name}`. Error: {e}")
                else:
                    raise TypeError(f"Unexpected function format for `{user_func_name}`.")

            elif isinstance(func, str):
                new_column_name = f"{col}_expanding_{func}"
                # Get the expanding function (like mean, sum, etc.) specified by `func` for the given column and window settings
                if func == "quantile":
                    new_column_name = f"{col}_expanding_{func}_50"
                    group_df[new_column_name] = group_df[col].expanding(min_periods=min_periods, **kwargs).quantile(q=0.5)
                    warnings.warn(
                        "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                        "For more control over the quantile value, consider using the function `pd_quantile()`. "
                        "For example: ('quantile_75', pd_quantile(q=0.75))."
                    )
                else:
                    expanding_function = getattr(group_df[col].expanding(min_periods=min_periods, **kwargs), func, None)
                    # Apply expanding function to data and store in new column
                    if expanding_function:
                        group_df[new_column_name] = expanding_function()
                    else:
                        raise ValueError(f"Invalid function name: {func}")
            else:
                raise TypeError(f"Invalid function type: {type(func)}")
                    
        result_dfs.append(group_df)
    
    return pd.concat(result_dfs)

def _augment_expanding_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    **kwargs,
) -> pl.DataFrame:
    """
    Augments the given dataframe with expanding calculations using the Polars library.
    """
    
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
    
    # Initialize lists to store expanding expressions and new column names  
    expanding_exprs = []
    new_column_names = []

    # Construct expanding expressions for each column and function combination
    for col in value_column:
        for func in window_func:
            
            # Handle functions passed as tuples
            if isinstance(func, tuple):
                # Ensure the tuple is of length 2 and begins with a string
                if len(func) != 2:
                    raise ValueError(f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}.")
                if not isinstance(func[0], str):
                    raise TypeError(f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}.")
                
                user_func_name, func = func
                new_column_name = f"{col}_expanding_{user_func_name}"
                
                # Try handling a lambda function of the form lambda x: x
                if inspect.isfunction(func) and len(inspect.signature(func).parameters) == 1:
                    try:
                        # Construct expanding window expression
                        expanding_expr = pl.col(col) \
                            .cast(pl.Float64) \
                            .rolling_map(
                                function=func,
                                window_size=pandas_df.shape[0], 
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
                            'window_size': pandas_df.shape[0],
                            'min_periods': min_periods
                        }
                        # Combine local values with user-provided parameters for the configurable function
                        user_kwargs.update(local_values)
                        # Update the default configurable parameters (without adding new keys)
                        default_kwargs = update_dict(default_kwargs, user_kwargs)
                    except Exception as e:
                        raise ValueError("Error encountered while updating parameters for the configurable function `{func_name}` passed to `window_func`: {e}")
                    
                    try:
                        # Construct expanding window expression
                        expanding_expr = getattr(pl.col(col), f"rolling_{func_name}")(**default_kwargs)
                    except AttributeError as e:
                        raise AttributeError(f"The function `{user_func_name}` tried to access a non-existent attribute or method in Polars. Error: {e}")
                    except Exception as e:
                        raise Exception(f"Error during the execution of `{user_func_name}` in Polars. Error: {e}")
                
                else:
                    raise TypeError(f"Unexpected function format for `{user_func_name}`.")
                
                expanding_expr = expanding_expr.alias(new_column_name)

            elif isinstance(func, str):
                func_name = func
                new_column_name = f"{col}_expanding_{func_name}"
                if not hasattr(pl.col(col), f"{func_name}"):
                    raise ValueError(f"{func_name} is not a recognized function for Polars.")
                
                # Construct expanding window expression and handle specific case of 'skew'
                if func_name == "skew":
                    expanding_expr = getattr(pl.col(col), f"rolling_{func_name}")(window_size=pandas_df.shape[0])
                elif func_name == "quantile":
                        new_column_name = f"{col}_expanding_{func}_50"
                        expanding_expr = getattr(pl.col(col), f"rolling_{func_name}")(quantile=0.5, window_size=pandas_df.shape[0], min_periods=min_periods, interpolation='midpoint')
                        warnings.warn(
                            "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                            "For more control over the quantile value, consider using the function `pl_quantile()`. "
                            "For example: ('quantile_75', pl_quantile(quantile=0.75))."
                        )
                else: 
                    expanding_expr = getattr(pl.col(col), f"rolling_{func_name}")(window_size=pandas_df.shape[0], min_periods=min_periods)

                expanding_expr = expanding_expr.alias(new_column_name)
                
            else:
                raise TypeError(f"Invalid function type: {type(func)}")
            
            # Add constructed expressions and new column names to respective lists
            expanding_exprs.append(expanding_expr)
            new_column_names.append(new_column_name)

    # Select the columns
    selected_columns = expanding_exprs
    
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



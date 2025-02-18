import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union, Optional, Callable, Tuple, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def augment_ewm(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list] = 'mean',
    alpha: float = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Add Exponential Weighted Moving (EWM) window functions to a DataFrame or 
    GroupBy object.
    
    The `augment_ewm` function applies Exponential Weighted Moving (EWM) window 
    functions to specified value columns of a DataFrame and adds the results as 
    new columns.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input DataFrame or GroupBy object.
    date_column : str
        The name of the column containing date information in the input 
        DataFrame or GroupBy object.
    value_column : Union[str, list]
        The `value_column` parameter is used to specify the column(s) on which 
        the Exponential Weighted Moving (EWM) calculations will be performed. It 
        can be either a string or a list of strings, representing the name(s) of 
        the column(s) in the input DataFrame or GroupBy
    window_func : Union[str, list], optional
        The `window_func` parameter is used to specify the Exponential Weighted 
        Moving (EWM) window function(s) to apply. It can be a string or a list 
        of strings. The possible values are:
        
        - 'mean': Calculate the exponentially weighted mean.
        - 'median': Calculate the exponentially weighted median.
        - 'std': Calculate the exponentially weighted standard deviation.
        - 'var': Calculate the exponentially weighted variance.
        
    alpha : float
        The `alpha` parameter is a float that represents the smoothing factor 
        for the Exponential Weighted Moving (EWM) window function. It controls 
        the rate at which the weights decrease exponentially as the data points 
        move further away from the current point.
    **kwargs: 
        Additional arguments that are directly passed to the pandas EWM method. 
        For more details, refer to the "Notes" section below.
    
    Returns
    -------
    pd.DataFrame
        The function `augment_ewm` returns a DataFrame augmented with the 
        results of the Exponential Weighted Moving (EWM) calculations.
    
    Notes
    ------
    Any additional arguments provided through **kwargs are directly passed 
    to the pandas EWM method. These arguments can include parameters like 
    'com', 'span', 'halflife', 'ignore_na', 'adjust' and more.

    For a comprehensive list and detailed description of these parameters:
    
    - Refer to the official pandas documentation: 
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
    
    - Or, within an interactive Python environment, use: 
        `?pandas.DataFrame.ewm` to display the method's docstring.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    from pytimetk import augment_ewm
    import pandas as pd
    import numpy as np

    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    ```

    ```{python}
    # This example demonstrates the use of string-named functions on an EWM.
    # The decay parameter used in this example is 'alpha', but other methods 
    #  (e.g., 'com', 'span', 'halflife') can also be utilized.

    ewm_df = (
        df
            .groupby('id')
            .augment_ewm(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',
                    'std', 
                ],
                alpha = 0.1, 
            )
    )
    display(ewm_df)
    ```
    """
    # Checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
    
    # Convert string value column to list for consistency
    if isinstance(value_column, str):
        value_column = [value_column]
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, str):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the EWM calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
    
    # Helper function to identify and retrieve the active decay parameter (alpha, com, span, halflife) 
    # and its value for inclusion in the column name.
    def determine_decay_parameter(alpha, **kwargs):
        if alpha is not None:
            return 'alpha', alpha
        for param in ['com', 'span', 'halflife']:
            if param in kwargs:
                return param, kwargs[param]
        return None, None
    
    # Retrieve active decay parameter and value
    decay_param, value = determine_decay_parameter(alpha, **kwargs)
    
    # Raise error if no valid decay parameter
    if decay_param is None:
        raise ValueError("No valid decay parameter provided. Specify 'alpha' through function arguments, or one of 'com', 'span', or 'halflife' through **kwargs.")
    
    # Apply Series-based Exponential Weighted Moving window functions
    result_dfs = []
    for _, group_df in grouped:
        for value_col in value_column:
            for func in window_func: 
                if isinstance(func, str):
                    new_column_name = f"{value_col}_ewm_{func}_{decay_param}_{value}"
                    # Get the EWM function (like mean, sum, etc.) specified by `func` for the given column and window settings
                    try:
                        ewm_function = getattr(group_df[value_col].ewm(alpha=alpha, **kwargs), func, None)
                    except Exception as e:
                         # Check for the specific error message from pandas
                        if str(e) == "Must pass one of comass, span, halflife, or alpha":
                            # Raise a new error with more informative message
                            raise ValueError("Must specify the 'alpha' decay parameter through function arguments, or one of 'com', 'span', or 'halflife' through **kwargs.")
                        else:
                            # If it's a different Error, just raise it as is
                            raise
                                
                    # Apply EWM function to data and store in new column
                    if ewm_function:
                        group_df[new_column_name] = ewm_function()
                    else:
                        raise ValueError(f"Invalid function name: {func}")
                else:
                    raise TypeError(f"Invalid function type: {type(func)}")
                    
        result_dfs.append(group_df)
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_ewm = augment_ewm
    
    

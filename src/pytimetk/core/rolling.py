import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union, Optional, Callable, Tuple, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def augment_rolling(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    window: Union[int, tuple, list] = 2,
    min_periods: Optional[int] = None,
    center: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more Series-based rolling functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    value_column : Union[str, list]
        Column(s) to which the rolling window functions should be applied. Can be a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_rolling` function specifies the function(s) to be applied to the rolling windows of the value column(s).

        1. It can be either:
            - A string representing the name of a standard function (e.g., 'mean', 'sum').
            
        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name for the function and the function itself.
            - Each custom function should accept a Pandas Series as its input and operate on that series.
              Example: ("range", lambda x: x.max() - x.min())
        
        (See more Examples below.)

        Note: If your function needs to operate on multiple columns (i.e., it requires access to a DataFrame rather than just a Series), consider using the `augment_rolling_apply` function in this library.   
    window : Union[int, tuple, list], optional, default 2
        Specifies the size of the rolling windows.
        - An integer applies the same window size to all columns in `value_column`.
        - A tuple generates windows from the first to the second value (inclusive).
        - A list of integers designates multiple window sizes for each respective column.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window size. If set, a value will be produced even if fewer observations are present than the window size.
    center : bool, optional, default False
        If `True`, the rolling window will be centered on the current value. For even-sized windows, the window will be left-biased. Otherwise, it uses a trailing window.
    
    Returns
    -------
    pd.DataFrame
        The `augment_rolling` function returns a DataFrame with new columns for each applied function, window size, and value column.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    ```
    
    ```{python}
    # This example demonstrates the use of both string-named functions 
    # and lambda functions on a rolling window. 
    # We specify a list of window sizes: [2,7]. 
    # As a result, the output will have computations for both window sizes 2 and 7.

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
                ]
            )
    )
    display(rolled_df)
    ```
    
    ```{python}
    # Example showcasing the use of string function names and lambda functions 
    # applied on rolling windows.
    # The `window` tuple (1,3) will generate window sizes of 1, 2, and 3.
    
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
                ]
            )
    )
    display(rolled_df) 
    ```
    '''
    # Ensure data is a DataFrame or a GroupBy object
    check_dataframe_or_groupby(data)
    
    # Ensure date column exists and is properly formatted
    check_date_column(data, date_column)
    
    # Ensure value column(s) exist
    check_value_column(data, value_column)
    
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
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the rolling calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
    
    # Apply Series-based rolling window functions
    result_dfs = []
    for _, group_df in grouped:
        for value_col in value_column:
            for window_size in window:
                min_periods = window_size if min_periods is None else min_periods
                for func in window_func:
                    if isinstance(func, tuple):
                        func_name, func = func
                        new_column_name = f"{value_col}_rolling_{func_name}_win_{window_size}"
                        group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs).apply(func, raw=True)
     
                    elif isinstance(func, str):
                        new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                        # Get the rolling function (like mean, sum, etc.) specified by `func` for the given column and window settings
                        rolling_function = getattr(group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs), func, None)
                        # Apply rolling function to data and store in new column
                        if rolling_function:
                            group_df[new_column_name] = rolling_function()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                    else:
                        raise TypeError(f"Invalid function type: {type(func)}")
                    
        result_dfs.append(group_df)
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling


@pf.register_dataframe_method
def augment_rolling_apply(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    window_func: Union[Tuple[str, Callable], List[Tuple[str, Callable]]], 
    window: Union[int, tuple, list] = 2, 
    min_periods: Optional[int] = None,
    center: bool = False,
) -> pd.DataFrame:
    '''Apply one or more DataFrame-based rolling functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    window_func : Union[Tuple[str, Callable], List[Tuple[str, Callable]]]
        The `window_func` parameter in the `augment_rolling_apply` function specifies the function(s) that operate on a rolling window with the consideration of multiple columns.

        The specification can be:
        - A tuple where the first element is a string representing the function's name and the second element is the callable function itself.
        - A list of such tuples for multiple functions.
        
        (See more Examples below.)

        Note: For functions targeting only a single value column without the need for contextual data from other columns, consider using the `augment_rolling` function in this library.
    window : Union[int, tuple, list], optional
        Specifies the size of the rolling windows.
        - An integer applies the same window size to all columns in `value_column`.
        - A tuple generates windows from the first to the second value (inclusive).
        - A list of integers designates multiple window sizes for each respective column.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window size. If set, a value will be produced even if fewer observations are present than the window size.
    center : bool, optional
        If `True`, the rolling window will be centered on the current value. For even-sized windows, the window will be left-biased. Otherwise, it uses a trailing window.
    
    Returns
    -------
    pd.DataFrame
        The `augment_rolling` function returns a DataFrame with new columns for each applied function, window size, and value column.
    
    Examples
    --------
        ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    ```
    
    ```{python}
    # Example showcasing the rolling correlation between two columns (`value1` and `value2`).
    # The correlation requires both columns as input.
    
    # Sample DataFrame with id, date, value1, and value2 columns.
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })
    
    # Compute the rolling correlation for each group of 'id'
    # Using a rolling window of size 3 and a lambda function to calculate the correlation.
    rolled_df = (
        df.groupby('id')
        .augment_rolling_apply(
            date_column='date',
            window=3,
            window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],  # Lambda function for correlation
            center = False  # Not centering the rolling window
        )
    )
    display(rolled_df)
    ```
    
    ```{python}
    # Rolling Regression Example: Using `value1` as the dependent variable and `value2` and `value3` as the independent variables.
    # This example demonstrates how to perform a rolling regression using two independent variables.

    # Required module (scikit-learn) for regression.
    from sklearn.linear_model import LinearRegression

    # Sample DataFrame with `id`, `date`, `value1`, `value2`, and `value3` columns.
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [5, 16, 24, 35, 45, 58],
        'value3': [2, 3, 6, 9, 10, 13]
    })
    
    # Define Regression Function to be applied on the rolling window.
    def regression(df):
    
        model = LinearRegression()
        X = df[['value2', 'value3']]  # Independent variables
        y = df['value1']  # Dependent variable
        model.fit(X, y)
        ret = pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])
        
        return ret # Return intercept and slope as a Series
        
    # Compute the rolling regression for each group of `id`
    # Using a rolling window of size 3 and the regression function.
    rolled_df = (
        df.groupby('id')
        .augment_rolling_apply(
            date_column='date',
            window=3,
            window_func=[('regression', regression)]
        )
        .dropna()
    )

    # Format the results to have each regression output (slope and intercept) in separate columns.
    regression_wide_df = pd.concat(rolled_df['rolling_regression_win_3'].to_list(), axis=1).T
    regression_wide_df = pd.concat([rolled_df.reset_index(drop = True), regression_wide_df], axis=1)
    display(regression_wide_df)
    ```
    '''
    # Ensure data is a DataFrame or a GroupBy object
    check_dataframe_or_groupby(data)
    
    # Ensure date column exists and is properly formatted
    check_date_column(data, date_column)
    
    # Validate window argument and convert it to a consistent list format
    if not isinstance(window, (int, tuple, list)):
        raise TypeError("`window` must be an integer, tuple, or list.")
    if isinstance(window, int):
        window = [window]
    elif isinstance(window, tuple):
        window = list(range(window[0], window[1] + 1))
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the rolling calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
    
    # Helper function to apply rolling calculations on a dataframe
    def rolling_apply(func, df, window_size, min_periods, center):
        num_rows = len(df)
        results = [np.nan] * num_rows
        adjusted_window = window_size // 2 if center else window_size - 1  # determine the offset for centering
        
        for center_point in range(num_rows):
            if center:
                if window_size % 2 == 0:  # left biased window if window size is even
                    start = max(0, center_point - adjusted_window)
                    end = min(num_rows, center_point + adjusted_window)
                else: 
                    start = max(0, center_point - adjusted_window)
                    end = min(num_rows, center_point + adjusted_window + 1)
            else:
                start = max(0, center_point - adjusted_window)
                end = center_point + 1
            
            window_df = df.iloc[start:end]
            
            if len(window_df) >= min_periods:
                results[center_point if center else end - 1] = func(window_df)
        
        return pd.DataFrame({'result': results}, index=df.index)
    
    # Apply DataFrame-based rolling window functions
    result_dfs = []
    for _, group_df in grouped:
        for window_size in window:
            min_periods = window_size if min_periods is None else min_periods
            for func in window_func:
                if isinstance(func, tuple):
                    func_name, func = func
                    new_column_name = f"rolling_{func_name}_win_{window_size}"
                    group_df[new_column_name] = rolling_apply(func, group_df, window_size, min_periods=min_periods, center=center)
                else:
                    raise TypeError(f"Expected 'tuple', but got invalid function type: {type(func)}")     
                    
        result_dfs.append(group_df)
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling_apply = augment_rolling_apply
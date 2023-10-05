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
    window: Union[int, tuple, list] = 2, 
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    window_func_with_iv: Union[str, list, Tuple[str, Callable]] = [],
    min_periods: Optional[int] = None,
    center: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more rolling functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    value_column : Union[str, list]
        Column(s) to which the rolling window functions should be applied. Can be a single column name or a list.
    window : Union[int, tuple, list], optional
        Specifies the size of the rolling windows.
        - An integer applies the same window size to all columns in `value_column`.
        - A tuple generates windows from the first to the second value (inclusive).
        - A list of integers designates multiple window sizes for each respective column.
    window_func : Union[str, list, Tuple[str, Callable]], optional
        The `window_func` parameter in the `augment_rolling` function specifies the function(s) to apply to the rolling windows.

        1. If your function operates solely on a single value column:
            - It can be a string representing the name of a standard function (e.g., 'mean', 'sum').
            - For custom functions, provide a list of tuples, where each tuple consists of a custom name and the function itself. The function should accept a Pandas Series as input. (See Examples below.)

        2. If your function requires independent variables (i.e., it uses more than just the single value column), you should use 'window_func_with_iv' instead. Such functions should expect a DataFrame input, representing the current window of rows. (Refer to the Examples section below.)
    
    window_func_with_iv : Union[str, list, Tuple[str, Callable]], optional
        The `window_func_with_iv` parameter in the `augment_rolling` function specifies function(s) requiring independent variables for the rolling windows.

        1. It can be either:
            - A string representing the name of a predefined function.
            - A list of strings, each specifying a function name.
            - A list of tuples, where each tuple contains a custom function name and the function itself.

        2. Functions specified under `window_func_with_iv` take a Pandas DataFrame, representing the current rolling window of rows, as input. They operate on more than just one value column, utilizing multiple columns or contextual data from the entire window.

        3. If your function processes only a single value column and doesn't need other columns as context, consider using 'window_func' instead. (Refer to the Examples section below.)
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
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    ```
    
    ```{python}
    # This example demonstrates the use of both string-named functions 
    # and lambda functions on a rolling window with no independent variables. 
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
    # applied on rolling windows. In this case, no independent variables are required.
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
    
    ```{python}
    # Example showcasing the rolling correlation between two columns (`value1` and `value2`).
    # The correlation requires both columns as input (i.e., independent variables).

    # Sample DataFrame with id, date, value1, and value2 columns.
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })
    
    # Compute the rolling correlation for each group of 'id'
    # Using a rolling window of size 3 and a lambda function to calculate the correlation.
    rolling_df = (
        df.groupby('id')
        .augment_rolling(
            date_column='date',
            value_column='value1',
            window=3,
            window_func=[],  # No standard window functions specified
            window_func_with_iv=[('corr', lambda x: x['value1'].corr(x['value2']))],  # Lambda function for correlation
            center = False  # Not centering the rolling window
        )
    )
    display(rolling_df)
    
    ```
    
    ```{python}
    # Rolling Regression Example: Using independent variables (`value2` and `value3`)
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
    result_df = (
        df.groupby('id')
        .augment_rolling(
            date_column='date',
            value_column='value1',
            window=3,
            window_func=[],
            window_func_with_iv=[('regression', regression)]
        )
        .dropna()
    )

    # Format the results to have each regression output (slope and intercept) in separate columns.
    regression_wide_df = pd.concat(result_df['rolling_regression_win_3'].to_list(), axis=1).T
    regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)
    display(regression_wide_df)
    
    ```

    ```{python}
    # This example is a showcase of the diverse functionalities available through the `augment_rolling` function.
    # Key Highlights:
    # - Use of built-in Pandas rolling window functions: mean and std.
    # - Incorporation of custom-defined functions for more specific tasks: sample and population standard deviations.
    # - Advanced rolling operations requiring independent variables, represented by correlation and regression.
    # - Handling of multiple value columns, capturing broader data dynamics.

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
    
    # Define a function to calculate the sample standard deviation.
    def sample_std(data):
        n = len(data)
        if n < 2:
            return float('nan')
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1)
        return variance ** 0.5

    # Define a function to calculate the population standard deviation.
    def population_std(data):
        n = len(data)
        if n == 0:
            return float('nan')
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        return variance ** 0.5
    
    # Use the `augment_rolling` function to compute various rolling window metrics:
    # mean, standard deviation, sample std, population std, correlation, and regression.
    rolled_df = (
        df.augment_rolling(
            date_column = 'date', 
            value_column = ['value1', 'value2', 'value3'], 
            window=[2, 4],
            window_func = [
                'mean',
                'std',
                ('sample_std', lambda x: sample_std(x)),
                ('pop_std', lambda x: population_std(x))
            ],
            window_func_with_iv = [
                ('corr', lambda x: x['value1'].corr(x['value2'])),
                ('regression', regression)
                ],
            min_periods=1,
            center=True
        )
)
    rolled_df
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
    
    
    # Helper function to apply rolling calculations that require independent variables
    def rolling_apply(func, df, window_size, min_periods, center):
        results = [np.nan] * len(df)
        adjusted_window = window_size // 2 if center else window_size - 1  # determine the offset for centering
        
        for center_point in range(len(df)):
            if center:
                if window_size % 2 == 0:  # left biased window if window size is even
                    start = max(0, center_point - adjusted_window)
                    end = min(len(df), center_point + adjusted_window)
                else: 
                    start = max(0, center_point - adjusted_window)
                    end = min(len(df), center_point + adjusted_window + 1)
            else:
                start = max(0, center_point - adjusted_window)
                end = center_point + 1
            
            window_df = df.iloc[start:end]
            
            if len(window_df) >= min_periods:
                results[center_point if center else end - 1] = func(window_df)
        
        return pd.DataFrame({'result': results}, index=df.index)
    
    # Apply rolling window functions
    result_dfs = []
    for _, group_df in grouped:

        # Apply the basic window functions
        for value_col in value_column:
            for window_size in window:
                # Set min_periods to window_size if not specified
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
                    
        # Apply the functions that require independent variables
        for window_size in window:
            for func in window_func_with_iv:
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
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling

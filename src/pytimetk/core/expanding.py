import pandas as pd
import polars as pl
import pandas_flavor as pf
import numpy as np

from typing import Union, Optional, Callable, Tuple, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def augment_expanding(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    quantile: Optional[float] = 0.5,
    engine: str = 'pandas',
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more Series-based expanding functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    value_column : Union[str, list]
        Column(s) to which the expanding window functions should be applied. Can be a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_expanding` function specifies the function(s) to be applied to the expanding windows of the value column(s).

        1. It can be either:
            - A string representing the name of a standard function (e.g., 'mean', 'sum').
            
        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name for the function and the function itself.
            - Each custom function should accept a Pandas Series as its input and operate on that series.
              Example: ("range", lambda x: x.max() - x.min())
        
        (See more Examples below.)

        Note: If your function needs to operate on multiple columns (i.e., it requires access to a DataFrame rather than just a Series), consider using the `augment_expanding_apply` function in this library.   
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window size. If set, a value will be produced even if fewer observations are present than the window size.
    engine : str, optional, default 'pandas'
        Specifies the backend computation library for augmenting expanding window functions. 
    
        The options are:
            - "pandas" (default): Uses the `pandas` library.
            - "polars": Uses the `polars` library, which may offer performance benefits for larger datasets.
    
    quantile : float, optional, default 0.5
        Specifies the quantile value to be used when the "quantile" string is passed to the `window_func` parameter. 
        The value should be between 0 and 1, inclusive. For example, 0.5 represents the median.
    **kwargs : additional keyword arguments
        Additional arguments passed to the `pandas.Series.expanding` method when using the Pandas engine.
    
    Returns
    -------
    pd.DataFrame
        The `augment_expanding` function returns a DataFrame with new columns for each applied function, window size, and value column.
    
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
    # (including 'quantile') and lambda functions on an expanding window
    # using the Pandas backend for calculations.

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function
                    'quantile',  # Built-in quantile function
                    ('range', lambda x: x.max() - x.min()),  # Lambda function to compute the range of values within the expanding window
                ],
                min_periods = 1,
                quantile = 0.5,  # Specify the quantile level for the 'quantile' function in window_func 
                engine = 'pandas',  # Utilize pandas for the underlying computations
            )
    )
    display(expanded_df)
    ```
    '''
    
    if engine == 'pandas':
        return _augment_expanding_pandas(data, date_column, value_column, window_func, min_periods, quantile, **kwargs)
    elif engine == 'polars':
        return _augment_expanding_polars(data, date_column, value_column, window_func, min_periods, quantile, **kwargs)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_expanding = augment_expanding


def _augment_expanding_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    quantile: Optional[float] = 0.5,
    **kwargs,
) -> pd.DataFrame:

    # Ensure data is a DataFrame or a GroupBy object
    check_dataframe_or_groupby(data)
    
    # Ensure date column exists and is properly formatted
    check_date_column(data, date_column)
    
    # Ensure value column(s) exist
    check_value_column(data, value_column)
    
    # Convert string value column to list for consistency
    if isinstance(value_column, str):
        value_column = [value_column]
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the expanding calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
        
    # Set min_periods to 1 if not specified
    min_periods = 1 if min_periods is None else min_periods
    
    # Apply Series-based expanding window functions
    result_dfs = []
    for _, group_df in grouped:
        for value_col in value_column:
            for func in window_func:
                if isinstance(func, tuple):
                    func_name, func = func
                    new_column_name = f"{value_col}_expanding_{func_name}"
                    group_df[new_column_name] = group_df[value_col].expanding(min_periods=min_periods, **kwargs).apply(func, raw=True)
    
                elif isinstance(func, str):
                    new_column_name = f"{value_col}_expanding_{func}"
                    # Get the expanding function (like mean, sum, etc.) specified by `func` for the given column and window settings
                    if func == "quantile":
                        group_df[new_column_name] = group_df[value_col].expanding(min_periods=min_periods, **kwargs).quantile(q=quantile)
                    else:
                        expanding_function = getattr(group_df[value_col].expanding(min_periods=min_periods, **kwargs), func, None)
                        # Apply expanding function to data and store in new column
                        if expanding_function:
                            group_df[new_column_name] = expanding_function()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                else:
                    raise TypeError(f"Invalid function type: {type(func)}")
                    
        result_dfs.append(group_df)
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df


@pf.register_dataframe_method
def augment_expanding_apply(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    window_func: Union[Tuple[str, Callable], List[Tuple[str, Callable]]], 
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    '''Apply one or more DataFrame-based expanding functions to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    window_func : Union[Tuple[str, Callable], List[Tuple[str, Callable]]]
        The `window_func` parameter in the `augment_expanding_apply` function specifies the function(s) that operate on a expanding window with the consideration of multiple columns.

        The specification can be:
        - A tuple where the first element is a string representing the function's name and the second element is the callable function itself.
        - A list of such tuples for multiple functions.

        Note: For functions targeting only a single value column without the need for contextual data from other columns, consider using the `augment_expanding` function in this library.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window size. If set, a value will be produced even if fewer observations are present than the window size.
    
    Returns
    -------
    pd.DataFrame
        The `augment_expanding` function returns a DataFrame with new columns for each applied function, window size, and value column.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    ```
    
    ```{python}
    # Example showcasing the expanding correlation between two columns (`value1` and `value2`).
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
        )
    )
    display(expanding_df)
    ```
    
    ```{python}
    # expanding Regression Example: Using `value1` as the dependent variable and `value2` and `value3` as the independent variables.
    # This example demonstrates how to perform a expanding regression using two independent variables.

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
            window_func=[('regression', regression)]
        )
        .dropna()
    )

    # Format the results to have each regression output (slope and intercept) in separate columns.
    regression_wide_df = pd.concat(result_df['expanding_regression'].to_list(), axis=1).T
    regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)
    display(regression_wide_df)
    ```
    '''
    # Ensure data is a DataFrame or a GroupBy object
    check_dataframe_or_groupby(data)
    
    # Ensure date column exists and is properly formatted
    check_date_column(data, date_column)
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # Group data if it's a GroupBy object; otherwise, prepare it for the expanding calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
        
     # Set min_periods to 1 if not specified
    min_periods = 1 if min_periods is None else min_periods
    
    # Helper function to apply expanding calculations on a dataframe
    def expanding_apply(func, df, min_periods):
        num_rows = len(df)
        results = [np.nan] * num_rows
        
        for end_point in range(1, num_rows + 1):
            window_df = df.iloc[: end_point]
            if len(window_df) >= min_periods:
                results[end_point - 1] = func(window_df)

        return pd.DataFrame({'result': results}, index=df.index)
    
    # Apply DataFrame-based expanding window functions
    result_dfs = []
    for _, group_df in grouped:
        for func in window_func:
            if isinstance(func, tuple):
                func_name, func = func
                new_column_name = f"expanding_{func_name}"
                group_df[new_column_name] = expanding_apply(func, group_df, min_periods=min_periods)
            else:
                raise TypeError(f"Expected 'tuple', but got invalid function type: {type(func)}")     
                
        result_dfs.append(group_df)
    
    # Combine processed dataframes and sort by index
    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_expanding_apply = augment_expanding_apply


def _augment_expanding_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list],  
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    quantile: Optional[float] = 0.5,
    **kwargs,
) -> pl.DataFrame:
    
    # Ensure data is a DataFrame or a GroupBy object
    check_dataframe_or_groupby(data)
    
    # Ensure date column exists and is properly formatted
    check_date_column(data, date_column)
    
    # Ensure value column(s) exist
    check_value_column(data, value_column)
    
    # Convert string value column to list for consistency
    if isinstance(value_column, str):
        value_column = [value_column]
    
    # Convert single window function to list for consistent processing    
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    # If a GroubBy Object: Retreive the group by column name, and sort by group and date
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data_copy = data_copy.sort_values(by=[*group_names, date_column])
    else: 
        group_names = None
        data_copy = data_copy.sort_values(by=[date_column])
        
    # Set min_periods to 1 if not specified
    min_periods = 1 if min_periods is None else min_periods

    # Convert various data input types to a Pandas DataFrame
    if isinstance(data_copy, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data_copy.apply(lambda x: x)
    elif isinstance(data_copy, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data_copy
    elif isinstance(data_copy, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data_copy.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    # Helper function to map the pandas aggregating function name to the corresponding Polars rolling function
    def rolling_function(col, func_name, window_size, min_periods, quantile):
        """Maps a Pandas string function name to a Polars rolling function."""
        
        # Ensure the requested function name is a valid method of the column object
        if not hasattr(pl.col(col), f"rolling_{func_name}"):
            raise ValueError(f"{func_name} is not a recognized rolling function for Polars.")
        
        # Construct the rolling function dynamically
        func = getattr(pl.col(col), f"rolling_{func_name}")
        
        # Handle special case for 'quantile' and 'skew'
        if func_name == "quantile":
            return func(quantile=quantile, window_size=window_size, min_periods=min_periods, interpolation='midpoint')
        if func_name == "skew":
            return func(window_size=window_size)
        
        return func(window_size=window_size, min_periods=min_periods)
        
        
    expanding_exprs = []

    # For each column and each function, construct the respective expanding
    for col in value_column:
        for func in window_func:
            # Handle custom functions passed as tuple
            if isinstance(func, tuple):
                func_name, func = func
                new_column_name = f"{col}_expanding_{func_name}"
                # Construct the expanding expression that when executed will create the new column
                expanding_expr = pl.col(col) \
                    .cast(pl.Float64) \
                    .rolling_apply(
                        function=func,
                        window_size=pandas_df.shape[0], 
                        min_periods=min_periods
                    )
                # Add groupby instructions to the expanding expression
                if group_names:
                    expanding_expr = expanding_expr.over(group_names)
                # Add column naming instructions to expression
                expanding_expr = expanding_expr.alias(new_column_name)
           
            # Handle built-in functions passed as string
            elif isinstance(func, str):
                new_column_name = f"{col}_expanding_{func}"
                # Construct the expanding expression (mapped from the Pandas string-function) that when executed will create the new column
                expanding_expr = rolling_function(
                    col=col,
                    func_name=func, 
                    window_size=pandas_df.shape[0], 
                    min_periods=min_periods,
                    quantile=quantile
                )
                if group_names:
                    expanding_expr = expanding_expr.over(group_names)
                # Add column naming instructions to expression
                expanding_expr = expanding_expr.alias(new_column_name)
                
            else:
                raise TypeError(f"Invalid function type: {type(func)}")
            
            # Store the expanding expression in a list.
            expanding_exprs.append(expanding_expr)

    # Convert the given Pandas DataFrame into a Polars DataFrame for processing
    df = pl.DataFrame(pandas_df)
    # Explicitly evaluate the accumulated expanding expressions to create new columns in a Polars DataFrame
    out_df = df.select(expanding_exprs)
    # Merge the original and the newly computed columns horizontally and convert back to a Pandas DataFrame
    df = pl.concat([df, out_df], how="horizontal").to_pandas()

    return df

import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union, Optional, Callable, Tuple, List

@pf.register_dataframe_method
def augment_rolling(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list], 
    use_independent_variables: bool = False,
    window: Union[int, tuple, list] = 2, 
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    center: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more rolling functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input DataFrame or GroupBy object that contains the data to be processed. It can be either a Pandas DataFrame or a GroupBy object.
    date_column : str
        The `date_column` parameter is the name of the datetime column in the DataFrame by which the data should be sorted within each group.
    value_column : Union[str, list]
        The `value_column` parameter is the name of the column(s) in the DataFrame to which the rolling window function(s) should be applied. It can be a single column name or a list of column names.
    use_independent_variables : bool
        The `use_independent_variables` parameter is an optional parameter that specifies whether the rolling function(s) require independent variables, such as rolling correlation or rolling regression. (See Examples below.)
    window : Union[int, tuple, list], optional
        The `window` parameter in the `augment_rolling` function is used to specify the size of the rolling windows. It can be either an integer or a list of integers. 
        
        - If it is an integer, the same window size will be applied to all columns specified in the `value_column`. 
        
        - If it is a tuple, it will generate windows from the first to the second value (inclusive).
        
        - If it is a list of integers, each integer in the list will be used as the window size for the corresponding column in the `value_column` list.
    window_func : Union[str, list, Tuple[str, Callable]], optional
        The `window_func` parameter in the `augment_rolling` function is used to specify the function(s) to be applied to the rolling windows. 
        
        1. It can be a string or a list of strings, where each string represents the name of the function to be applied. 
        
        2. Alternatively, it can be a list of tuples, where each tuple contains the name of the function to be applied and the function itself. The function is applied as a Pandas Series. (See Examples below.)
        
        3. If the function requires independent variables, the `use_independent_variables` parameter must be specified. The independent variables will be passed to the function as a DataFrame containing the window of rows. (See Examples below.)
        
    center : bool, optional
        The `center` parameter in the `augment_rolling` function determines whether the rolling window is centered or not. If `center` is set to `True`, the rolling window will be centered, meaning that the value at the center of the window will be used as the result. If `
    
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
    # String Function Name and Series Lambda Function (no independent variables)
    # window = [2,7] yields only 2 and 7
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = [2,7], 
                window_func = ['mean', ('std', lambda x: x.std())]
            )
    )
    rolled_df
    ```
    
    ```{python}
    # String Function Name and Series Lambda Function (no independent variables)
    # window = (1,3) yields 1, 2, and 3
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = (1,3), 
                window_func = ['mean', ('std', lambda x: x.std())]
            )
    )
    rolled_df 
    ```
    
    ```{python}
    # Rolling Correlation: Uses independent variables (value2)
    
    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })
    
    result_df = (
        df.groupby('id')
        .augment_rolling(
            date_column='date',
            value_column='value1',
            use_independent_variables=True,
            window=3,
            window_func=[('corr', lambda df: df['value1'].corr(df['value2']))],
            center = False
        )
    )
    result_df
    
    ```
    
    ```{python}
    # Rolling Regression: Using independent variables (value2 and value3)
    
    # Requires: scikit-learn
    from sklearn.linear_model import LinearRegression

    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [5, 16, 24, 35, 45, 58],
        'value3': [2, 3, 6, 9, 10, 13]
    })
    
    # Define Regression Function
    def regression(df):
    
        model = LinearRegression()
        X = df[['value2', 'value3']]  # Extract X values (independent variables)
        y = df['value1']  # Extract y values (dependent variable)
        model.fit(X, y)
        ret = pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])
        
        return ret # Return intercept and slope as a Series
        

    # Example to call the function
    result_df = (
        df.groupby('id')
        .augment_rolling(
            date_column='date',
            value_column='value1',
            use_independent_variables=True,
            window=3,
            window_func=[('regression', regression)]
        )
        .dropna()
    )

    # Display Results in Wide Format since returning multiple values
    regression_wide_df = pd.concat(result_df['value1_rolling_regression_win_3'].to_list(), axis=1).T
    
    regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)
    
    regression_wide_df
    ```
    '''
    
    
    def rolling_apply_2(func, df, window_size, min_periods, center):
        results = [np.nan] * len(df)
        adjusted_window = window_size // 2 if center else window_size - 1  # determine the offset for centering
        
        for center_point in range(len(df)):
            if center:
                start = max(0, center_point - adjusted_window)
                end = min(len(df), center_point + adjusted_window + 1)
            else:
                start = max(0, center_point - adjusted_window)
                end = min(len(df), start + window_size)
            
            window_df = df.iloc[start:end]
            
            if len(window_df) >= min_periods:
                results[center_point if center else end - 1] = func(window_df)
        
        return pd.DataFrame(results, columns=['result'], index=df.index)

    
    
    # def rolling_apply(func, series, *args):        
    #     result = series.rolling(window=window_size, center=center, **kwargs).apply(lambda x: func(x, *args), raw=False)
    #     return result
    
    # def rolling_apply_2(func, df):
        
    #     results = []
    #     for start in range(len(df) - window_size + 1):
    #         window_df = df.iloc[start:start + window_size]
    #         result = func(window_df)
    #         results.append(result)
        
    #     ret = pd.DataFrame(results, index=df.index[window_size - 1:])

    #     return ret
    
    
    if not isinstance(data, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
        raise TypeError("`data` must be a Pandas DataFrame or GroupBy object.")
        
    if isinstance(value_column, str):
        value_column = [value_column]
    
    if not isinstance(window, (int, tuple, list)):
        raise TypeError("`window` must be an integer, tuple, or list.")
    
    if isinstance(window, int):
        window = [window]
    elif isinstance(window, tuple):
        window = list(range(window[0], window[1] + 1))
        
    if isinstance(window_func, (str, tuple)):
        window_func = [window_func]
    
    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(group_names)
    else: 
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]
    
    result_dfs = []
    for _, group_df in grouped:
        for value_col in value_column:
            for window_size in window:
                
                if min_periods is None:
                    min_periods = window_size
                
                for func in window_func:
                    if isinstance(func, tuple):
                        func_name, func = func
                        new_column_name = f"{value_col}_rolling_{func_name}_win_{window_size}"
                        
                        if use_independent_variables:                           
                            group_df[new_column_name] = rolling_apply_2(func, group_df, window_size, min_periods=min_periods, center=center)
                        else:
                            # group_df[new_column_name] = rolling_apply(func, group_df[value_col])
                            group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=min_periods,center=center, **kwargs).apply(func, raw=True)
                    
                            
                    elif isinstance(func, str):
                        new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                        
                        rolling_method = getattr(group_df[value_col].rolling(window=window_size, min_periods=min_periods, center=center, **kwargs), func, None)
                        
                        if rolling_method:
                            group_df[new_column_name] = rolling_method()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                    
                    else:
                        raise TypeError(f"Invalid function type: {type(func)}")
                    
        result_dfs.append(group_df)

    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling

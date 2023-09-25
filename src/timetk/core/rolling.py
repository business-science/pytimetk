import pandas as pd
import pandas_flavor as pf

    
@pf.register_dataframe_method
def augment_rolling(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy, 
    date_column: str, 
    value_column: str or list, 
    window: int or tuple or list, 
    window_func: str or list = 'mean',
    center: bool = False,
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more rolling functions and window sizes to one or more columns of a DataFrame.
    
    The `augment_rolling` function applies multiple rolling window functions with varying window sizes to specified columns of a DataFrame, considering grouping columns and a datetime column for sorting within each group.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The input DataFrame or GroupBy object.
    date_column : str
        The `date_column` parameter is the name of the datetime column in the DataFrame by which the data should be sorted within each group.
    value_column : str or list
        The `value_column` parameter is the name of the column(s) in the DataFrame to which the rolling window function(s) should be applied. It can be a single column name or a list of column names.
    window : int or tuple or list
        The `window` parameter in the `augment_rolling` function is used to specify the size of the rolling windows. It can be either an integer or a list of integers. 
        
        - If it is an integer, the same window size will be applied to all columns specified in the `value_column`. 
        
        - If it is a tuple, it will generate windows from the first to the second value (inclusive).
        
        - If it is a list of integers, each integer in the list will be used as the window size for the corresponding column in the `value_column` list.
    window_func : str or list, optional
        The `window_func` parameter in the `augment_rolling` function is used to specify the function(s) to be applied to the rolling windows. It can be a string or a list of strings, where each string represents the name of the function to be applied. Alternatively, it can be a list of tuples, where each tuple contains the name of the function to be applied and the function itself. 
        
        - If it is a string or a list of strings, the same function will be applied to all columns specified in the `value_column`. 
        
        - If it is a list of tuples, each tuple in the list will be used as the function to be applied to the corresponding column in the `value_column` list.
    center : bool, optional
        The `center` parameter in the `augment_rolling` function determines whether the rolling window is centered or not. If `center` is set to `True`, the rolling window will be centered, meaning that the alue at the center of the window will be used as the result. If `False`, the rolling window will not be centered, meaning that the value at the end of the window will be used as the result. The default value is `False`.
    **kwargs : optional
        Additional keyword arguments to be passed to the `pandas.DataFrame.rolling` function.
    
    Returns
    -------
    pd.DataFrame
        The function `augment_rolling` returns a DataFrame with new columns for each applied function, window size, and value column.
    
    Examples
    --------
    ```{python}
    import timetk as tk
    import pandas as pd
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    df
    ```
    
    ```{python}
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
    '''
    
    
    # Check if data is a Pandas DataFrame or GroupBy object
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
        
    if isinstance(value_column, str):
        value_column = [value_column]
        
    if isinstance(window, int):
        window = [window]
    elif isinstance(window, tuple):
        window = list(range(window[0], window[1] + 1))
    elif not isinstance(window, list):
        raise TypeError(f"Invalid window type: {type(window)}. Please use int, tuple, or list.")
        
    if isinstance(window_func, str):
        window_func = [window_func]
        
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        group_names = None
        group_df = data.sort_values(by=[date_column])
        grouped = [([], data)]
    else: 
        group_names = data.grouper.names
        grouped = data.obj.copy()
        grouped = grouped.sort_values(by=[*group_names, date_column]).groupby(group_names)           
    
    result_dfs = []
    
    for _, group_df in grouped:
        for value_col in value_column:
            for window_size in window:
                for func in window_func:
                    
                    if isinstance(func, tuple):
                        
                        func_name, func = func
                        
                        new_column_name = f"{value_col}_rolling_{func_name}_win_{window_size}"
                        
                        group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=1,center=center, **kwargs).apply(func, raw=True)
                    
                    elif isinstance(func, str):
                        
                        new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                        
                        rolling_method = getattr(group_df[value_col].rolling(window=window_size, min_periods=1,center=center, **kwargs), func, None)
                        
                        if rolling_method:
                            group_df[new_column_name] = rolling_method()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                    
                    else:
                        raise TypeError(f"Invalid function type: {type(func)}")
        
        result_dfs.append(group_df)

    if group_names is None:
        result_df = pd.concat(result_dfs).sort_values(by=[date_column])
    else:
        result_df = pd.concat(result_dfs).sort_values(by=[*group_names, date_column])
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling

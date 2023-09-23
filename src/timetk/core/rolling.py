import pandas as pd
import pandas_flavor as pf

@pf.register_dataframe_method
def augment_rolling(
    data: pd.DataFrame, 
    date_column: str, 
    value_column: str or list, 
    window: int or list, 
    apply_func: str or list = 'mean',
    center: bool = False,
    **kwargs_rolling,
) -> pd.DataFrame:
    """
    Apply multiple rolling window apply_func with varying window sizes to specified columns of a DataFrame,
    considering grouping columns and a datetime column for sorting within each group.
    
    :param df: Input DataFrame.
    :param group_names: List of column names to group by.
    :param date_column: The name of the datetime column by which the data should be sorted within each group.
    :param value_columns: List of column names to which the apply_func should be applied.
    :param window: List of the sizes of the rolling windows.
    :param apply_func: List of apply_func or tuples where, if it's a tuple, the first element is the name of the new column,
                      and the second element is the function to be applied to the rolling windows.
    :return: DataFrame with new columns for each applied function, window size, and value column.
    
    ```{python}
    import timetk as tk
    import pandas as pd
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    df
    
    df.groupby('id').augment_rolling(date_column = 'date', value_column = 'value', window = [2,7], apply_func = ['mean', ('std', lambda x: x.std())])
    
    ```
    """
    
    
    
     # Check if data is a Pandas DataFrame or GroupBy object
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
        
    if isinstance(value_column, str):
        value_column = [value_column]
        
    if isinstance(window, int):
        window = [window]
        
    if isinstance(apply_func, str):
        apply_func = [apply_func]
    
    if isinstance(data, pd.DataFrame):
        
        df = data.copy()
        
        df.sort_values(by=[date_column], inplace=True)
        
        # TODO
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj
        
        df = data.copy()
        result_dfs = []
        
        for _, group_df in df.groupby(group_names):
            
            df = data.copy()
            group_df = df.sort_values(by=[*group_names, date_column])
            
            for value_col in value_column:
                for window_size in window:
                    for func in apply_func:
                        
                        if isinstance(func, tuple):
                            
                            func_name, func = func
                            
                            new_column_name = f"{value_col}_rolling_{func_name}_win_{window_size}"
                            
                            group_df[new_column_name] = group_df[value_col].rolling(window=window_size, min_periods=1,center=center, **kwargs_rolling).apply(func, raw=True)
                        
                        elif isinstance(func, str):
                            
                            new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                            
                            rolling_method = getattr(group_df[value_col].rolling(window=window_size, min_periods=1,center=center, **kwargs_rolling), func, None)
                            
                            if rolling_method:
                                group_df[new_column_name] = rolling_method()
                            else:
                                raise ValueError(f"Invalid function name: {func}")
                        
                        else:
                            raise TypeError(f"Invalid function type: {type(func)}")
            
            result_dfs.append(group_df)
    
        result_df = pd.concat(result_dfs).sort_values(by=[*group_names, date_column])
    
    return result_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_rolling = augment_rolling

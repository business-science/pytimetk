import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union, Optional, Callable, Tuple
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def augment_expanding(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str, 
    value_column: Union[str, list], 
    use_independent_variables: bool = False,
    window_func: Union[str, list, Tuple[str, Callable]] = 'mean',
    min_periods: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    '''Apply one or more expanding functions and window sizes to one or more columns of a DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input DataFrame or GroupBy object that contains the data to be processed. It can be either a Pandas DataFrame or a GroupBy object.
    date_column : str
        The `date_column` parameter is the name of the datetime column in the DataFrame by which the data should be sorted within each group.
    value_column : Union[str, list]
        The `value_column` parameter is the name of the column(s) in the DataFrame to which the expanding window function(s) should be applied. It can be a single column name or a list of column names.
    use_independent_variables : bool
        The `use_independent_variables` parameter is an optional parameter that specifies whether the expanding function(s) require independent variables, such as expanding correlation or expanding regression. (See Examples below.)
    window_func : Union[str, list, Tuple[str, Callable]], optional
        The `window_func` parameter in the `augment_expanding` function is used to specify the function(s) to be applied to the expanding windows. 
        
        1. It can be a string or a list of strings, where each string represents the name of the function to be applied. 
        
        2. Alternatively, it can be a list of tuples, where each tuple contains the name of the function to be applied and the function itself. The function is applied as a Pandas Series. (See Examples below.)
        
        3. If the function requires independent variables, the `use_independent_variables` parameter must be specified. The independent variables will be passed to the function as a DataFrame containing the window of rows. (See Examples below.)
        
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
    # String Function Name and Series Lambda Function (no independent variables)
    rolled_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value',  
                window_func = ['mean', ('std', lambda x: x.std())]
            )
    )
    rolled_df
    ```

    ```{python}
    # Expanding Correlation: Uses independent variables (value2)

    df = pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })

    result_df = (
        df.groupby('id')
        .augment_expanding(
            date_column='date',
            value_column='value1',
            use_independent_variables=True,
            window_func=[('corr', lambda df: df['value1'].corr(df['value2']))],
        )
    )
    result_df

    ```

    ```{python}
    # Expanding Regression: Using independent variables (value2 and value3)

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
        .augment_expanding(
            date_column='date',
            value_column='value1',
            use_independent_variables=True,
            window_func=[('regression', regression)]
        )
        .dropna()
    )
    result_df

    # Display Results in Wide Format since returning multiple values
    regression_wide_df = pd.concat(result_df['value1_expanding_regression'].to_list(), axis=1).T

    regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)

    regression_wide_df
```
    '''
    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
    
    # Expanding Apply Function for Functions that Require Independent Variables
    def expanding_apply(func, df, min_periods):
        n_rows = len(df)
        results = [np.nan] * n_rows
        
        for end_point in range(1, n_rows + 1):
            window_df = df.iloc[: end_point]
            if len(window_df) >= min_periods:
                results[end_point - 1] = func(window_df)

        return pd.DataFrame({'result': results}, index=df.index)

        
    if isinstance(value_column, str):
        value_column = [value_column]
        
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
            if min_periods is None:
                min_periods = 1
            
            for func in window_func:
                if isinstance(func, tuple):
                    func_name, func = func
                    new_column_name = f"{value_col}_expanding_{func_name}"
                    
                    if use_independent_variables:                           
                        group_df[new_column_name] = expanding_apply(func, group_df, min_periods=min_periods)
                    else:
                        try:   
                            group_df[new_column_name] = group_df[value_col].expanding(min_periods=min_periods, **kwargs).apply(func, raw=True)
                        except Exception as e:
                                try: # try independent variables incase user mistakenly did not set to True
                                    group_df[new_column_name] = expanding_apply(func, group_df, min_periods=min_periods)
                                except:
                                    raise e
                                
                elif isinstance(func, str):
                    new_column_name = f"{value_col}_expanding_{func}"
                    
                    expanding_method = getattr(group_df[value_col].expanding(min_periods=min_periods, **kwargs), func, None)
                    
                    if expanding_method:
                        group_df[new_column_name] = expanding_method()
                    else:
                        raise ValueError(f"Invalid function name: {func}")     
                else:
                    raise TypeError(f"Invalid function type: {type(func)}")
                
        result_dfs.append(group_df)

    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    
    return result_df


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_expanding = augment_expanding

import pandas as pd
import pandas_flavor as pf
from timetk.utils.pandas_helpers import flatten_multiindex_column_names

from itertools import cycle

@pf.register_dataframe_method
def summarize_by_time(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str,
    value_column: str or list,
    rule: str = "D",
    agg_func: list = 'sum',
    kind: str = "timestamp",
    wide_format: bool = False,
    fillna: int = 0,
    flatten_column_names: bool = True,
    reset_index: bool = True,
    *args,
    **kwargs
) -> pd.DataFrame:
    '''
    Summarize a DataFrame or GroupBy object by time.
    
    The `summarize_by_time` function aggregates data by a specified time period and one or more numeric columns, allowing for grouping and customization of the time-based aggregation.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        A pandas DataFrame or a pandas GroupBy object. This is the data that you want to summarize by time.
    date_column : str
        The name of the column in the data frame that contains the dates or timestamps to be aggregated by. This column must be of type datetime64.
    value_column : str or list
        The `value_column` parameter is the name of one or more columns in the DataFrame that you want to aggregate by. It can be either a string representing a single column name, or a list of strings representing multiple column names.
    rule : str, optional
        The `rule` parameter specifies the frequency at which the data should be aggregated. It accepts a string representing a pandas frequency offset, such as "D" for daily or "MS" for month start. The default value is "D", which means the data will be aggregated on a daily basis.
    agg_func : list, optional
        The `agg_func` parameter is used to specify one or more aggregating functions to apply to the value column(s) during the summarization process. It can be a single function or a list of functions. The default value is `"sum"`, which represents the sum function. 
    kind : str, optional
        The `kind` parameter specifies whether the time series data is represented as a "timestamp" or a "period". If `kind` is set to "timestamp", the data is treated as a continuous time series with specific timestamps. If `kind` is set to "period", the data is treated as a discrete time series with specific periods. The default value is "timestamp".
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in "wide" or "long" format. If set to `True`, the output will be in wide format, where each group is represented by a separate column. If set to False, the output will be in long format, where each group is represented by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data with. By default, it is set to 0. If you want to keep missing values as NaN, you can use `np.nan` as the value for `fillna`.
    flatten_column_names : bool, optional
        A boolean parameter that determines whether or not to flatten the multiindex column names. If set to `True`, the multiindex column names will be flattened. If set to `False`, the multiindex column names will be preserved. The default value is `True`.
    reset_index : bool, optional
        A boolean parameter that determines whether or not to reset the index of the resulting DataFrame. If set to True, the index will be reset to the default integer index. If set to False, the index will not be reset. The default value is True.
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame that is summarized by time.
        
    Examples
    --------
    ```{python}
    import timetk as tk
    import pandas as pd
    
    df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
    
    df
    ```
    
    ```{python}
    # Summarize by time with a DataFrame object
    ( 
        df 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price',
                rule         = "MS",
                agg_func     = ['mean', 'sum']
            )
    )
    ```
    
    ```{python}
    # Summarize by time with a GroupBy object (Long Format)
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                rule         = 'MS',
                agg_func     = 'sum',
                wide_format  = False, 
            )
    )
    ```
    
    ```{python}
    # Summarize by time with a GroupBy object (Wide Format)
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                rule         = 'MS',
                agg_func     = 'sum',
                wide_format  = True, 
            )
    )
    ```
    
    ```{python}
    # Summarize by time with a GroupBy object and multiple summaries (Wide Format)
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                rule         = 'MS',
                agg_func     = ['sum', 'mean', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75))],
                wide_format  = True, 
            )
    )
    ```
    '''
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    # Convert value_column to a list if it is not already
    if not isinstance(value_column, list):
        value_column = [value_column]
    
    # Set the index of data to the date_column
    if isinstance(data, pd.DataFrame):
        data = data.set_index(date_column)
    
    group_names = None
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.set_index(date_column).groupby(group_names)
    
    # Group data by the groups columns if groups is not None
    # if groups is not None:
    #     data = data.groupby(groups)
    
    # Resample data based on the specified rule and kind
    data = data.resample(rule=rule, kind=kind)
    
    # Create a dictionary mapping each value column to the aggregating function(s)
    agg_dict = {col: agg_func for col in value_column}
    
    # **** FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 1) ****
    
    unique_first_elements = [func[0] for value in agg_dict.values() for func in value if isinstance(func, tuple)]
    
    # print(unique_first_elements)

    if not unique_first_elements == []:
        for key, value in agg_dict.items():
            agg_dict[key] = [func[1] if isinstance(func, tuple) else func for func in value]
            
    # **** END FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 1) ****

    
    # Apply the aggregation using the dict method of the resampled data
    data = data.agg(func=agg_dict, *args, **kwargs)    
    
    
    # Unstack the grouped columns if wide_format is True and groups is not None
    if wide_format:
        if group_names is not None:
            data = data.unstack(group_names) 
        if kind == 'period':
            data.index = data.index.to_period()
    
    # Fill missing values with the specified fillna value
    data = data.fillna(fillna)
    
    # Flatten the multiindex column names if flatten_column_names is True
    if flatten_column_names:
        data = flatten_multiindex_column_names(data)
    
    # Reset the index of data   
    if reset_index: 
        data.reset_index(inplace=True)
        
    # **** FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 2)
    if not unique_first_elements == []:        
        
        columns = data.columns
        # print(columns)
        
        names_iter = cycle(unique_first_elements)
        
        new_columns = [col.replace('<lambda>', next(names_iter)) if '<lambda>' in col else col for col in columns]
        
        data.columns = new_columns
    # **** END FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 2)
    
    return data

# Monkey patch the summarize_by_time method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time = summarize_by_time


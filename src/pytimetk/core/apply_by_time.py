import pandas as pd
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column
from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names
from pytimetk.utils.memory_helpers import reduce_memory_usage


@pf.register_dataframe_method
def apply_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    freq: str = "D",
    wide_format: bool = False,
    fillna: int = 0,
    reduce_memory: bool = False,
    **named_funcs
) -> pd.DataFrame:
    '''
    Apply for time series.
        
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter can be either a pandas DataFrame or a pandas 
        DataFrameGroupBy object. It represents the data on which the apply operation 
        will be performed.
    date_column : str
        The name of the column in the DataFrame that contains the dates.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be 
        resampled. It accepts a string representing a time frequency, such as "D" 
        for daily, "W" for weekly, "M" for monthly, etc. The default value is "D", 
        which means the data will be resampled on a daily basis. Some common 
        frequency aliases include:
            
        - S: secondly frequency
        - min: minute frequency
        - H: hourly frequency
        - D: daily frequency
        - W: weekly frequency
        - M: month end frequency
        - MS: month start frequency
        - Q: quarter end frequency
        - QS: quarter start frequency
        - Y: year end frequency
        - YS: year start frequency
            
    wide_format : bool, optional
        The `wide_format` parameter is a boolean flag that determines whether the 
        output should be in wide format or not. If `wide_format` is set to `True`, 
        the output will have a multi-index column structure, where the first level 
        represents the original columns and the second level represents the group 
        names.
    fillna : int, optional
        The `fillna` parameter is used to specify the value that will be used to 
        fill missing values in the resulting DataFrame. By default, it is set to 0.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    **named_funcs
        The `**named_funcs` parameter is used to specify one or more custom 
        aggregation functions to apply to the data. It accepts named functions 
        in the format:
            
        ``` python
            name = lambda df: df['column1'].corr(df['column2']])
        ```
            
        Where `name` is the name of the function and `df` is the DataFrame that will 
        be passed to the function. The function must return a single value.
            
            
        
    Returns
    -------
    pd.DataFrame
        The function `apply_by_time` returns a pandas DataFrame object.
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
        
    df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
        
    df.glimpse()
    ```
        
    ```{python}    
    # Apply by time with a DataFrame object
    # Allows access to multiple columns at once
    ( 
        df[['order_date', 'price', 'quantity']] 
            .apply_by_time(
                
                # Named apply functions
                price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),
                
                # Parameters
                date_column  = 'order_date', 
                freq         = "MS",
                
            )
    )
    ```
    
    ```{python}    
    # Apply by time with a GroupBy object
    ( 
        df[['category_1', 'order_date', 'price', 'quantity']] 
            .groupby('category_1')
            .apply_by_time(
                
                # Named functions
                price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),
                
                # Parameters
                date_column  = 'order_date', 
                freq         = "MS",
                
            )
    )
    ```
    
    ```{python}    
    # Return complex objects
    ( 
        df[['order_date', 'price', 'quantity']] 
            .apply_by_time(
                
                # Named apply functions
                complex_object = lambda df: [df],
                
                # Parameters
                date_column  = 'order_date', 
                freq         = "MS",
                
            )
    )
    ```
    '''
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)

    # Start by setting the index of data to the date_column
    if isinstance(data, pd.DataFrame):
        data = data.set_index(date_column)
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        if date_column not in group_names:
            data = data.obj.set_index(date_column).groupby(group_names)

    # Resample data based on the specified freq and kind
    grouped = data.resample(rule=freq, kind="timestamp")

    # Apply custom aggregation functions using apply
    def custom_agg(group):
        agg_values = {}
                
        # Apply column-specific functions from **named_funcs
        for name, func in named_funcs.items():
            agg_values[name] = func(group)

        return pd.Series(agg_values)

    data = grouped.apply(custom_agg)

    # Unstack the grouped columns if wide_format is True and group_names is not None
    if wide_format and group_names is not None:
        data = data.unstack(group_names)

    # Fill missing values with the specified fillna value
    data = data.fillna(fillna)

    # Flatten the multiindex column names if needed
    data = flatten_multiindex_column_names(data)

    # Reset the index of data   
    data.reset_index(inplace=True)

    if reduce_memory:
        data = reduce_memory_usage(data)
    
    return data


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.apply_by_time = apply_by_time

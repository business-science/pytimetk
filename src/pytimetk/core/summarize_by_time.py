import pandas as pd
import pandas_flavor as pf
import polars as pl

from typing import Union, Optional, Callable, Tuple, List
import re 
from itertools import cycle

from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from pytimetk.utils.polars_helpers import pandas_to_polars_frequency, pandas_to_polars_aggregation_mapping


# FUNCTIONS -------------------------------------------------------------------

@pf.register_dataframe_method
def summarize_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]],
    freq: str = "D",  
    agg_func: Union[str, list, Tuple[str, Callable]] = 'sum',
    wide_format: bool = False,  
    fillna: int = 0,
    engine: str = 'pandas',  
):
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
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be aggregated. It accepts a string representing a pandas frequency offset, such as "D" for daily or "MS" for month start. The default value is "D", which means the data will be aggregated on a daily basis. Some common frequency aliases include:
        
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
        
    agg_func : list, optional
        The `agg_func` parameter is used to specify one or more aggregating functions to apply to the value column(s) during the summarization process. It can be a single function or a list of functions. The default value is `"sum"`, which represents the sum function. Some common aggregating functions include:
        
        - "sum": Sum of values
        - "mean": Mean of values
        - "median": Median of values
        - "min": Minimum of values
        - "max": Maximum of values
        - "std": Standard deviation of values
        - "var": Variance of values
        - "first": First value in group
        - "last": Last value in group
        - "count": Count of values
        - "nunique": Number of unique values
        - "corr": Correlation between values
        
        Pandas Engine Only:
        Custom `lambda` aggregating functions can be used too. Here are several common examples:
        
        - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
        - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
        - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
        - ("range", lambda x: x.max() - x.min()): Range of values
        
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in "wide" or "long" format. If set to `True`, the output will be in wide format, where each group is represented by a separate column. If set to False, the output will be in long format, where each group is represented by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data with. By default, it is set to 0. If you want to keep missing values as NaN, you can use `np.nan` as the value for `fillna`.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for summarizing the data. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library for summarizing the data. This can be faster than using "pandas" for large datasets. 
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame that is summarized by time.
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
    
    df
    ```
    
    ```{python}
    # Example 1 - Summarize by time with a DataFrame object, pandas engine
    ( 
        df 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price',
                freq         = "MS",
                agg_func     = ['mean', 'sum'],
                engine       = 'pandas'
            )
    )
    ```
    
    ```{python}
    # Example 2 - Summarize by time with a GroupBy object (Long Format), polars engine
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                freq         = 'MS',
                agg_func     = 'sum',
                wide_format  = False, 
                engine       = 'polars'
            )
    )
    ```
    
    ```{python}
    # Example 3 - Summarize by time with a GroupBy object (Wide Format)
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                freq         = 'MS',
                agg_func     = 'sum',
                wide_format  = True,
                engine       = 'pandas' 
            )
    )
    ```
    
    ```{python}
    # Example 4 - Summarize by time with a GroupBy object and multiple value columns and summaries (Wide Format)
    # Note - This example only works with the pandas engine
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = ['total_price', 'quantity'], 
                freq         = 'MS',
                agg_func     = [
                    'sum', 
                    'mean', 
                    ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75))
                ],
                wide_format  = True,
                engine       = 'pandas' 
            )
    )
    ```
    '''
    if engine == 'pandas':
        return _summarize_by_time_pandas(data, date_column, value_column, freq, agg_func, wide_format, fillna)
    elif engine == 'polars':
        return _summarize_by_time_polars(data, date_column, value_column, freq, agg_func, wide_format, fillna)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time = summarize_by_time



def _summarize_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, list],
    freq: str = "D",
    agg_func: Union[str, list, Tuple[str, Callable]] = 'sum',
    wide_format: bool = False,
    fillna: int = 0,
    *args,
    **kwargs
) -> pd.DataFrame:
    
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)

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
    
    # Resample data based on the specified freq
    data = data.resample(rule=freq, kind='timestamp')
    
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
    if wide_format and group_names is not None:
        data = data.unstack(group_names)
    
    # Fill missing values with the specified fillna value
    data = data.fillna(fillna)
    
    # Flatten the multiindex column names if flatten_column_names is True
    data = flatten_multiindex_column_names(data)
    
    # Reset the index of data   
    data.reset_index(inplace=True)

    # Move date column to front of pandas dataframe
    data = data[[date_column] + [col for col in data.columns if col != date_column] ]
        
    # **** FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 2)
    if not unique_first_elements == []:        
        
        columns = data.columns
        # print(columns)
        
        names_iter = cycle(unique_first_elements)
        
        # new_columns = [col.replace('<lambda>', next(names_iter)) if '<lambda>' in col else col for col in columns]
        new_columns = [re.sub(pattern=r"<lambda.*?>",repl=next(names_iter), string=col) if '<lambda' in col else col for col in columns]
        
        data.columns = new_columns
    # **** END FIX BUG WITH GROUPBY RESAMPLED OBJECTS (PART 2)
    
    return data


def _summarize_by_time_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]],  
    freq: str = "D",  
    agg_func: Union[str, list, Tuple[str, Callable]] = 'sum',
    wide_format: bool = False,  
    fillna: int = 0,
):
    

    # Translate the pandas frequency offset to Polars
    polars_freq = pandas_to_polars_frequency(freq, default="d")  # Default to daily if not found

    # Define a dictionary mapping aggregation function names to Polars aggregation expressions
    aggregation_mapping = pandas_to_polars_aggregation_mapping(value_column)

    # If value_column is a string, convert it to a list
    if isinstance(value_column, str):
        value_column = [value_column]
    
    # If agg_func is a string, convert it to a list
    if not isinstance(agg_func, list):
        agg_func = [agg_func]
    
    # Check if agg_func contains any unsupported functions 
    for func in agg_func:
        if isinstance(func, tuple):
            raise TypeError(f"Polars does not currently support custom lambda functions or functions provided as tuples. Here are a list of supported functions: {list(aggregation_mapping.keys())}")

    # Select columns for aggregation based on agg_func
    agg_columns = [aggregation_mapping[func] for func in agg_func if func in aggregation_mapping]

    # Check if the input data is a DataFrame or a GroupBy object
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Extract name from groupby object
        groups = data.grouper.names

        # Convert the GroupBy object into a Polars DataFrame
        df_pl = (
            pl.from_pandas(data.apply(lambda x: x))
                 .groupby(groups, maintain_order=True)
                 .agg(pl.all().sort_by(date_column))
        )

        # Create a list of column names to explode
        columns_to_explode = [col for col in df_pl.columns if col != groups[0]]

        # Explode the selected columns
        exploded_df = df_pl.explode(columns=columns_to_explode)

        # Group by group and date
        data = (
            exploded_df
                .select([date_column, groups[0]] + value_column)
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .groupby([date_column, groups[0]])
                .agg(agg_columns)
                .sort([date_column, groups[0]])
        )
        
        if wide_format:
            # Value columns for aggregation
            values = data.select(pl.exclude([date_column, groups[0]])).columns

            # Pivot the data in Polars using the renamed columns
            data = (
                data.pivot(values=values, index=[date_column], columns=[groups[0]])
                    .fill_null(fillna)
            ).to_pandas()
        else:
            # Convert back to a pandas DataFrame
            data = data.to_pandas()
    elif isinstance(data, pd.DataFrame):
        # Convert the pandas DataFrame into a Polars DataFrame
        df_pl = pl.from_pandas(data).sort(date_column)

        # Group by date
        data = (df_pl
                .select([date_column] + value_column)
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .groupby([date_column])
                .agg(agg_columns)
                .sort([date_column]))
        
        if wide_format and len(value_column) > 1:
            # Pivot the data in Polars when multiple value columns are present
            values = data.select(pl.exclude([date_column])).columns
            data = (data.pivot(values=values, index=[date_column])
                    .fill_null(fillna)
                    ).to_pandas()
        else:
            # Convert back to a pandas DataFrame
            data = data.to_pandas()

    return data








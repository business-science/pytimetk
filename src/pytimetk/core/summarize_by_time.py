import pandas as pd
import pandas_flavor as pf
import polars as pl

from typing import Union, Callable, Tuple, List
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
    
    The `summarize_by_time` function aggregates data by a specified time period 
    and one or more numeric columns, allowing for grouping and customization of 
    the time-based aggregation.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        A pandas DataFrame or a pandas GroupBy object. This is the data that you 
        want to summarize by time.
    date_column : str
        The name of the column in the data frame that contains the dates or 
        timestamps to be aggregated by. This column must be of type datetime64.
    value_column : str or list
        The `value_column` parameter is the name of one or more columns in the 
        DataFrame that you want to aggregate by. It can be either a string 
        representing a single column name, or a list of strings representing 
        multiple column names.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be 
        aggregated. It accepts a string representing a pandas frequency offset, 
        such as "D" for daily or "MS" for month start. The default value is "D", 
        which means the data will be aggregated on a daily basis. Some common 
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
        
    agg_func : list, optional
        The `agg_func` parameter is used to specify one or more aggregating 
        functions to apply to the value column(s) during the summarization 
        process. It can be a single function or a list of functions. The default 
        value is `"sum"`, which represents the sum function. Some common 
        aggregating functions include:
        
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
        Custom `lambda` aggregating functions can be used too. Here are several 
        common examples:
        
        - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
        - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
        - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
        - ("range", lambda x: x.max() - x.min()): Range of values
        
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in 
        "wide" or "long" format. If set to `True`, the output will be in wide 
        format, where each group is represented by a separate column. If set to 
        False, the output will be in long format, where each group is represented 
        by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data 
        with. By default, it is set to 0. If you want to keep missing values as 
        NaN, you can use `np.nan` as the value for `fillna`.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        summarizing the data. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for summarizing the data. This can be faster than using "pandas" for 
          large datasets. 
    
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
    # Example 2 - Summarize by time with a GroupBy object (Wide Format), polars engine
    (
        df 
            .groupby(['category_1', 'frame_material']) 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = ['total_price', 'quantity'], 
                freq         = 'MS',
                agg_func     = 'sum',
                wide_format  = True, 
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
                    ('q25', lambda x: x.quantile(0.25)), 
                    ('q75', lambda x: x.quantile(0.75))
                ],
                wide_format  = False,
                engine       = 'pandas' 
            )
    )
    ```
    '''
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)
    
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
    
    
    # Get a list of unique first elements in the agg_dict values (used for renaming lambda columns)
    unique_first_elements = [func[0] for value in agg_dict.values() for func in value if isinstance(func, tuple)]

    if not unique_first_elements == []:
        for key, value in agg_dict.items():
            agg_dict[key] = [func[1] if isinstance(func, tuple) else func for func in value]
    
    
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

    # Rename any lambda columns
    if not unique_first_elements == []:        
        
        columns = data.columns
        
        names_iter = cycle(unique_first_elements)
        
        new_columns = [re.sub(pattern=r"<lambda.*?>",repl=next(names_iter), string=col) if '<lambda' in col else col for col in columns]
        
        data.columns = new_columns
    
    return data


def _summarize_by_time_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]],  
    freq: str = "D",  
    agg_func: Union[str, list] = 'sum',
    wide_format: bool = False,  
    fillna: int = 0,
) -> pd.DataFrame:
    polars_freq = pandas_to_polars_frequency(freq, default="d")
    if isinstance(value_column, str):
        value_column = [value_column]
    if not isinstance(agg_func, list):
        agg_func = [agg_func]

    aggregation_mapping = pandas_to_polars_aggregation_mapping(value_column)
    for func in agg_func:
        if func not in aggregation_mapping:
            raise ValueError(f"Unsupported aggregation function '{func}' for Polars.")

    agg_columns = [
        pl.col(val_col).sum().fill_null(fillna).alias(f"{val_col}_{func}")
        for val_col in value_column
        for func in agg_func
    ]

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        groups = data.grouper.names if isinstance(data.grouper.names, list) else [data.grouper.names]
        df_pl = pl.from_pandas(data.obj.copy(), include_index=False)[groups + [date_column] + value_column]
        
        # Get group combinations in pandas-like order
        group_combinations = df_pl.select(groups).unique(maintain_order=False).sort(groups).rows()
        
        data = (
            df_pl.lazy()
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .group_by(groups + [date_column], maintain_order=True)
                .agg(agg_columns)
                .sort(groups + [date_column])
                .collect(streaming=True)
        )
        
        if wide_format:
            values = data.select(pl.exclude([date_column] + groups)).columns
            data = data.pivot(
                values=values,
                index=[date_column],
                columns=groups,
                aggregate_function="sum"
            ).fill_null(fillna)
            
            # Rename columns to match pandas style and order
            new_columns = [date_column]
            for val_col in value_column:  # Preserve value_column order
                for group_vals in group_combinations:  # Use sorted group combinations
                    new_columns.append(f"{val_col}_{'_'.join(str(col) for col in group_vals)}")
            
            data = data.to_pandas()
            data.columns = new_columns
        else:
            data = data.to_pandas()
    
    elif isinstance(data, pd.DataFrame):
        df_pl = pl.from_pandas(data, include_index=False)[[date_column] + value_column]
        data = (
            df_pl.lazy()
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .group_by([date_column], maintain_order=True)
                .agg(agg_columns)
                .sort([date_column])
                .collect(streaming=True)
        )
        
        if wide_format and len(value_column) > 1:
            values = data.select(pl.exclude([date_column])).columns
            data = data.pivot(values=values, index=[date_column]).fill_null(fillna).to_pandas()
        else:
            data = data.to_pandas()

    return data








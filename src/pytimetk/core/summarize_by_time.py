import pandas as pd
import pandas_flavor as pf
import polars as pl

from typing import Union, Optional, Callable, Tuple
import re 
from itertools import cycle

from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def summarize_by_time_pandas(
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
        
        Custom `lambda` aggregating functions can be used too. Here are several common examples:
        
        - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
        - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
        - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
        - ("range", lambda x: x.max() - x.min()): Range of values
        
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in "wide" or "long" format. If set to `True`, the output will be in wide format, where each group is represented by a separate column. If set to False, the output will be in long format, where each group is represented by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data with. By default, it is set to 0. If you want to keep missing values as NaN, you can use `np.nan` as the value for `fillna`.
    
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
    # Summarize by time with a DataFrame object
    ( 
        df 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price',
                freq         = "MS",
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
                freq         = 'MS',
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
                freq         = 'MS',
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
                freq         = 'MS',
                agg_func     = ['sum', 'mean', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75))],
                wide_format  = True, 
            )
    )
    ```
    '''
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

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time_pandas = summarize_by_time_pandas


@pf.register_dataframe_method
def summarize_by_time_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_columns: Union[str, List[str]],  
    freq: str = "D",  
    agg_func: Union[str, list, Tuple[str, Callable]] = 'sum',
    wide_format: bool = False,  
    fillna: int = 0,
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
        
        Custom `lambda` aggregating functions can be used too. Here are several common examples:
        
        - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
        - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
        - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
        - ("range", lambda x: x.max() - x.min()): Range of values
        
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in "wide" or "long" format. If set to `True`, the output will be in wide format, where each group is represented by a separate column. If set to False, the output will be in long format, where each group is represented by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data with. By default, it is set to 0. If you want to keep missing values as NaN, you can use `np.nan` as the value for `fillna`.
    
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
    # Summarize by time with a DataFrame object
    ( 
        df 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price',
                freq         = "MS",
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
                freq         = 'MS',
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
                freq         = 'MS',
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
                freq         = 'MS',
                agg_func     = ['sum', 'mean', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75))],
                wide_format  = True, 
            )
    )
    ```
    '''
    # Mapping of frequency offsets between pandas and Polars
    frequency_mapping = {
        "S"   : "1s",
        "min" : "1m",
        "H"   : "1h",
        "D"   : "1d",
        "W"   : "1w",
        "M"   : "1mo",
        "MS"  : "1mo",
        "Q"   : "3mo",
        "QS"  : "3mo",
        "Y"   : "1y",
        "YS"  : "1y"
    }

    # Translate the pandas frequency offset to Polars
    polars_freq = frequency_mapping.get(freq, "1d")  # Default to daily if not found

    # Define a dictionary mapping aggregation function names to Polars aggregation expressions
    aggregation_mapping = {
        'sum'    : pl.col(value_columns) .sum().suffix("_sum"),
        'mean'   : pl.col(value_columns).mean().suffix("_mean"),
        'median' : pl.col(value_columns).median().suffix("_median"),
        'min'    : pl.col(value_columns).min().suffix("_min"),
        'max'    : pl.col(value_columns).max().suffix("_max"),
        'std'    : pl.col(value_columns).std().suffix("_std"),
        'var'    : pl.col(value_columns).var().suffix("_var"),
        'first'  : pl.col(value_columns).first().suffix("_first"),
        'last'   : pl.col(value_columns).last().suffix("_last"),
        'count'  : pl.col(value_columns).count().suffix("_count"),
        'nunique': pl.col(value_columns).n_unique().suffix("_nunique")
    }

    # If agg_func is a string, convert it to a list
    if isinstance(agg_func, str):
        agg_func = [agg_func]

    # Select columns for aggregation based on agg_func
    agg_columns = [aggregation_mapping[func] for func in agg_func if func in aggregation_mapping]

    # Check if the input data is a DataFrame or a GroupBy object
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Extract name from groupby object
        groups = data.grouper.names

        # Convert the GroupBy object into a Polars DataFrame
        df_pl = (pl.from_pandas(data.apply(lambda x: x))
                 .groupby(groups, maintain_order=True)
                 .agg(pl.all().sort_by(date_column)))

        # Create a list of column names to explode
        columns_to_explode = [col for col in df_pl.columns if col != groups[0]]

        # Explode the selected columns
        exploded_df = df_pl.explode(columns=columns_to_explode)

        # Group by group and date
        data = (exploded_df
                .select([date_column, groups[0]] + value_columns)
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .groupby([date_column, groups[0]])
                .agg(agg_columns)
                .sort([date_column, groups[0]]))
        
        if wide_format:
            # Value columns for aggregation
            values = data.select(pl.exclude([date_column, groups[0]])).columns

            # Pivot the data in Polars using the renamed columns
            data = (data.pivot(values=values, index=[date_column], columns=[groups[0]])
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
                .select([date_column] + value_columns)
                .with_columns(pl.col(date_column).dt.truncate(polars_freq))
                .groupby([date_column])
                .agg(agg_columns)
                .sort([date_column]))
        
        if wide_format and len(value_columns) > 1:
            # Pivot the data in Polars when multiple value columns are present
            values = data.select(pl.exclude([date_column])).columns
            data = (data.pivot(values=values, index=[date_column])
                    .fill_null(fillna)
                    ).to_pandas()
        else:
            # Convert back to a pandas DataFrame
            data = data.to_pandas()

    return data

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time_polars = summarize_by_time_polars


@pf.register_dataframe_method
def summarize_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_columns: Union[str, List[str]],
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
        
        Custom `lambda` aggregating functions can be used too. Here are several common examples:
        
        - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
        - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
        - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
        - ("range", lambda x: x.max() - x.min()): Range of values
        
    wide_format : bool, optional
        A boolean parameter that determines whether the output should be in "wide" or "long" format. If set to `True`, the output will be in wide format, where each group is represented by a separate column. If set to False, the output will be in long format, where each group is represented by a separate row. The default value is `False`.
    fillna : int, optional
        The `fillna` parameter is used to specify the value to fill missing data with. By default, it is set to 0. If you want to keep missing values as NaN, you can use `np.nan` as the value for `fillna`.
    
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
    # Summarize by time with a DataFrame object
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
    # Summarize by time with a GroupBy object (Long Format)
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
    # Summarize by time with a GroupBy object (Wide Format)
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
    # Summarize by time with a GroupBy object and multiple summaries (Wide Format)
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                date_column  = 'order_date', 
                value_column = 'total_price', 
                freq         = 'MS',
                agg_func     = ['sum', 'mean', ('q25', lambda x: x.quantile(0.25)), ('q75', lambda x: x.quantile(0.75))],
                wide_format  = True,
                engine       = 'pandas' 
            )
    )
    ```
    '''
    if engine == 'pandas':
        return summarize_by_time_pandas(data, date_column, value_columns, freq, agg_func, wide_format, fillna)
    elif engine == 'polars':
        return summarize_by_time_polars(data, date_column, value_columns, freq, agg_func, wide_format, fillna)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time = summarize_by_time


'''The `apply_by_time` function applies custom aggregation functions to a time series data, either in a
    wide or long format, and returns the result as a pandas DataFrame.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter represents the data on which the apply operation will be performed. It can be
    either a pandas DataFrame or a pandas DataFrameGroupBy object.
    date_column : str
        The `date_column` parameter is a string that represents the name of the column in the DataFrame
    that contains the dates. This column will be used as the index for resampling the data.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be resampled. It accepts a
    string representing a time frequency, such as "D" for daily, "W" for weekly, "M" for monthly, etc.
    The default value is "D", which means the data will
    wide_format : bool, optional
        The `wide_format` parameter is a boolean flag that determines whether the output should be in wide
    format or not. If `wide_format` is set to `True`, the output will have a multi-index column
    structure, where the first level represents the original columns and the second level represents the
    group names
    fillna : int, optional
        The `fillna` parameter is used to specify the value that will be used to fill missing values in the
    resulting DataFrame. By default, it is set to 0.
    
    Returns
    -------
        The function `apply_by_time` returns a pandas DataFrame object.
    
    '''

@pf.register_dataframe_method
def apply_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    freq: str = "D",
    wide_format: bool = False,
    fillna: int = 0,
    **named_funcs
) -> pd.DataFrame:
    '''Apply for time series.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter can be either a pandas DataFrame or a pandas DataFrameGroupBy object. It represents the data on which the apply operation will be performed.
    date_column : str
        The name of the column in the DataFrame that contains the dates.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be resampled. It accepts a string representing a time frequency, such as "D" for daily, "W" for weekly, "M" for monthly, etc. The default value is "D", which means the data will be resampled on a daily basis. Some common frequency aliases include:
        
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
        The `wide_format` parameter is a boolean flag that determines whether the output should be in wide format or not. If `wide_format` is set to `True`, the output will have a multi-index column structure, where the first level represents the original columns and the second level represents the group names
    fillna : int, optional
        The `fillna` parameter is used to specify the value that will be used to fill missing values in the resulting DataFrame. By default, it is set to 0.
    **named_funcs
        The `**named_funcs` parameter is used to specify one or more custom aggregation functions to apply to the data. It accepts named functions in the format:
        
        ``` python
        name = lambda df: df['column1'].corr(df['column2']])
        ```
        
        Where `name` is the name of the function and `df` is the DataFrame that will be passed to the function. The function must return a single value.
        
        
    
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

    return data


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.apply_by_time = apply_by_time

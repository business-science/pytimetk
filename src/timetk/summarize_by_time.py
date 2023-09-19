import pandas as pd
import pandas_flavor as pf
from timetk.utils.pandas_helpers import flatten_multiindex_column_names

@pf.register_dataframe_method
def summarize_by_time(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str,
    value_column: str or list,
    groups: str or list or None = None,
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
    """
    Applies one or more aggregating functions by a Pandas Period
    or TimeStamp to one or more numeric columns.

    Args:
        data (pd.DataFrame): 
            A pandas data frame with a date column and value
            column.
        date_column (str):
            The name of a single date or datetime column to be 
            aggregate by. Must be datetime64.
        value_column (str or list): 
            The names of one or more value columns to be 
            aggregated by.
        groups (str or list or None, optional):
            One or moe column names representing groups to 
            aggregate by. Defaults to None.
        rule (str, optional): 
            A pandas frequency (offset) such as "D" for Daily or 
            "MS" for month start. Defaults to "D".
        agg_func (function or list, optional):
            One or more aggregating functions such as np.sum.
            Defaults to np.sum.
        kind (str, optional):
            One of "timestamp" or "period". Defaults to "timestamp".
        wide_format (bool, optional): 
            Whether or not to return "wide" of "long format.
            Defaults to False.
        fillna (int, optional): 
            Value to fill missing data. Defaults to 0.
            If missing values are desired, use np.nan.
        flatten_column_names (bool, optional):
            Whether or not to flatten the multiindex column
            names. Defaults to True.
        reset_index (bool, optional):
            Whether or not to reset the index. Defaults to True.
        *args, **kwargs:
            Arguments passed to pd.DataFrame.agg()

    Returns:
        pd.DataFrame: A data frame that is summarized
            by time.
            
    Examples:
    import timetk
    import pandas as pd
    
    df = timetk.data.load_dataset('bike_sales_sample')
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Summarize by time with a DataFrame object
    ( 
        df 
            .summarize_by_time(
                date_column ='order_date', 
                value_column = 'total_price',
                groups = "category_2",
                rule = "MS",
                kind = 'timestamp',
                agg_func = ['mean', 'sum']
            )
    )
    
    # Summarize by time with a GroupBy object
    (
        df 
            .groupby('category_1') 
            .summarize_by_time(
                'order_date', 'total_price', 
                rule = 'MS',
                wide_format = True,
            )
    )
    """
    
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
    if groups is not None:
        data = data.groupby(groups)
    
    # Resample data based on the specified rule and kind
    data = data.resample(rule=rule, kind=kind)
    
    # Create a dictionary mapping each value column to the aggregating function(s)
    agg_dict = {col: agg_func for col in value_column}
    
    # Apply the aggregation using the agg method of the resampled data
    data = data.agg(agg_dict, *args, **kwargs)
    
    # Unstack the grouped columns if wide_format is True and groups is not None
    if wide_format:
        if groups is not None:
            data = data.unstack(groups)
        elif group_names is not None:
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
    
    return data

# Monkey patch the summarize_by_time method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.summarize_by_time = summarize_by_time

import pandas as pd
import numpy as np
import pandas_flavor as pf

@pf.register_dataframe_method
def summarize_by_time(
    data: pd.DataFrame,
    date_column: str,
    value_column: str or list,
    groups: str or list or None = None,
    rule: str = "D",
    agg_func: list = np.sum,
    kind: str = "timestamp",
    wide_format: bool = True,
    fillna: int = 0,
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
            Defaults to True.
        fillna (int, optional): 
            Value to fill is missing data. Defaults to 0.
            If missing values are desired, use np.nan.
        *args, **kwargs:
            Arguments passed to pd.DataFrame.agg()

    Returns:
        pd.DataFrame: A data frame that is summarized
            by time.
    """
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` is not a Pandas DataFrame.")
    
    # Convert value_column to a list if it is not already
    if not isinstance(value_column, list):
        value_column = [value_column]
    
    # Set the index of data to the date_column
    data = data.set_index(date_column)
    
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
    if wide_format and groups is not None:
        data = data.unstack(groups)
        if kind == 'period':
            data.index = data.index.to_period()
    
    # Fill missing values with the specified fillna value
    data = data.fillna(fillna)
    
    return data

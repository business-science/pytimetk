
import pandas as pd
import numpy as np

from typing import Union, List

def check_dataframe_or_groupby(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]) -> None:
    
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame or GroupBy object.")
        
    return None

def check_date_column(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], date_column: str) -> None:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
    
    if date_column not in data.columns:
        raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
    
    # Check if date_column is a datetime64[ns] dtype
    if data[date_column].dtype != 'datetime64[ns]':
        raise TypeError(f"`date_column` ({date_column}) is not a datetime64[ns] dtype.")
        
    return None

def check_value_column(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], value_column: Union[str, List[str]]) -> None:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
        
    if not isinstance(value_column, list):
        value_column = [value_column]
        
    for column in value_column:
        if column not in data.columns:
            raise ValueError(f"`value_column` ({column}) not found in `data`.")
        
        # Check if value_column is a numeric dtype
        if not np.issubdtype(data[column].dtype, np.number):
            raise TypeError(f"`value_column` ({column}) is not a numeric dtype.")
    
    return None

def check_series_or_datetime(data: Union[pd.Series, pd.DatetimeIndex]) -> None:
    
    if not isinstance(data, pd.Series):
        if not isinstance(data, pd.DatetimeIndex):
            raise TypeError("`data` is not a Pandas Series or DatetimeIndex.")
        
    return None
    
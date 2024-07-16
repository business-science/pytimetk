
import pandas as pd
import numpy as np
import polars as pl

from importlib.metadata import distribution, PackageNotFoundError

from typing import Union, List

def check_anomalize_data(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]) -> None:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
        
    expected_colnames = [
        'observed',
        'seasonal',
        'seasadj',
        'trend',
        'remainder',
        'anomaly',
        'anomaly_score',
        'anomaly_direction',
        'recomposed_l1',
        'recomposed_l2',
        'observed_clean'
    ]
    
    if not all([column in data.columns for column in expected_colnames]):
        raise ValueError(f"data does not have required colnames: {expected_colnames}. Did you run `anomalize()`?")
    
    return None


def check_data_type(data, authorized_dtypes: list, error_str=None):
    if not error_str:
        error_str = f'Input type must be one of {authorized_dtypes}'
    if not sum(map(lambda dtype: isinstance(data, dtype), authorized_dtypes)) > 0:
        raise TypeError(error_str)


def check_dataframe_or_groupby(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]) -> None:
    check_data_type(
        data, authorized_dtypes = [
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy
    ], error_str='`data` is not a Pandas DataFrame or GroupBy object.')


def check_dataframe_or_groupby_polars(data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy]) -> None:
    check_data_type(data, authorized_dtypes = [
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy
    ], error_str='`data` is not a Polars DataFrame or GroupBy object.')


def check_series_polars(data: pl.Series) -> None:
    check_data_type(data, authorized_dtypes = [pl.Series], 
                    error_str='Expected `data` to be a Polars Series.')


def check_date_column(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], date_column: str) -> None:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
    
    if date_column not in data.columns:
        raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
    
    # Check if date_column is a datetime64[ns] dtype    
    if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
        raise TypeError(f"`date_column` ({date_column}) is not a datetime64[ns] dtype. Dtype Found: {data[date_column].dtype}")
        
    return None

def check_value_column(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], value_column: Union[str, List[str]], require_numeric_dtype = True) -> None:
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
        
    if not isinstance(value_column, list):
        value_column = [value_column]
        
    for column in value_column:
        if column not in data.columns:
            raise ValueError(f"`value_column` ({column}) not found in `data`.")
        
        # Check if value_column is a numeric dtype
        if require_numeric_dtype:
            if not np.issubdtype(data[column].dtype, np.number):
                raise TypeError(f"`value_column` ({column}) is not a numeric dtype.")
    
    return None

def check_series_or_datetime(data: Union[pd.Series, pd.DatetimeIndex]) -> None:
    
    if not isinstance(data, pd.Series):
        if not isinstance(data, pd.DatetimeIndex):
            raise TypeError("`data` is not a Pandas Series or DatetimeIndex.")
        
    return None

def check_installed(package_name: str):
    try:
        distribution(package_name)
    except PackageNotFoundError:
        raise ImportError(f"The '{package_name}' package was not found in the active python environment. Please install it by running 'pip install {package_name}'.")


# def ensure_datetime64_date_column(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], date_column = str) -> Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
    
#     group_names = None
#     if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
#         group_names = list(data.groups.keys())
#         data = data.obj
    
#     if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
#         try:
#             data[date_column] = pd.to_datetime(data[date_column])
#             return data
#         except:
#             raise ValueError("Failed to convert series to datetime64.")
    
#     if group_names is not None:
#         data = data.groupby(group_names)    
    
#     return data



    
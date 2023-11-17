import numpy as np
import pandas as pd
import pandas_flavor as pf

from typing import Union

@pf.register_dataframe_method
def reduce_memory_usage(data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]):
  """
  Iterate through all columns of a Pandas DataFrame and modify the dtypes to reduce memory usage.

  Parameters:
  -----------
  data: pd.DataFrame
      Input dataframe to reduce memory usage.
  
  Returns:
  --------
  pd.DataFrame
    Dataframe with reduced memory usage.

  """
  
  if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
    try:
      data.obj = _reduce_memory(data.obj, convert_string_to_categorical = True)
    except:
      try:
        data.obj = _reduce_memory(data.obj, convert_string_to_categorical = False)
      except:
        data = data
    return data
  else:
    try:
      data = _reduce_memory(data, convert_string_to_categorical = True)
    except:
      try:
        data.obj = _reduce_memory(data.obj, convert_string_to_categorical = False)
      except:
        data = data
    return data

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.reduce_memory_usage = reduce_memory_usage

def _reduce_memory(
  data: pd.DataFrame, 
  convert_string_to_categorical: bool = True,
  # categorical_threshold: int = 100
):
  
  data = data.copy()
  
  # Iterate over each column in the dataframe
  for col in data.columns:
    # Get the current column dtype
    col_type = data[col].dtype

    # Check if column is boolean
    if col_type == bool:
      # If the column is boolean, convert it to int8 to save memory
      data = _convert_boolean_to_int8(data, col)

    # Check if the column is not an object (i.e., it's not a numeric column)
    elif col_type != object:
      # Get the minimum and maximum values of the current column
      c_min = data[col].min()
      c_max = data[col].max()

      # Check if the column is an integer type
      if str(col_type)[:3] == 'int':
        # Iterate over possible integer types and find the smallest type that can accomodate the column values
        for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
          if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
            data[col] = data[col].astype(dtype)
            break

      # Check if the column is a float type
      elif str(col_type)[:5] == 'float':
      # Iterate over possible float types and find the smallest type that can accomodate the column values
        # TODO - NEED TO BE CAREFUL HERE:
        # ISSUE #274 - Precision Effects Rounding
        for dtype in [np.float32, np.float64]:
          if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
            data[col] = data[col].astype(dtype)
            break

    # If the column is an object type, convert it to categorical type to save memory
    else:
      # TODO - NEED TO BE CAREFUL HERE: 
      # - Some object columns could be lists
      # - Some object columns could be dates
      # - Some users may expect string data returned
      if convert_string_to_categorical:
        if pd.api.types.is_string_dtype(data[col]):
          #if data[col].nunique() <= categorical_threshold:
          data[col] = data[col].astype('category')
  
  return data

def _convert_boolean_to_int8(data, col):
    """
    Convert a boolean column to int8 to save memory.
    """
    if data[col].dtype != bool:
      return data
    data[col] = data[col].astype(np.int8)
    return data
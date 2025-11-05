import numpy as np
import pandas as pd
import pandas_flavor as pf
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame

from typing import Union


@pf.register_groupby_method
@pf.register_dataframe_method
def reduce_memory_usage(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    *,
    convert_string_to_categorical: bool = True,
    inplace: bool = False,
):
    """
    Iterate through all columns of a Pandas DataFrame and modify the dtypes to reduce memory usage.

    Parameters:
    -----------
    data: pd.DataFrame
        Input dataframe to reduce memory usage.
    convert_string_to_categorical: bool, default True
        Convert string/object columns to categoricals when safe. Set to False to keep string dtype.
    inplace: bool, default False
        When True, mutate the supplied DataFrame (or backing GroupBy frame) in place instead of creating a copy.

    Returns:
    --------
    pd.DataFrame
      Dataframe with reduced memory usage.

    """

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        try:
            reduced = _reduce_memory(
                frame,
                convert_string_to_categorical=convert_string_to_categorical,
                inplace=inplace,
            )
        except Exception:
            if not convert_string_to_categorical:
                raise
            reduced = _reduce_memory(
                frame,
                convert_string_to_categorical=False,
                inplace=inplace,
            )

        if hasattr(data, "obj"):
            try:
                data.obj = reduced  # type: ignore[attr-defined]
            except AttributeError:
                if hasattr(data, "__dict__"):
                    data.__dict__["obj"] = reduced  # type: ignore[index]
        return data

    if isinstance(data, pd.DataFrame):
        try:
            return _reduce_memory(
                data,
                convert_string_to_categorical=convert_string_to_categorical,
                inplace=inplace,
            )
        except Exception:
            if not convert_string_to_categorical:
                raise
            return _reduce_memory(
                data,
                convert_string_to_categorical=False,
                inplace=inplace,
            )

    raise TypeError("Unsupported data type for reduce_memory_usage")


def _reduce_memory(
    data: pd.DataFrame,
    *,
    convert_string_to_categorical: bool = True,
    inplace: bool = False,
    # categorical_threshold: int = 100
):
    frame = data if inplace else data.copy()

    # Iterate over each column in the dataframe
    for col in frame.columns:
        # Get the current column dtype
        col_type = frame[col].dtype

        if pd.api.types.is_categorical_dtype(col_type):
            continue

        # Check if column is boolean
        if col_type == bool:
            # If the column is boolean, convert it to int8 to save memory
            frame = _convert_boolean_to_int8(frame, col)

        # Check if the column is not an object (i.e., it's not a numeric column)
        elif col_type != object:
            # Get the minimum and maximum values of the current column
            c_min = frame[col].min()
            c_max = frame[col].max()

            # Check if the column is an integer type
            if str(col_type)[:3] == "int":
                # Iterate over possible integer types and find the smallest type that can accomodate the column values
                for dtype in [
                    np.int8,
                    np.uint8,
                    np.int16,
                    np.uint16,
                    np.int32,
                    np.uint32,
                    np.int64,
                    np.uint64,
                ]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        frame[col] = frame[col].astype(dtype)
                        break

            # Check if the column is a float type
            elif str(col_type)[:5] == "float":
                # Iterate over possible float types and find the smallest type that can accomodate the column values
                # TODO - NEED TO BE CAREFUL HERE:
                # ISSUE #274 - Precision Effects Rounding
                for dtype in [np.float32, np.float64]:
                    if c_min > np.finfo(dtype).min and c_max < np.finfo(dtype).max:
                        frame[col] = frame[col].astype(dtype)
                        break

        # If the column is an object type, convert it to categorical type to save memory
        else:
            # TODO - NEED TO BE CAREFUL HERE:
            # - Some object columns could be lists
            # - Some object columns could be dates
            # - Some users may expect string data returned
            if convert_string_to_categorical and pd.api.types.is_string_dtype(frame[col]):
                frame[col] = frame[col].astype("category")

    return frame


def _convert_boolean_to_int8(data, col):
    """
    Convert a boolean column to int8 to save memory.
    """
    if data[col].dtype != bool:
        return data
    data[col] = data[col].astype(np.int8)
    return data

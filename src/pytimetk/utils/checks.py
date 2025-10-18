import numpy as np
import polars as pl

import pandas as pd
from importlib.metadata import distribution, PackageNotFoundError

from typing import Union, List, Iterable

try:  # Optional dependency for GPU acceleration
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore
    CudfDataFrameGroupBy = None  # type: ignore

from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame


def check_anomalize_data(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
) -> None:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = resolve_pandas_groupby_frame(data)

    expected_colnames = [
        "observed",
        "seasonal",
        "seasadj",
        "trend",
        "remainder",
        "anomaly",
        "anomaly_score",
        "anomaly_direction",
        "recomposed_l1",
        "recomposed_l2",
        "observed_clean",
    ]

    if not all([column in data.columns for column in expected_colnames]):
        raise ValueError(
            f"data does not have required colnames: {expected_colnames}. Did you run `anomalize()`?"
        )

    return None


def check_data_type(data, authorized_dtypes: list, error_str=None):
    if not error_str:
        error_str = f"Input type must be one of {authorized_dtypes}"
    if not sum(map(lambda dtype: isinstance(data, dtype), authorized_dtypes)) > 0:
        raise TypeError(error_str)


def check_dataframe_or_groupby(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        "pl.DataFrame",
        "pl.dataframe.group_by.GroupBy",
        "pl.LazyFrame",
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
) -> None:
    authorized: Iterable[type] = [
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
    ]

    try:
        authorized += [
            pl.DataFrame,
            pl.dataframe.group_by.GroupBy,
            pl.LazyFrame,
        ]  # type: ignore[attr-defined]
    except AttributeError:
        pass
    if cudf is not None:
        authorized += [cudf.DataFrame]  # type: ignore[attr-defined]
        if CudfDataFrameGroupBy is not None:
            authorized += [CudfDataFrameGroupBy]

    check_data_type(
        data,
        authorized_dtypes=list(authorized),
        error_str="`data` is not a Pandas DataFrame or GroupBy object.",
    )


def check_dataframe_or_groupby_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
) -> None:
    check_data_type(
        data,
        authorized_dtypes=[pl.DataFrame, pl.dataframe.group_by.GroupBy],
        error_str="`data` is not a Polars DataFrame or GroupBy object.",
    )


def check_series_polars(data: pl.Series) -> None:
    check_data_type(
        data,
        authorized_dtypes=[pl.Series],
        error_str="Expected `data` to be a Polars Series.",
    )


def check_date_column(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        "pl.DataFrame",
        "pl.dataframe.group_by.GroupBy",
        "pl.LazyFrame",
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
) -> None:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        dtype = frame[date_column].dtype if date_column in frame.columns else None
        if dtype is None:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        if not pd.api.types.is_datetime64_any_dtype(dtype):
            raise TypeError(
                f"`date_column` ({date_column}) is not a datetime64[ns] dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    if isinstance(data, pd.DataFrame):
        if date_column not in data.columns:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            raise TypeError(
                f"`date_column` ({date_column}) is not a datetime64[ns] dtype. "
                f"Dtype Found: {data[date_column].dtype}"
            )
        return None

    if isinstance(data, pl.dataframe.group_by.GroupBy):
        frame = data.df
        schema = frame.schema
        if date_column not in schema:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        dtype = schema[date_column]
        if not dtype.is_temporal():
            raise TypeError(
                f"`date_column` ({date_column}) is not a temporal dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    if isinstance(data, pl.DataFrame):
        schema = data.schema
        if date_column not in schema:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        dtype = schema[date_column]
        if not dtype.is_temporal():
            raise TypeError(
                f"`date_column` ({date_column}) is not a temporal dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    if isinstance(data, pl.LazyFrame):
        schema = data.schema
        if date_column not in schema:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        dtype = schema[date_column]
        if not dtype.is_temporal():
            raise TypeError(
                f"`date_column` ({date_column}) is not a temporal dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    if cudf is not None and CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        if date_column not in frame.columns:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        dtype = frame[date_column].dtype
        if not np.issubdtype(dtype, np.datetime64):
            raise TypeError(
                f"`date_column` ({date_column}) is not a datetime64 dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    if cudf is not None and isinstance(data, cudf.DataFrame):
        if date_column not in data.columns:
            raise ValueError(f"`date_column` ({date_column}) not found in `data`.")
        dtype = data[date_column].dtype
        if not np.issubdtype(dtype, np.datetime64):
            raise TypeError(
                f"`date_column` ({date_column}) is not a datetime64 dtype. "
                f"Dtype Found: {dtype}"
            )
        return None

    raise TypeError(
        "`data` is not a pandas/polars/cudf DataFrame, GroupBy, or LazyFrame."
    )


def check_value_column(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        "pl.DataFrame",
        "pl.dataframe.group_by.GroupBy",
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    value_column: Union[str, List[str]],
    require_numeric_dtype: bool = True,
) -> None:
    if not isinstance(value_column, list):
        value_column = [value_column]

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        _check_columns_pandas(frame, value_column, require_numeric_dtype)
        return None

    if isinstance(data, pd.DataFrame):
        _check_columns_pandas(data, value_column, require_numeric_dtype)
        return None

    if isinstance(data, pl.dataframe.group_by.GroupBy):
        frame = data.df
        _check_columns_polars(frame, value_column, require_numeric_dtype)
        return None

    if isinstance(data, pl.DataFrame):
        _check_columns_polars(data, value_column, require_numeric_dtype)
        return None

    if isinstance(data, pl.LazyFrame):
        _check_columns_polars(data, value_column, require_numeric_dtype)
        return None

    if cudf is not None and CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        _check_columns_cudf(frame, value_column, require_numeric_dtype)
        return None

    if cudf is not None and isinstance(data, cudf.DataFrame):
        _check_columns_cudf(data, value_column, require_numeric_dtype)
        return None

    raise TypeError(
        "`data` is not a pandas/polars/cudf DataFrame, GroupBy, or LazyFrame."
    )


def _check_columns_pandas(
    frame: pd.DataFrame,
    columns: List[str],
    require_numeric_dtype: bool,
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"`value_column` ({column}) not found in `data`.")
        if require_numeric_dtype and not np.issubdtype(frame[column].dtype, np.number):
            raise TypeError(f"`value_column` ({column}) is not a numeric dtype.")


def _check_columns_polars(
    frame: Union[pl.DataFrame, pl.LazyFrame],
    columns: List[str],
    require_numeric_dtype: bool,
) -> None:
    schema = frame.schema
    for column in columns:
        if column not in schema:
            raise ValueError(f"`value_column` ({column}) not found in `data`.")
        if require_numeric_dtype and not schema[column].is_numeric():
            raise TypeError(f"`value_column` ({column}) is not a numeric dtype.")


def _check_columns_cudf(
    frame: "cudf.DataFrame",
    columns: List[str],
    require_numeric_dtype: bool,
) -> None:
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"`value_column` ({column}) not found in `data`.")
        dtype = frame[column].dtype
        if require_numeric_dtype and not np.issubdtype(dtype, np.number):
            raise TypeError(f"`value_column` ({column}) is not a numeric dtype.")


def check_series_or_datetime(data: Union[pd.Series, pd.DatetimeIndex]) -> None:
    if not isinstance(data, pd.Series):
        if not isinstance(data, pd.DatetimeIndex):
            raise TypeError("`data` is not a Pandas Series or DatetimeIndex.")

    return None


def check_installed(package_name: str):
    try:
        distribution(package_name)
    except PackageNotFoundError:
        raise ImportError(
            f"The '{package_name}' package was not found in the active python environment. Please install it by running 'pip install {package_name}'."
        )


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

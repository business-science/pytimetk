import pandas as pd
import pandas_flavor as pf
import polars as pl

from typing import Union, Callable, Tuple, List
import re
import warnings
from itertools import cycle

from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)

from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    resolve_pandas_groupby_frame,
)

try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cudf = None  # type: ignore

# FUNCTIONS -------------------------------------------------------------------


@pf.register_groupby_method
@pf.register_dataframe_method
def summarize_by_time(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    freq: str = "D",
    agg_func: Union[str, list, Tuple[str, Callable]] = "sum",
    wide_format: bool = False,
    fillna: int = 0,
    engine: str = "pandas",
):
    """
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
        summarizing the data. It can be "pandas", "polars", or "cudf".

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
    # Example 2b - Summarize by time on a polars DataFrame using the tk accessor
    import polars as pl


    pl_df = pl.from_pandas(df)

    (
        pl_df
            .tk.summarize_by_time(
                date_column='order_date',
                value_column='total_price',
                freq='MS',
                agg_func='sum',
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
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)

    agg_has_custom = _agg_contains_custom(agg_func)
    agg_string_funcs: List[str] = []
    if not agg_has_custom:
        try:
            agg_string_funcs = _agg_collect_strings(agg_func)
        except TypeError:
            agg_has_custom = True

    engine_resolved = normalize_engine(engine, data)

    conversion_engine = engine_resolved
    if engine_resolved == "cudf":
        if agg_has_custom or wide_format:
            warnings.warn(
                "summarize_by_time cudf path: custom aggregations or wide_format=True "
                "are currently unsupported. Falling back to the pandas implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
            conversion_engine = "pandas"
        elif cudf is None:  # pragma: no cover - optional dependency
            raise ImportError("cudf is required for engine='cudf', but it is not installed.")
    conversion = convert_to_engine(data, conversion_engine)
    prepared = conversion.data

    if conversion_engine == "pandas":
        result = _summarize_by_time_pandas(
            prepared,
            date_column=date_column,
            value_column=value_column,
            freq=freq,
            agg_func=agg_func,
            wide_format=wide_format,
            fillna=fillna,
        )
    elif conversion_engine == "polars":
        result = _summarize_by_time_polars(
            prepared,
            date_column=date_column,
            value_column=value_column,
            freq=freq,
            agg_func=agg_func,
            wide_format=wide_format,
            fillna=fillna,
            conversion=conversion,
        )
    elif conversion_engine == "cudf":
        result = _summarize_by_time_cudf(
            prepared,
            date_column=date_column,
            value_column=value_column,
            freq=freq,
            agg_funcs=agg_string_funcs,
            fillna=fillna,
            conversion=conversion,
        )
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

    return restore_output_type(result, conversion)


def _summarize_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, list],
    freq: str = "D",
    agg_func: Union[str, list, Tuple[str, Callable]] = "sum",
    wide_format: bool = False,
    fillna: int = 0,
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
        data = resolve_pandas_groupby_frame(data).set_index(date_column).groupby(group_names)

    # Group data by the groups columns if groups is not None
    # if groups is not None:
    #     data = data.groupby(groups)

    # Resample data based on the specified freq
    data = data.resample(rule=freq, kind="timestamp")

    # Create a dictionary mapping each value column to the aggregating function(s)
    agg_dict = {col: agg_func for col in value_column}

    # Get a list of unique first elements in the agg_dict values (used for renaming lambda columns)
    unique_first_elements = [
        func[0]
        for value in agg_dict.values()
        for func in value
        if isinstance(func, tuple)
    ]

    if not unique_first_elements == []:
        for key, value in agg_dict.items():
            agg_dict[key] = [
                func[1] if isinstance(func, tuple) else func for func in value
            ]

    # Apply the aggregation using the dict method of the resampled data
    data = data.agg(func=agg_dict)

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

        new_columns = [
            re.sub(pattern=r"<lambda.*?>", repl=next(names_iter), string=col)
            if "<lambda" in col
            else col
            for col in columns
        ]

        data.columns = new_columns

    return data
 

def _summarize_by_time_cudf(
    prepared: Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"],
    date_column: str,
    value_column: Union[str, List[str]],
    freq: str,
    agg_funcs: List[str],
    fillna: int,
    conversion: FrameConversion,
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf summarize_by_time backend.")

    if hasattr(prepared, "obj"):
        df = resolve_pandas_groupby_frame(prepared).copy(deep=True)
    else:
        df = prepared.copy(deep=True)

    value_cols = [value_column] if isinstance(value_column, str) else list(value_column)
    agg_dict = {col: agg_funcs for col in value_cols}

    group_cols = conversion.group_columns or []
    sort_cols: List[str] = list(group_cols)
    sort_cols.append(date_column)
    df_sorted = df.sort_values(sort_cols)

    if date_column not in df_sorted.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    df_sorted[date_column] = cudf.to_datetime(df_sorted[date_column])

    def _flatten_columns(frame: "cudf.DataFrame") -> "cudf.DataFrame":
        rename_map = {}
        for col in frame.columns:
            if isinstance(col, tuple) and len(col) == 2:
                rename_map[col] = f"{col[0]}_{col[1]}"
        if rename_map:
            frame = frame.rename(columns=rename_map)
        return frame

    if group_cols:
        frames: List["cudf.DataFrame"] = []
        grouped = df_sorted.groupby(group_cols, sort=False)
        for keys, group_df in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            group_resampled = (
                group_df.set_index(date_column)
                .resample(freq)
                .agg(agg_dict)
                .reset_index()
            )
            for col_name, key_value in zip(group_cols, keys):
                group_resampled[col_name] = key_value
            frames.append(_flatten_columns(group_resampled))
        if not frames:
            result = cudf.DataFrame(columns=[date_column, *group_cols])
        else:
            result = cudf.concat(frames, ignore_index=True)
        ordered_cols = [date_column] + list(group_cols)
        for col in result.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        result = result[ordered_cols]
    else:
        result = (
            df_sorted.set_index(date_column)
            .resample(freq)
            .agg(agg_dict)
            .reset_index()
        )
        result = _flatten_columns(result)

    if fillna is not None:
        result = result.fillna(fillna)

    return result


def _summarize_by_time_polars(
    prepared: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    freq: str,
    agg_func: Union[str, List[str]],
    wide_format: bool,
    fillna: int,
    conversion: FrameConversion,
) -> pl.DataFrame:
    agg_funcs = [agg_func] if isinstance(agg_func, str) else list(agg_func)
    if any(not isinstance(func, str) for func in agg_funcs):
        raise ValueError(
            "Polars engine only supports string aggregation functions. "
            "Use the pandas engine for custom callables."
        )

    frame = (
        prepared.df if isinstance(prepared, pl.dataframe.group_by.GroupBy) else prepared
    )

    row_id_col = conversion.row_id_column
    if row_id_col and row_id_col in frame.columns:
        frame = frame.drop(row_id_col)

    pandas_frame = frame.to_pandas()

    if conversion.group_columns:
        pandas_data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy] = (
            pandas_frame.groupby(conversion.group_columns, sort=False)
        )
    else:
        pandas_data = pandas_frame

    pandas_result = _summarize_by_time_pandas(
        pandas_data,
        date_column=date_column,
        value_column=value_column,
        freq=freq,
        agg_func=agg_func,
        wide_format=wide_format,
        fillna=fillna,
    )

    return pl.from_pandas(pandas_result)


def _agg_contains_custom(agg_spec: Union[str, List, Tuple]) -> bool:
    if isinstance(agg_spec, tuple):
        return True
    if isinstance(agg_spec, list):
        return any(_agg_contains_custom(item) for item in agg_spec)
    return False


def _agg_collect_strings(agg_spec: Union[str, List]) -> List[str]:
    if isinstance(agg_spec, str):
        return [agg_spec]
    if isinstance(agg_spec, list):
        collected: List[str] = []
        for item in agg_spec:
            if isinstance(item, str):
                collected.append(item)
            elif isinstance(item, list):
                collected.extend(_agg_collect_strings(item))
            else:
                raise TypeError(
                    "Only string aggregations are supported for the cudf backend."
                )
        return collected
    raise TypeError("Only string aggregations are supported for the cudf backend.")

import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import List, Optional, Sequence, Tuple, Union

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_lags(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    lags: Union[int, Tuple[int, int], List[int]] = 1,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Adds lags to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_lags` function takes a Pandas DataFrame or GroupBy object, a
    date column, a value column or list of value columns, and a lag or list of
    lags, and adds lagged versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The input tabular data or grouped data to augment with lagged columns.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the
        column in the DataFrame that contains the dates. This column will be
        used to sort the data before adding the lagged values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you
        want to add lagged values for. It can be either a single column name
        (string) or a list of column names.
    lags : int or tuple or list, optional
        The `lags` parameter is an integer, tuple, or list that specifies the
        number of lagged values to add to the DataFrame.

        - If it is an integer, the function will add that number of lagged
          values for each column specified in the `value_column` parameter.

        - If it is a tuple, it will generate lags from the first to the second
          value (inclusive).

        - If it is a list, it will generate lags based on the values in the list.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. When "auto" (default) the backend is inferred from the
        input data type. Use "pandas" or "polars" to force a specific backend.

    Returns
    -------
    DataFrame
        A DataFrame with lagged columns appended. The returned object matches the
        backend of the input (pandas or polars).

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```

    ```{python}
    # Example 1 - Add 7 lagged values for a single DataFrame object, pandas engine
    lagged_df_single = (
        df
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=(1, 7),
                engine='pandas'
            )
    )
    lagged_df_single
    ```
    ```{python}
    # Example 2 - Add a single lagged value of 2 for each GroupBy object, polars engine
    lagged_df = (
        df
            .groupby('id')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=(1, 3),
                engine='polars'
            )
    )
    lagged_df
    ```

    ```{python}
    # Example 3 add 2 lagged values, 2 and 4, for a single DataFrame object, pandas engine
    lagged_df_single_two = (
        df
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=[2, 4],
                engine='pandas'
            )
    )
    lagged_df_single_two
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column, require_numeric_dtype=False)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_lags_pandas(
            data=sorted_data,
            date_column=date_column,
            value_column=value_column,
            lags=lags,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_lags_polars(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            lags=lags,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_lags_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    lags: Union[int, Tuple[int, int], List[int]] = 1,
) -> pd.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    lags = _normalize_shift_values(lags, label="lags")

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, apply lag function
    if isinstance(data, pd.DataFrame):
        df = data.copy()

        for col in value_column:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    # GROUPED EXTENSION - If data is a GroupBy object, add lags by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        for col in value_column:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df.groupby(group_names)[col].shift(lag)

    return df


def _augment_lags_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    lags: Union[int, Tuple[int, int], List[int]],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    lags = _normalize_shift_values(lags, label="lags")
    resolved_groups: Sequence[str] = (
        list(group_columns) if group_columns else _resolve_group_columns(data)
    )
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    lag_columns = []
    for col in value_column:
        for lag in lags:
            expr = pl.col(col).shift(lag)
            if resolved_groups:
                expr = expr.over(resolved_groups)
            lag_columns.append(expr.alias(f"{col}_lag_{lag}"))

    augmented = sorted_frame.with_columns(lag_columns).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _normalize_shift_values(
    values: Union[int, Tuple[int, int], List[int]],
    label: str,
) -> List[int]:
    if isinstance(values, int):
        return [values]
    if isinstance(values, tuple):
        if len(values) != 2:
            raise ValueError(f"Invalid {label} specification: tuple must be length 2.")
        start, end = values
        return list(range(start, end + 1))
    if isinstance(values, list):
        return [int(v) for v in values]
    raise TypeError(
        f"Invalid {label} specification: type: {type(values)}. Please use int, tuple, or list."
    )


def _resolve_group_columns(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
) -> Sequence[str]:
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        columns = []
        for entry in data.by:
            if isinstance(entry, str):
                columns.append(entry)
            elif hasattr(entry, "meta"):
                columns.append(entry.meta.output_name())
            elif isinstance(entry, list) and entry and all(
                isinstance(item, str) for item in entry
            ):
                columns.extend(entry)
            else:
                raise TypeError("Unsupported polars group key type.")
        return columns
    return []

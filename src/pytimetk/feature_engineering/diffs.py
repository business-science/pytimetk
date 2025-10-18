import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import List, Optional, Sequence, Tuple, Union

try:  # Optional dependency for GPU acceleration
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore
    CudfDataFrameGroupBy = None  # type: ignore

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
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_diffs(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    normalize: bool = False,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Adds differences and percentage difference (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_diffs` function takes a Pandas DataFrame or GroupBy object, a
    date column, a value column or list of value columns, and a period or list of
    periods, and adds differenced versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The input data to augment with differenced columns.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the
        column in the DataFrame that contains the dates. This column will be
        used to sort the data before adding the differenced values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you
        want to add differences values for. It can be either a single column name
        (string) or a list of column names.
    periods : int or tuple or list, optional
        The `periods` parameter is an integer, tuple, or list that specifies the
        periods to shift values when differencing.

        - If it is an integer, the function will add that number of differences
          values for each column specified in the `value_column` parameter.

        - If it is a tuple, it will generate differences from the first to the second
          value (inclusive).

        - If it is a list, it will generate differences based on the values in the list.
    normalize : bool, optional
        The `normalize` parameter is used to specify whether to normalize the
        differenced values as a percentage difference. Default is False.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : {"auto", "pandas", "polars", "cudf"}, optional
        Execution engine. When "auto" (default) the backend is inferred from the
        input data type. Use "pandas", "polars", or "cudf" to force a specific backend.

    Returns
    -------
    DataFrame
        DataFrame with differenced columns added. The return type matches the
        input backend.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```

    ```{python}
    # Example 1 - Add 7 differenced values for a single DataFrame object (pandas)
    diffed_df_single = (
        df
            .query('id == "D10"')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=(1, 7)
            )
    )
    diffed_df_single.glimpse()
    ```
    ```{python}
    # Example 2 - Add differenced values via the polars accessor
    diffed_df = (
        pl.from_pandas(df)
        .group_by('id')
        .tk.augment_diffs(
            date_column='date',
            value_column='value',
            periods=2,
        )
    )
    diffed_df
    ```

    ```{python}
    # Example 3 add 2 differenced values, 2 and 4, for a single DataFrame object (pandas)
    diffed_df_single_two = (
        df
            .query('id == "D10"')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=[2, 4]
            )
    )
    diffed_df_single_two
    ```
    """
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved in ("polars", "cudf"):
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_diffs_pandas(
            data=sorted_data,
            value_column=value_column,
            periods=periods,
            normalize=normalize,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif engine_resolved == "polars":
        result = _augment_diffs_polars(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            periods=periods,
            normalize=normalize,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )
    elif engine_resolved == "cudf":
        result = _augment_diffs_cudf(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            periods=periods,
            normalize=normalize,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )
    else:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Unhandled engine: {engine_resolved}")

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_diffs_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    value_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]],
    normalize: bool,
) -> pd.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    periods = _normalize_shift_values(periods, label="periods")

    if isinstance(data, pd.DataFrame):
        df = data.copy()

        if normalize:
            for col in value_column:
                for period in periods:
                    df[f"{col}_pctdiff_{period}"] = df[col].pct_change(period)
        else:
            for col in value_column:
                for period in periods:
                    df[f"{col}_diff_{period}"] = df[col].diff(period)

        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = resolve_pandas_groupby_frame(data).copy()

        if normalize:
            for col in value_column:
                for period in periods:
                    base_df[f"{col}_pctdiff_{period}"] = base_df.groupby(group_names)[
                        col
                    ].pct_change(period)
        else:
            for col in value_column:
                for period in periods:
                    base_df[f"{col}_diff_{period}"] = base_df.groupby(group_names)[
                        col
                    ].diff(period)

        return base_df

    raise TypeError("Unsupported data type passed to _augment_diffs_pandas.")


def _augment_diffs_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]],
    normalize: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    periods = _normalize_shift_values(periods, label="periods")
    resolved_groups = resolve_polars_group_columns(data, group_columns)

    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    diff_columns = []
    for col in value_column:
        for period in periods:
            if normalize:
                expr = pl.col(col).pct_change(n=period)
                name = f"{col}_pctdiff_{period}"
            else:
                expr = pl.col(col).diff(n=period)
                name = f"{col}_diff_{period}"

            if resolved_groups:
                expr = expr.over(resolved_groups)

            diff_columns.append(expr.alias(name))

    augmented = sorted_frame.with_columns(diff_columns).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _augment_diffs_cudf(
    data: Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"],
    date_column: str,
    value_column: Union[str, List[str]],
    periods: Union[int, Tuple[int, int], List[int]],
    normalize: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
):
    if cudf is None:
        raise ImportError(
            "cudf is required for GPU execution but is not installed. "
            "Install pytimetk with the 'gpu' extra."
        )

    if isinstance(value_column, str):
        value_column = [value_column]

    periods = _normalize_shift_values(periods, label="periods")

    if CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data).copy(deep=True)
        resolved_groups: Sequence[str] = list(group_columns) if group_columns else []
    else:
        frame = data.copy(deep=True)  # type: ignore[assignment]
        resolved_groups = list(group_columns) if group_columns else []

    temp_row_col = row_id_column
    generated_row_id = False
    if temp_row_col is None or temp_row_col not in frame.columns:
        temp_base = "__pytimetk_row_id__"
        temp_row_col = temp_base
        suffix = 0
        while temp_row_col in frame.columns:
            suffix += 1
            temp_row_col = f"{temp_base}_{suffix}"
        frame[temp_row_col] = cudf.Series(range(len(frame)), dtype="int64")
        generated_row_id = True

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    frame = frame.sort_values(sort_keys)

    grouped = frame.groupby(resolved_groups, sort=False) if resolved_groups else None

    for col in value_column:
        for period in periods:
            target_col = f"{col}_{'pctdiff' if normalize else 'diff'}_{period}"
            if normalize:
                if grouped is not None:
                    series = grouped[col].pct_change(periods=period)
                else:
                    series = frame[col].pct_change(periods=period)
            else:
                if grouped is not None:
                    series = grouped[col].diff(periods=period)
                else:
                    series = frame[col].diff(periods=period)
            frame[target_col] = series

    frame = frame.sort_values(temp_row_col)

    if generated_row_id:
        frame = frame.drop(columns=[temp_row_col])

    return frame


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

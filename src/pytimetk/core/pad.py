import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Optional, Union, Sequence, List

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
)
from pytimetk.utils.dataframe_ops import (
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    conversion_to_pandas,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
)
from pytimetk.utils.polars_helpers import pandas_to_polars_frequency
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.datetime_helpers import parse_human_duration

try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore

@pf.register_groupby_method
@pf.register_dataframe_method
def pad_by_time(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: Union[str, ColumnSelector],
    freq: str = "D",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    engine: str = "pandas",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Make irregular time series regular by padding with missing dates.

    The `pad_by_time` function inserts missing dates into a Pandas DataFrame or
    DataFrameGroupBy object, through the process making an irregularly spaced
    time series regularly spaced.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The `data` parameter can be either a pandas/polars DataFrame or a grouped
        object. It represents the data that you want to pad with missing dates.
    date_column : str or ColumnSelector
        Column containing the timestamps used to determine padding bounds.
    freq : str, optional
        Frequency for padding. Accepts pandas aliases (``"H"``, ``"MS"``, ...)
        or human-friendly durations like ``"15 minutes"`` or ``"3 days"``.

        - S: secondly frequency
        - min: minute frequency
        - H: hourly frequency
        - B: business day frequency
        - D: daily frequency
        - W: weekly frequency
        - M: month end frequency
        - MS: month start frequency
        - BMS: Business month start
        - Q: quarter end frequency
        - QS: quarter start frequency
        - Y: year end frequency
        - YS: year start frequency
    start_date, end_date : str, optional
        Optional bounds for padding. Accepts ISO strings or durations relative
        to the observed min/max (e.g., ``"start: 1 month ago"``).
    engine : {"pandas", "polars", "cudf", "auto"}, optional
        Execution engine. ``"pandas"`` (default) performs the computation using pandas.
        ``"polars"`` converts the result to a polars DataFrame on return. ``"auto"``
        infers the engine from the input data.


    Returns
    -------
    DataFrame
        The function `pad_by_time` returns a DataFrame extended with the padded dates.
        The concrete type matches the engine used to process the data.

    Notes
    -----

    ## Performance

    This function uses a number of techniques to speed up computation for large
    datasets with many time series groups.

    - We use a vectorized approach to generate the Cartesian product of all
      unique group values and all dates in the date range.
    - We then merge this Cartesian product with the original data to introduce
      NaN values for missing rows. This approach is much faster than looping
      through each group and applying a function to each group.

    Note: There is no parallel processing since the vectorized approach is
          almost always faster.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
    df
    ```

    ```{python}
    # Pad Single Time Series: Fill missing dates
    padded_df = (
        df
            .query('symbol == "AAPL"')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D'
            )
    )
    padded_df
    ```

    ```{python}
    # Pad by Group: Pad each group with missing dates
    padded_df = (
        df
            .groupby('symbol')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D'
            )
    )
    padded_df
    ```

    ```{python}
    # Pad with end dates specified
    padded_df = (
        df
            .groupby('symbol')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D',
                start_date  = '2013-01-01',
                end_date    = '2023-09-22'
            )
    )
    padded_df.query('symbol == "AAPL"')
    ```

    ```{python}
    # Polars DataFrame using the tk accessor
    import pandas as pd
    import polars as pl


    sample = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=3, freq="D"),
            "value": [1, 2, 3],
        }
    )

    pl_df = pl.from_pandas(sample)

    pl_df.tk.pad_by_time(
        date_column='date',
        freq='D',
    )
    ```
    """
    # Common checks
    check_dataframe_or_groupby(data)

    def _resolve_date_col(obj):
        if isinstance(date_column, str):
            return date_column
        resolved = resolve_column_selection(
            obj, date_column, allow_none=False, require_match=True
        )
        if len(resolved) != 1:
            raise ValueError(
                f"`date_column` selector must resolve to exactly one column (resolved={resolved})."
            )
        return resolved[0]

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        resolved_frame = resolve_pandas_groupby_frame(data)
        date_column = _resolve_date_col(resolved_frame)
        check_date_column(resolved_frame, date_column)
    elif isinstance(data, pd.DataFrame):
        date_column = _resolve_date_col(data)
        check_date_column(data, date_column)
    elif isinstance(data, pl.dataframe.group_by.GroupBy):
        base = getattr(data, "df", None)
        if base is None:
            raise TypeError(
                "Unable to resolve columns from this polars GroupBy for selector resolution."
            )
        date_column = _resolve_date_col(base.to_pandas())
    elif isinstance(data, pl.DataFrame):
        date_column = _resolve_date_col(data.to_pandas())
    else:
        date_column = _resolve_date_col(
            resolve_pandas_groupby_frame(data)
            if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy)
            else data
        )

    engine_resolved = normalize_engine(engine, data)

    def _parse_date_bound(text: Optional[str]) -> Optional[pd.Timestamp]:
        if text is None:
            return None
        try:
            return pd.Timestamp(text)
        except Exception:
            duration = parse_human_duration(text)
            reference = None
            if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
                reference = resolve_pandas_groupby_frame(data)[date_column].min()
            elif isinstance(data, pd.DataFrame):
                reference = data[date_column].min()
            elif isinstance(data, pl.DataFrame):
                reference = data.to_pandas()[date_column].min()
            else:
                reference = resolve_pandas_groupby_frame(data)[date_column].min()
            if reference is None:
                raise ValueError(f"Unable to resolve reference date for '{text}'.")
            if isinstance(duration, pd.DateOffset):
                return pd.Timestamp(reference) + duration
            delta = pd.to_timedelta(duration)
            return pd.Timestamp(reference) + delta

    start_ts = _parse_date_bound(start_date)
    end_ts = _parse_date_bound(end_date)

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("Start date cannot be greater than end date.")

    resolved_freq = freq
    if isinstance(freq, str):
        try:
            pd.tseries.frequencies.to_offset(freq)
        except Exception:
            duration = parse_human_duration(freq)
            if isinstance(duration, pd.DateOffset):
                resolved_freq = duration
            else:
                resolved_freq = pd.to_timedelta(duration)

    if engine_resolved == "pandas":
        conversion = convert_to_engine(data, "pandas")
        prepared = conversion.data
        result = _pad_by_time_pandas(
            prepared,
            date_column=date_column,
            freq=resolved_freq,
            start_date=start_ts,
            end_date=end_ts,
        )
        return restore_output_type(result, conversion)

    if engine_resolved == "polars":
        conversion = convert_to_engine(data, "polars")
        prepared = conversion.data
        result = _pad_by_time_polars(
            prepared,
            date_column=date_column,
            freq=resolved_freq,
            start_date=start_ts,
            end_date=end_ts,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )
        return restore_output_type(result, conversion)

    if engine_resolved == "cudf":
        if cudf is None:  # pragma: no cover - optional dependency
            raise ImportError("cudf is required for engine='cudf', but it is not installed.")
        conversion = convert_to_engine(data, "cudf")
        prepared = conversion.data
        result = _pad_by_time_cudf_dataframe(
            prepared,
            date_column=date_column,
            freq=resolved_freq,
            start_date=start_ts,
            end_date=end_ts,
            group_columns=conversion.group_columns,
        )
        return restore_output_type(result, conversion)

    raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")


def _pad_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    *,
    date_column: str,
    freq: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=[date_column], inplace=True)

        min_date = start_date if start_date is not None else df[date_column].min()
        max_date = end_date if end_date is not None else df[date_column].max()

        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        padded_df = pd.DataFrame({date_column: date_range})
        padded_df = padded_df.merge(df, on=[date_column], how="left")

        padded_df.sort_values(by=[date_column], inplace=True)
        padded_df.reset_index(drop=True, inplace=True)

        col_name_candidates = padded_df.columns[padded_df.nunique() == 1]
        col_name = col_name_candidates[0] if not col_name_candidates.empty else None

        if col_name is not None:
            padded_df = padded_df.assign(**{col_name: padded_df[col_name].ffill()})

        return padded_df

    group_names = data.grouper.names
    df = resolve_pandas_groupby_frame(data).copy()

    df[date_column] = pd.to_datetime(df[date_column])
    df.sort_values(by=[*group_names, date_column], inplace=True)

    padded_df = _pad_by_time_vectorized(
        data=df,
        date_column=date_column,
        groupby_columns=list(group_names),
        freq=freq,
        start_date=start_date,
        end_date=end_date,
    )

    return padded_df[df.columns]


def _pad_by_time_vectorized(
    data: pd.DataFrame,
    date_column: str,
    groupby_columns: list,
    freq: str = "D",
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    # Calculate the overall min and max dates across the entire dataset if not provided
    if start_date is None:
        start_date = data[date_column].min()
    if end_date is None:
        end_date = data[date_column].max()

    # Create a full date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Generate the Cartesian product of all_dates and unique group values
    idx = pd.MultiIndex.from_product(
        [data[col].unique() for col in groupby_columns] + [all_dates],
        names=groupby_columns + [date_column],
    )
    cartesian_df = pd.DataFrame(index=idx).reset_index()

    # Merge to introduce NaN values for missing rows
    padded_data = pd.merge(
        cartesian_df, data, on=groupby_columns + [date_column], how="left"
    )

    return padded_data


def _pad_by_time_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    *,
    date_column: str,
    freq: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    if date_column not in frame.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    freq_polars = pandas_to_polars_frequency(freq)
    dtype_date = frame.schema[date_column]

    # Determine padding bounds
    if start_date is None:
        start_value = frame.select(pl.col(date_column).min()).to_series().item()
    else:
        start_value = start_date.to_pydatetime()

    if end_date is None:
        end_value = frame.select(pl.col(date_column).max()).to_series().item()
    else:
        end_value = end_date.to_pydatetime()

    if start_value is None or end_value is None:
        raise ValueError("Unable to determine start or end date for padding.")

    date_range = pl.date_range(
        start=start_value,
        end=end_value,
        interval=freq_polars,
        eager=True,
    )
    if dtype_date == pl.Date:
        date_range = date_range.cast(pl.Date)
    elif isinstance(dtype_date, pl.datatypes.Datetime):
        date_range = date_range.cast(
            pl.Datetime(time_unit=dtype_date.time_unit, time_zone=dtype_date.time_zone)
        )

    date_df = pl.DataFrame({date_column: date_range})

    if resolved_groups:
        groups_df = frame.select(resolved_groups).unique()
        cartesian = groups_df.join(date_df, how="cross")
        join_keys = list(resolved_groups) + [date_column]
    else:
        cartesian = date_df
        join_keys = [date_column]

    padded = cartesian.join(frame, on=join_keys, how="left")

    # Forward fill constant columns for non-grouped data
    if not resolved_groups:
        constant_cols: List[str] = []
        for col_name in frame.columns:
            if col_name in join_keys:
                continue
            unique_count = frame.select(pl.col(col_name).n_unique()).to_series().item()
            if unique_count == 1:
                constant_cols.append(col_name)
        if constant_cols:
            padded = padded.with_columns(
                [pl.col(col).forward_fill() for col in constant_cols]
            )

    padded = padded.sort(join_keys)
    ordered_cols = join_keys + [col for col in frame.columns if col not in join_keys]
    padded = padded.select(ordered_cols)

    # Drop row id column if it exists to avoid leaking conversion internals.
    if row_id_column and row_id_column in padded.columns:
        padded = padded.drop(row_id_column)

    return padded


def _pad_by_time_cudf_dataframe(
    data: Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"],
    *,
    date_column: str,
    freq: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    group_columns: Optional[Sequence[str]],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf pad_by_time backend.")

    if hasattr(data, "obj"):
        df = resolve_pandas_groupby_frame(data).copy(deep=True)
    else:
        df = data.copy(deep=True)

    if date_column not in df.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    df[date_column] = cudf.to_datetime(df[date_column])
    group_cols = list(group_columns) if group_columns else []

    frames: List["cudf.DataFrame"] = []

    if group_cols:
        grouped = df.groupby(group_cols, sort=False)
        for keys, group_df in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            min_date_value = start_date if start_date is not None else group_df[date_column].min()
            max_date_value = end_date if end_date is not None else group_df[date_column].max()
            start_val = _convert_to_datetime(min_date_value)
            end_val = _convert_to_datetime(max_date_value)
            date_range = cudf.Series(cudf.date_range(start=start_val, end=end_val, freq=freq))
            padded = cudf.DataFrame({date_column: date_range})
            for col_name, key_value in zip(group_cols, keys):
                padded[col_name] = key_value
            merged = padded.merge(
                group_df,
                on=group_cols + [date_column],
                how="left",
                sort=False,
            )
            merged = merged.sort_values(by=group_cols + [date_column])
            for col in merged.columns:
                if col not in group_cols + [date_column]:
                    merged[col] = merged[col].fillna(method="ffill")
            frames.append(merged)

        result = cudf.concat(frames, ignore_index=True) if frames else cudf.DataFrame(columns=group_cols + [date_column])
        ordered_cols = group_cols + [date_column]
        other_cols = [col for col in result.columns if col not in ordered_cols]
        return result[ordered_cols + other_cols]

    min_date_value = start_date if start_date is not None else df[date_column].min()
    max_date_value = end_date if end_date is not None else df[date_column].max()
    start_val = _convert_to_datetime(min_date_value)
    end_val = _convert_to_datetime(max_date_value)
    date_range = cudf.Series(cudf.date_range(start=start_val, end=end_val, freq=freq))
    padded = cudf.DataFrame({date_column: date_range})
    merged = padded.merge(df, on=[date_column], how="left", sort=False)
    merged = merged.sort_values(by=[date_column])
    const_cols = [col for col in merged.columns if col != date_column and merged[col].nunique() == 1]
    for col in const_cols:
        merged[col] = merged[col].fillna(method="ffill")
    return merged


def _convert_to_datetime(value: Union[pd.Timestamp, "cudf.Scalar", None]) -> pd.Timestamp:
    if value is None:
        raise ValueError("Unable to determine start or end date for padding.")
    if isinstance(value, pd.Timestamp):
        return value
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.to_datetime(value)

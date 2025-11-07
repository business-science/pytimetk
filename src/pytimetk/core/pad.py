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
from functools import lru_cache
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.datetime_helpers import parse_human_duration, normalize_frequency_alias


def _resolve_selector_frame(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
) -> pd.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        return resolve_pandas_groupby_frame(data).copy()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        base = getattr(data, "df", None)
        if base is None:
            raise TypeError(
                "Unable to resolve columns from this polars GroupBy for selector resolution."
            )
        return base.to_pandas()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    raise TypeError(
        "Column selectors currently require pandas or polars data for `pad_by_time`."
    )


def _normalize_frequency(freq: Optional[str]) -> Union[str, pd.DateOffset, pd.Timedelta]:
    if freq is None:
        return "D"
    if isinstance(freq, (pd.DateOffset, pd.Timedelta)):
        return freq
    if isinstance(freq, str):
        freq = normalize_frequency_alias(freq)
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except Exception:
        duration = parse_human_duration(freq)
        if isinstance(duration, pd.DateOffset):
            return duration
        return pd.to_timedelta(duration)


def _freq_to_string(freq: Union[str, pd.DateOffset, pd.Timedelta]) -> str:
    if isinstance(freq, str):
        return freq
    try:
        return pd.tseries.frequencies.to_offset(freq).freqstr
    except Exception:
        return str(freq)


@lru_cache(maxsize=128)
def _cached_polars_date_range(start_iso: str, end_iso: str, freq_str: str):
    return pl.date_range(
        start=pd.Timestamp(start_iso),
        end=pd.Timestamp(end_iso),
        interval=freq_str,
        eager=True,
    )

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
    fillna: Optional[Union[int, float]] = None,
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
    fillna : scalar, optional
        When provided, all newly padded rows have their non-date/group columns
        filled with this value instead of the default forward/backward fill.
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

    selector_snapshot = _resolve_selector_frame(data)

    def _parse_date_bound(text: Optional[str], reference: pd.Series) -> Optional[pd.Timestamp]:
        if text is None:
            return None
        try:
            return pd.Timestamp(text)
        except Exception:
            duration = parse_human_duration(text)
            base = pd.Timestamp(reference.min())
            if isinstance(duration, pd.DateOffset):
                return base + duration
            delta = pd.to_timedelta(duration)
            return base + delta

    base_series = pd.Series(selector_snapshot[date_column])
    start_ts = _parse_date_bound(start_date, base_series)
    end_ts = _parse_date_bound(end_date, base_series)

    if not base_series.empty:
        if start_ts is None:
            start_ts = pd.Timestamp(base_series.min())
        if end_ts is None:
            end_ts = pd.Timestamp(base_series.max())

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError("Start date cannot be greater than end date.")

    freq_normalized = _normalize_frequency(freq)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "pandas":
        conversion = convert_to_engine(data, "pandas")
        prepared = conversion.data
        result = _pad_by_time_pandas(
            prepared,
            date_column=date_column,
            freq=freq_normalized,
            start_date=start_ts,
            end_date=end_ts,
            fillna=fillna,
        )
        return restore_output_type(result, conversion)

    if engine_resolved == "polars":
        conversion = convert_to_engine(data, "polars")
        prepared = conversion.data
        result = _pad_by_time_polars(
            prepared,
            date_column=date_column,
            freq=_freq_to_string(freq_normalized),
            start_date=start_ts,
            end_date=end_ts,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
            fillna=fillna,
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
            freq=freq_normalized,
            start_date=start_ts,
            end_date=end_ts,
            group_columns=conversion.group_columns,
            fillna=fillna,
        )
        return restore_output_type(result, conversion)

    raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")


def _pad_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    *,
    date_column: str,
    freq: Union[str, pd.DateOffset, pd.Timedelta],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    fillna: Optional[Union[int, float]],
) -> pd.DataFrame:
    def _build_range(start_bound, end_bound):
        return pd.date_range(
            start=start_bound,
            end=end_bound,
            freq=freq,
        )

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=[date_column], inplace=True)

        min_date = start_date if start_date is not None else df[date_column].min()
        max_date = end_date if end_date is not None else df[date_column].max()

        base = (
            df.set_index(date_column)
            .reindex(_build_range(min_date, max_date))
            .reset_index()
            .rename(columns={"index": date_column})
        )
        if fillna is not None:
            fill_cols = [col for col in base.columns if col != date_column]
            base[fill_cols] = base[fill_cols].fillna(fillna)
            return base
        padded = base

        constant_cols = padded.columns[padded.nunique(dropna=False) == 1]
        if len(constant_cols) > 0:
            padded[constant_cols] = padded[constant_cols].ffill()

        return padded

    group_names = data.grouper.names
    df = resolve_pandas_groupby_frame(data).copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df.sort_values(by=[*group_names, date_column], inplace=True)

    grouped = df.groupby(group_names, sort=False, group_keys=False)

    padded_frames: List[pd.DataFrame] = []
    for keys, group_df in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        start_bound = start_date if start_date is not None else group_df[date_column].min()
        end_bound = end_date if end_date is not None else group_df[date_column].max()

        padded = (
            group_df.set_index(date_column)
            .reindex(_build_range(start_bound, end_bound))
            .reset_index()
            .rename(columns={"index": date_column})
        )

        for col_name, key_value in zip(group_names, keys):
            padded[col_name] = key_value

        if fillna is not None:
            fill_cols = [col for col in padded.columns if col not in (*group_names, date_column)]
            padded[fill_cols] = padded[fill_cols].fillna(fillna)
        else:
            constant_cols = [
                col
                for col in group_df.columns
                if col not in (*group_names, date_column)
                and group_df[col].nunique(dropna=False) == 1
            ]
            if constant_cols:
                padded[constant_cols] = padded[constant_cols].ffill()

        padded_frames.append(padded[group_df.columns])

    if not padded_frames:
        return df.head(0)

    result = pd.concat(padded_frames, axis=0, ignore_index=True)
    return result


def _pad_by_time_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    *,
    date_column: str,
    freq: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
    fillna: Optional[Union[int, float]],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    if date_column not in frame.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    freq_polars = pandas_to_polars_frequency(freq)
    dtype_date = frame.schema[date_column]

    def _cast_range(series: pl.Series) -> pl.Series:
        if dtype_date == pl.Date:
            return series.cast(pl.Date)
        if isinstance(dtype_date, pl.datatypes.Datetime):
            return series.cast(
                pl.Datetime(time_unit=dtype_date.time_unit, time_zone=dtype_date.time_zone)
            )
        return series

    def _build_date_df(start_value, end_value) -> pl.DataFrame:
        if start_value is None or end_value is None:
            raise ValueError("Unable to determine start or end date for padding.")
        start_ts = _convert_to_datetime(start_value)
        end_ts = _convert_to_datetime(end_value)
        if start_ts > end_ts:
            raise ValueError("Start date cannot be after end date for padding.")
        range_series = _cached_polars_date_range(
            start_ts.isoformat(),
            end_ts.isoformat(),
            freq_polars,
        ).clone()
        range_series = _cast_range(range_series)
        return pl.DataFrame({date_column: range_series})

    ordered_cols = list(frame.columns)

    if resolved_groups:
        partitions = frame.partition_by(resolved_groups, maintain_order=True)
        results: List[pl.DataFrame] = []
        for part in partitions:
            part_sorted = part.sort(date_column)
            group_start = start_date if start_date is not None else part_sorted.select(
                pl.col(date_column).min()
            ).to_series().item()
            group_end = end_date if end_date is not None else part_sorted.select(
                pl.col(date_column).max()
            ).to_series().item()

            date_df = _build_date_df(group_start, group_end)

            for col in resolved_groups:
                key_value = part_sorted.select(pl.col(col).first()).item()
                date_df = date_df.with_columns(pl.lit(key_value).alias(col))

            join_keys = list(resolved_groups) + [date_column]
            padded = date_df.join(part_sorted, on=join_keys, how="left").sort(join_keys)
            if fillna is not None:
                fill_cols = [col for col in padded.columns if col not in join_keys]
                padded = padded.with_columns(
                    [
                        pl.when(pl.col(col).is_null())
                        .then(pl.lit(fillna))
                        .otherwise(pl.col(col))
                        .alias(col)
                        for col in fill_cols
                    ]
                )
            else:
                constant_cols: List[str] = []
                for col_name in part_sorted.columns:
                    if col_name in join_keys or col_name == row_id_column:
                        continue
                    unique_count = (
                        part_sorted.select(pl.col(col_name).n_unique()).to_series().item()
                    )
                    if unique_count == 1:
                        constant_cols.append(col_name)
                if constant_cols:
                    padded = padded.with_columns(
                        [pl.col(col).forward_fill() for col in constant_cols]
                    )
            padded = padded.select(ordered_cols)
            results.append(padded)

        padded = pl.concat(results, how="vertical_relaxed") if results else frame.head(0)
    else:
        series_sorted = frame.sort(date_column)
        global_start = start_date if start_date is not None else series_sorted.select(
            pl.col(date_column).min()
        ).to_series().item()
        global_end = end_date if end_date is not None else series_sorted.select(
            pl.col(date_column).max()
        ).to_series().item()

        date_df = _build_date_df(global_start, global_end)
        padded = date_df.join(series_sorted, on=[date_column], how="left").sort(date_column)

        if fillna is not None:
            fill_cols = [col for col in padded.columns if col not in {date_column, row_id_column}]
            padded = padded.with_columns(
                [
                    pl.when(pl.col(col).is_null())
                    .then(pl.lit(fillna))
                    .otherwise(pl.col(col))
                    .alias(col)
                    for col in fill_cols
                ]
            )
        else:
            constant_cols: List[str] = []
            for col_name in series_sorted.columns:
                if col_name == date_column or col_name == row_id_column:
                    continue
                unique_count = (
                    series_sorted.select(pl.col(col_name).n_unique()).to_series().item()
                )
                if unique_count == 1:
                    constant_cols.append(col_name)
            if constant_cols:
                padded = padded.with_columns(
                    [pl.col(col).forward_fill() for col in constant_cols]
                )

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
    fillna: Optional[Union[int, float]],
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
            value_cols = [col for col in merged.columns if col not in group_cols + [date_column]]
            if fillna is not None:
                for col in value_cols:
                    merged[col] = merged[col].fillna(fillna)
            else:
                constant_cols = [
                    col
                    for col in group_df.columns
                    if col not in group_cols + [date_column]
                    and group_df[col].nunique(dropna=False) == 1
                ]
                for col in constant_cols:
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
    value_cols = [col for col in merged.columns if col != date_column]
    if fillna is not None:
        for col in value_cols:
            merged[col] = merged[col].fillna(fillna)
    else:
        const_cols = [
            col
            for col in merged.columns
            if col != date_column and merged[col].nunique(dropna=False) == 1
        ]
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

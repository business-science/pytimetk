import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union, Optional, Sequence, List

from pytimetk.core.frequency import get_frequency
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column

from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads

from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.dataframe_ops import (
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    conversion_to_pandas,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
)
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.datetime_helpers import parse_human_duration, normalize_frequency_alias
from pytimetk.utils.ray_helpers import run_ray_tasks


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
        "Column selectors currently require pandas or polars data for `future_frame`."
    )


def _normalize_frequency_spec(
    freq: Optional[Union[str, pd.DateOffset]]
) -> Optional[pd.DateOffset]:
    if freq is None:
        return None
    if isinstance(freq, str):
        freq = normalize_frequency_alias(freq)
    if isinstance(freq, pd.DateOffset):
        return freq
    try:
        return pd.tseries.frequencies.to_offset(freq)
    except Exception:
        duration = parse_human_duration(freq)
        if isinstance(duration, pd.DateOffset):
            return duration
        return pd.tseries.frequencies.to_offset(pd.to_timedelta(duration))


def _frequency_to_str(freq: Optional[pd.DateOffset]) -> Optional[str]:
    if freq is None:
        return None
    return freq.freqstr



try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore


@pf.register_groupby_method
@pf.register_dataframe_method
def future_frame(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: Union[str, ColumnSelector],
    length_out: int,
    freq: Optional[str] = None,
    force_regular: bool = False,
    bind_data: bool = True,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    engine: str = "pandas",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Extend a DataFrame or GroupBy object with future dates.

    The `future_frame` function extends a given DataFrame or GroupBy object with
    future dates based on a specified length, optionally binding the original data.


    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The `data` parameter is the input DataFrame or grouped object that you want to
        extend with future dates.
    date_column : str or ColumnSelector
        Column containing the timestamps that anchor the extension.
    freq : str, optional
        Frequency for generated dates. When ``None`` the cadence is inferred
        from the observed series (respecting ``force_regular``). Accepts pandas
        aliases (e.g., ``"MS"``, ``"H"``) or human-friendly durations like
        ``"2 weeks"``.
    length_out : int
        The `length_out` parameter specifies the number of future dates to be
        added to the DataFrame.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether
        the frequency of the future dates should be forced to be regular. If
        `force_regular` is set to `True`, the frequency of the future dates will
        be forced to be regular. If `force_regular` is set to `False`, the
        frequency of the future dates will be inferred from the input data (e.g.
        business calendars might be used). The default value is `False`.
    bind_data : bool, optional
        The `bind_data` parameter is a boolean flag that determines whether the
        extended data should be concatenated with the original data or returned
        separately. If `bind_data` is set to `True`, the extended data will be
        concatenated with the original data using `pd.concat`. If `bind_data` is
        set to `False`, the extended data will be returned separately. The
        default value is `True`.
    threads : int
        The `threads` parameter specifies the number of threads to use for
        parallel processing. If `threads` is set to `None`, it will use all
        available processors. If `threads` is set to `-1`, it will use all
        available processors as well.
    show_progress : bool, optional
        A boolean parameter that determines whether to display progress using tqdm.
        If set to True, progress will be displayed. If set to False, progress
        will not be displayed.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : {"pandas", "polars", "cudf", "auto"}, optional
        The `engine` parameter specifies the engine to use for computation.
        ``"pandas"`` (default) performs the computation using pandas. ``"polars"``
        converts the result to a polars DataFrame on return. ``"auto"`` infers the
        engine from the input data.

    Returns
    -------
    DataFrame
        An extended DataFrame with future dates. The concrete type matches the engine
        used to process the data.

    Notes
    -----

    ## Performance

    This function uses a number of techniques to speed up computation for large
    datasets with many time series groups:

    - We vectorize where possible and use parallel processing to speed up.
    - The `threads` parameter controls the number of Ray workers used for
      parallel processing (Ray initializes automatically when `threads != 1`).

        - Set threads = -1 to use all available processors.
        - Set threads = 1 to disable parallel processing.


    See Also
    --------
    make_future_timeseries: Generate future dates for a time series.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_hourly', parse_dates = ['date'])
    df

    # Example 1 - Extend the data for a single time series group by 12 hours
    extended_df = (
        df
            .query('id == "H10"')
            .future_frame(
                date_column = 'date',
                length_out  = 12
            )
    )
    extended_df
    ```

    ```{python}
    # Example 2 - Extend the data for each group by 12 hours
    extended_df = (
        df
            .groupby('id', sort = False) # Use sort = False to preserve the original order of the data
            .future_frame(
                date_column = 'date',
                length_out  = 12,
                threads     = 1 # Use 2 threads for parallel processing
            )
    )
    extended_df
    ```

    ```{python}
    # Example 3 - Same as above, but just return the extended data with bind_data=False
    extended_df = (
        df
            .groupby('id', sort = False)
            .future_frame(
                date_column = 'date',
                length_out  = 12,
                bind_data   = False # Returns just future data
            )
    )
    extended_df
    ```

    ```{python}
    # Example 4 - Working with irregular dates: Business Days (Stocks Data)

    import pytimetk as tk
    import pandas as pd

    # Stock data
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
    df

    # Allow irregular future dates (i.e. business days)
    extended_df = (
        df
            .groupby('symbol', sort = False)
            .future_frame(
                date_column = 'date',
                length_out  = 12,
                force_regular = False, # Allow irregular future dates (i.e. business days)),
                bind_data   = True,
                threads     = 1
            )
    )
    extended_df
    ```

    ```{python}
    # Force regular: Include Weekends
    extended_df = (
        df
            .groupby('symbol', sort = False)
            .future_frame(
                date_column = 'date',
                length_out  = 12,
                force_regular = True, # Force regular future dates (i.e. include weekends)),
                bind_data   = True
            )
    )
    extended_df
    ```

    ```{python}
    # Polars DataFrame using the tk accessor
    import pandas as pd
    import polars as pl


    sample = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods=4, freq="D"),
            "value": [1, 2, 3, 4],
        }
    )

    pl_df = pl.from_pandas(sample)

    pl_df.tk.future_frame(
        date_column='date',
        length_out=2,
    )
    ```
    """

    # Common checks
    check_dataframe_or_groupby(data)

    selector_frame = _resolve_selector_frame(data)
    if isinstance(date_column, str):
        resolved_date_column = date_column
    else:
        resolved = resolve_column_selection(
            selector_frame, date_column, allow_none=False, require_match=True
        )
        if len(resolved) != 1:
            raise ValueError(
                f"`date_column` selector must resolve to exactly one column (resolved={resolved})."
            )
        resolved_date_column = resolved[0]

    date_column = resolved_date_column
    check_date_column(selector_frame, date_column)

    freq_offset = _normalize_frequency_spec(freq)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "pandas":
        conversion = convert_to_engine(data, "pandas")
        prepared = conversion.data
        result = _future_frame_pandas(
            data=prepared,
            date_column=date_column,
            length_out=length_out,
            freq=freq_offset,
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress,
            reduce_memory=reduce_memory,
        )
        return restore_output_type(result, conversion)

    if engine_resolved == "polars":
        conversion = convert_to_engine(data, "polars")
        prepared = conversion.data
        result_polars = _future_frame_polars(
            prepared,
            date_column=date_column,
            length_out=length_out,
            freq=freq_offset,
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress,
            row_id_column=conversion.row_id_column,
            group_columns=conversion.group_columns,
        )
        return restore_output_type(result_polars, conversion)

    if engine_resolved == "cudf":
        if cudf is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "cudf is required for engine='cudf', but it is not installed."
            )
        conversion = convert_to_engine(data, "cudf")
        pandas_prepared = conversion_to_pandas(conversion)
        result_pd = _future_frame_pandas(
            data=pandas_prepared,
            date_column=date_column,
            length_out=length_out,
            freq=freq_offset,
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress,
            reduce_memory=reduce_memory,
        )
        result_cudf = cudf.from_pandas(result_pd)
        return restore_output_type(result_cudf, conversion)

    raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")


def _future_frame_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    length_out: int,
    freq: Optional[Union[str, pd.DateOffset]] = None,
    force_regular: bool = False,
    bind_data: bool = True,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
) -> pd.DataFrame:
    working = data
    if reduce_memory:
        working = reduce_memory_usage(working)

    if isinstance(working, pd.DataFrame):
        df = working.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        freq_resolved = freq
        if freq_resolved is None:
            freq_resolved = _normalize_frequency_spec(
                get_frequency(
                    df[date_column].sort_values(), force_regular=force_regular
                )
            )

        future_index = _generate_future_index(
            df[date_column].iloc[-1], freq_resolved, length_out
        )
        new_rows = pd.DataFrame({date_column: future_index})

        if bind_data:
            extended_df = pd.concat([df, new_rows], axis=0, ignore_index=True)
        else:
            extended_df = new_rows

        constant_cols = [
            col
            for col in extended_df.columns
            if col != date_column and extended_df[col].nunique(dropna=False) == 1
        ]
        if constant_cols:
            extended_df[constant_cols] = extended_df[constant_cols].ffill()

        result = extended_df

    # If the data is grouped
    elif isinstance(working, pd.core.groupby.generic.DataFrameGroupBy):
        grouped = working
        group_names = grouped.grouper.names

        # If freq is None, infer the frequency from the first series in the data
        freq_local = freq
        if freq_local is None:
            if len(grouped) == 0:
                raise ValueError(
                    "Cannot infer frequency from an empty grouped object. "
                    "Provide `freq` explicitly."
                )
            label_of_first_group = next(iter(grouped.groups.keys()))
            first_group = grouped.get_group(label_of_first_group)
            freq_local = _normalize_frequency_spec(
                get_frequency(
                    pd.to_datetime(first_group[date_column]).sort_values(),
                    force_regular=force_regular,
                )
            )

        last_dates_df = grouped.agg({date_column: "max"}).reset_index()
        last_dates_df[date_column] = pd.to_datetime(last_dates_df[date_column])

        # Use parallel processing if threads is greater than 1
        if threads != 1:
            threads_resolved = get_threads(threads)

            chunk_size = max(int(len(last_dates_df) / threads_resolved), 10)
            chunk_size = max(chunk_size, 1)
            subsets = [
                last_dates_df.iloc[i : i + chunk_size]
                for i in range(0, len(last_dates_df), chunk_size)
            ]

            args_list = [
                (subset, date_column, group_names, length_out, freq_local)
                for subset in subsets
            ]
            ray_results = run_ray_tasks(
                _process_future_frame_subset,
                args_list,
                num_cpus=threads_resolved,
                desc="Future framing...",
                show_progress=show_progress,
            )
            future_dates_list = []
            for subset_result in ray_results:
                future_dates_list.extend(subset_result)

        # Use non-parallel processing if threads is 1
        else:
            future_dates_list = []
            for _, row in conditional_tqdm(
                last_dates_df.iterrows(),
                total=len(last_dates_df),
                display=show_progress,
                desc="Future framing...",
            ):
                future_dates_subset = _process_future_frame_rows(
                    row, date_column, group_names, length_out, freq_local
                )
                future_dates_list.append(future_dates_subset)

        if future_dates_list:
            future_dates_df = (
                pd.concat(future_dates_list, axis=0)
                .reset_index(drop=True)
            )
        else:
            future_dates_df = pd.DataFrame(
                columns=list(group_names) + [date_column]
            )

        if bind_data:
            grouped_df = resolve_pandas_groupby_frame(grouped)
            extended_df = pd.concat([grouped_df, future_dates_df], axis=0).reset_index(
                drop=True
            )
        else:
            extended_df = future_dates_df

        result = extended_df
    else:
        raise TypeError("Unsupported data type for future_frame().")

    if reduce_memory:
        result = reduce_memory_usage(result)

    return result


# --------------------------------------------------------------------------- #
# Polars helper                                                               #
# --------------------------------------------------------------------------- #


def _future_frame_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    length_out: int,
    freq: Optional[Union[str, pd.DateOffset]],
    force_regular: bool,
    bind_data: bool,
    threads: int,
    show_progress: bool,
    row_id_column: Optional[str],
    group_columns: Optional[Sequence[str]],
) -> pl.DataFrame:
    if length_out < 0:
        raise ValueError("`length_out` must be non-negative.")

    freq = _normalize_frequency_spec(freq)

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    if date_column not in frame.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    dtype_date = frame.schema[date_column]
    ordered_cols = list(frame.columns)
    other_columns = [
        col
        for col in frame.columns
        if col != date_column and (row_id_column is None or col != row_id_column)
    ]

    # Prepare row id counter when the conversion inserted synthetic identifiers
    row_id_counter: Optional[int] = None
    if row_id_column and row_id_column in frame.columns:
        try:
            max_row_id = frame.select(pl.col(row_id_column).max()).item()
            row_id_counter = 0 if max_row_id is None else int(max_row_id) + 1
        except Exception:
            row_id_counter = None

    def _cast_series(series: pl.Series) -> pl.Series:
        if dtype_date == pl.Date:
            return series.cast(pl.Date)
        if isinstance(dtype_date, pl.datatypes.Datetime):
            return series.cast(
                pl.Datetime(time_unit=dtype_date.time_unit, time_zone=dtype_date.time_zone)
            )
        return series

    if resolved_groups:
        partitions = frame.partition_by(resolved_groups, maintain_order=True)
        if not partitions:
            return frame

        freq_local = freq
        if freq_local is None:
            sample_dates = (
                partitions[0]
                .select(pl.col(date_column))
                .to_series()
                .to_pandas()
                .sort_values()
            )
            freq_local = _normalize_frequency_spec(
                get_frequency(sample_dates, force_regular=force_regular)
            )

        results: List[pl.DataFrame] = []
        iterator = conditional_tqdm(
            partitions,
            total=len(partitions),
            display=show_progress,
            desc="Future framing...",
        )
        for part in iterator:
            part_sorted = part.sort(date_column)
            last_value = (
                part_sorted.select(pl.col(date_column).max()).to_series().item()
            )
            future_index = _generate_future_index(last_value, freq_local, length_out)
            if len(future_index) == 0:
                result_part = part_sorted if bind_data else part_sorted.head(0)
                results.append(result_part)
                continue

            future_series = _cast_series(pl.Series(future_index))

            new_rows_dict = {date_column: future_series}
            for col in resolved_groups:
                key_value = part_sorted.select(pl.col(col).first()).item()
                new_rows_dict[col] = pl.Series(
                    name=col,
                    values=[key_value] * len(future_series),
                    dtype=part_sorted.schema[col],
                )
            for col in other_columns:
                if col in resolved_groups:
                    continue
                new_rows_dict[col] = pl.Series(
                    name=col,
                    values=[None] * len(future_series),
                    dtype=part_sorted.schema[col],
                )
            if row_id_column and row_id_column in part_sorted.columns:
                new_rows_dict[row_id_column] = pl.Series(
                    name=row_id_column,
                    values=range(
                        row_id_counter or 0,
                        (row_id_counter or 0) + len(future_series),
                    ),
                    dtype=part_sorted.schema.get(row_id_column, pl.Int64),
                )
                if row_id_counter is not None:
                    row_id_counter += len(future_series)

            new_rows = pl.DataFrame(new_rows_dict)
            new_rows = new_rows.select(ordered_cols)

            if bind_data:
                combined = pl.concat(
                    [part_sorted, new_rows],
                    how="vertical_relaxed",
                ).sort(resolved_groups + [date_column])
                results.append(combined)
            else:
                results.append(new_rows.sort(resolved_groups + [date_column]))

        result = pl.concat(results, how="vertical_relaxed")
    else:
        frame_sorted = frame.sort(date_column)
        freq_resolved = freq
        if freq_resolved is None:
            sample_dates = (
                frame_sorted.select(pl.col(date_column)).to_series().to_pandas().sort_values()
            )
            freq_resolved = _normalize_frequency_spec(
                get_frequency(sample_dates, force_regular=force_regular)
            )
        last_value = frame_sorted.select(pl.col(date_column).max()).to_series().item()
        future_index = _generate_future_index(last_value, freq_resolved, length_out)

        if len(future_index) == 0:
            result = frame_sorted if bind_data else frame_sorted.head(0)
        else:
            future_series = _cast_series(pl.Series(future_index))
            new_rows_dict = {date_column: future_series}
            for col in other_columns:
                new_rows_dict[col] = pl.Series(
                    name=col,
                    values=[None] * len(future_series),
                    dtype=frame_sorted.schema[col],
                )
            if row_id_column and row_id_column in frame_sorted.columns:
                new_rows_dict[row_id_column] = pl.Series(
                    name=row_id_column,
                    values=range(
                        row_id_counter or 0,
                        (row_id_counter or 0) + len(future_series),
                    ),
                    dtype=frame_sorted.schema.get(row_id_column, pl.Int64),
                )
                if row_id_counter is not None:
                    row_id_counter += len(future_series)

            new_rows = pl.DataFrame(new_rows_dict)
            new_rows = new_rows.select(ordered_cols)
            if bind_data:
                result = pl.concat(
                    [frame_sorted, new_rows],
                    how="vertical_relaxed",
                ).sort(date_column)
            else:
                result = new_rows.sort(date_column)

            if bind_data:
                constant_cols: List[str] = []
                for col in other_columns:
                    unique = frame_sorted.select(pl.col(col).n_unique()).to_series().item()
                    if unique == 1:
                        constant_cols.append(col)
                if constant_cols:
                    result = result.with_columns(
                        [pl.col(col).forward_fill() for col in constant_cols]
                    )

    return result


# UTILITIES ------------------------------------------------------------------


def _process_future_frame_subset(
    subset, date_column, group_names, length_out, freq
):
    future_dates_list = []
    for _, row in subset.iterrows():
        future_dates = _generate_future_index(row[date_column], freq, length_out)
        if len(future_dates) == 0:
            continue

        future_dates_df = pd.DataFrame({date_column: future_dates})
        for group_name in group_names:
            future_dates_df[group_name] = row[group_name]

        future_dates_list.append(future_dates_df)
    return future_dates_list


def _process_future_frame_rows(
    row, date_column, group_names, length_out, freq
):
    future_dates = _generate_future_index(row[date_column], freq, length_out)

    future_dates_df = pd.DataFrame({date_column: future_dates})
    for group_name in group_names:
        future_dates_df[group_name] = row[group_name]

    return future_dates_df


def _generate_future_index(
    anchor_value, freq: Optional[pd.DateOffset], length_out: int
) -> pd.DatetimeIndex:
    if length_out <= 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if freq is None:
        raise ValueError(
            "Unable to determine frequency for future_frame. Provide `freq` explicitly."
        )
    anchor = pd.Timestamp(anchor_value)
    future = pd.date_range(start=anchor, periods=length_out + 1, freq=freq)
    return future[1:]

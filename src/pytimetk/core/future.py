import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union, Optional

from pytimetk.core.frequency import get_frequency
from pytimetk.core.make_future_timeseries import make_future_timeseries

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column

from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads

from concurrent.futures import ProcessPoolExecutor

from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.dataframe_ops import (
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    conversion_to_pandas,
    resolve_pandas_groupby_frame,
)

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
    date_column: str,
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
    date_column : str
        The `date_column` parameter is a string that specifies the name of the
        column in the DataFrame that contains the dates. This column will be
        used to generate future dates.
    freq : str, optional
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
    - The `threads` parameter controls the number of threads to use for parallel
      processing.

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
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "pandas":
        conversion = convert_to_engine(data, "pandas")
        prepared = conversion.data
        result = _future_frame_pandas(
            data=prepared,
            date_column=date_column,
            length_out=length_out,
            freq=freq,
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress,
            reduce_memory=reduce_memory,
        )
        return restore_output_type(result, conversion)

    if engine_resolved == "polars":
        conversion = convert_to_engine(data, "polars")
        pandas_prepared = conversion_to_pandas(conversion)
        result_pd = _future_frame_pandas(
            data=pandas_prepared,
            date_column=date_column,
            length_out=length_out,
            freq=freq,
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress,
            reduce_memory=reduce_memory,
        )
        return pl.from_pandas(result_pd)

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
            freq=freq,
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
    freq: Optional[str] = None,
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
        ts_series = df[date_column]

        new_dates = make_future_timeseries(
            idx=ts_series, length_out=length_out, freq=freq, force_regular=force_regular
        )

        new_rows = pd.DataFrame({date_column: new_dates})

        if bind_data:
            extended_df = pd.concat([df, new_rows], axis=0, ignore_index=True)
        else:
            extended_df = new_rows

        col_name_candidates = extended_df.columns[extended_df.nunique() == 1]
        col_name = col_name_candidates[0] if not col_name_candidates.empty else None

        if col_name is not None:
            extended_df = extended_df.assign(
                **{col_name: extended_df[col_name].ffill()}
            )

        result = extended_df

    # If the data is grouped
    elif isinstance(working, pd.core.groupby.generic.DataFrameGroupBy):
        grouped = working
        group_names = grouped.grouper.names

        # If freq is None, infer the frequency from the first series in the data
        freq_local = freq
        if freq_local is None:
            label_of_first_group = list(grouped.groups.keys())[0]

            first_group = grouped.get_group(label_of_first_group)

            freq_local = get_frequency(
                first_group[date_column].sort_values(), force_regular=force_regular
            )

        last_dates_df = grouped.agg({date_column: "max"}).reset_index()

        # Use parallel processing if threads is greater than 1
        if threads != 1:
            threads_resolved = get_threads(threads)

            chunk_size = int(len(last_dates_df) / threads_resolved)
            subsets = [
                last_dates_df.iloc[i : i + chunk_size]
                for i in range(0, len(last_dates_df), chunk_size)
            ]

            future_dates_list = []
            with ProcessPoolExecutor(max_workers=threads_resolved) as executor:
                results = list(
                    conditional_tqdm(
                        executor.map(
                            _process_future_frame_subset,
                            subsets,
                            [date_column] * len(subsets),
                            [group_names] * len(subsets),
                            [length_out] * len(subsets),
                            [freq_local] * len(subsets),
                            [force_regular] * len(subsets),
                        ),
                        total=len(subsets),
                        display=show_progress,
                        desc="Future framing...",
                    )
                )
                for future_dates_subset in results:
                    future_dates_list.extend(future_dates_subset)

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
                    row, date_column, group_names, length_out, freq_local, force_regular
                )
                future_dates_list.append(future_dates_subset)

        future_dates_df = pd.concat(future_dates_list, axis=0).reset_index(drop=True)

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


# UTILITIES ------------------------------------------------------------------


def _process_future_frame_subset(
    subset, date_column, group_names, length_out, freq, force_regular
):
    future_dates_list = []
    for _, row in subset.iterrows():
        future_dates = make_future_timeseries(
            idx=pd.Series(row[date_column]),
            length_out=length_out,
            freq=freq,
            force_regular=force_regular,
        )

        future_dates_df = pd.DataFrame({date_column: future_dates})
        for group_name in group_names:
            future_dates_df[group_name] = row[group_name]

        future_dates_list.append(future_dates_df)
    return future_dates_list


def _process_future_frame_rows(
    row, date_column, group_names, length_out, freq, force_regular
):
    future_dates = make_future_timeseries(
        idx=pd.Series(row[date_column]),
        length_out=length_out,
        freq=freq,
        force_regular=force_regular,
    )

    future_dates_df = pd.DataFrame({date_column: future_dates})
    for group_name in group_names:
        future_dates_df[group_name] = row[group_name]

    return future_dates_df

import pandas as pd
import pandas_flavor as pf
import numpy as np
import polars as pl

from typing import List, Sequence, Union

from pytimetk.core.frequency import get_frequency_summary

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_series_or_datetime,
)

from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    restore_output_type,
)
from pytimetk.utils.parallel_helpers import parallel_apply, get_threads, progress_apply


@pf.register_groupby_method
@pf.register_dataframe_method
def ts_summary(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    threads=1,
    show_progress=True,
    engine: str = "pandas",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Computes summary statistics for a time series data, either for the entire
    dataset or grouped by a specific column.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Tabular time series data. Grouped inputs are processed per group
        before the summary statistics are appended. Accepts both pandas and
        polars objects.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the
        column in the DataFrame that contains the dates. This column will be
        used to compute summary statistics for the time series data.
    engine : str, optional
        Execution engine. ``"pandas"`` (default) uses pandas for the summary
        calculations. When "polars", the function operates on polars frames.

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library.
        This can be faster than using "pandas" for large datasets.

    Returns
    -------
    DataFrame
        Summary statistics for the time series. The concrete return type matches
        the engine used for computation.
        summary includes the following statistics:
        - If grouped data is provided, the returned data will contain the
          grouping columns first.
        - `date_n`: The number of observations in the time series.
        - `date_tz`: The time zone of the time series.
        - `date_start`: The first date in the time series.
        - `date_end`: The last date in the time series.
        - `freq_inferred_unit`: The inferred frequency of the time series from
                               `pandas`.
        - `freq_median_timedelta`: The median time difference between
                                   consecutive observations in the time series.
        - `freq_median_scale`: The median time difference between consecutive
                               observations in the time series, scaled to a
                              common unit.
        - `freq_median_unit`: The unit of the median time difference between
                              consecutive observations in the time series.
        - `diff_min`: The minimum time difference between consecutive
                      observations in the time series as a timedelta.
        - `diff_q25`: The 25th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_median`: The median time difference between consecutive
                         observations in the time series as a timedelta.
        - `diff_mean`: The mean time difference between consecutive observations
                       in the time series as a timedelta.
        - `diff_q75`: The 75th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_max`: The maximum time difference between consecutive
                      observations in the time series as a timedelta.
        - `diff_min_seconds`: The minimum time difference between consecutive
                              observations in the time series in seconds.
        - `diff_q25_seconds`: The 25th percentile of the time difference between
                              consecutive observations in the time series in
                              seconds.
        - `diff_median_seconds`: The median time difference between consecutive
                                 observations in the time series in seconds.
        - `diff_mean_seconds`: The mean time difference between consecutive
                               observations in the time series in seconds.
        - `diff_q75_seconds`: The 75th percentile of the time difference between
                              consecutive observations in the time series in seconds.
        - `diff_max_seconds`: The maximum time difference between consecutive
                              observations in the time series in seconds.

    Notes
    -----
    ## Performance

    This function uses parallel processing to speed up computation for large
    datasets with many time series groups:

    Parallel processing has overhead and may not be faster on small datasets.

    To use parallel processing, set `threads = -1` to use all available processors.


    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd

    dates = pd.to_datetime(["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10"])
    df = pd.DataFrame(dates, columns = ["date"])

    df.ts_summary(date_column = 'date')
    ```

    ```{python}
    # Grouped ts_summary
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])

    df.groupby('symbol').ts_summary(date_column = 'date')
    ```

    ```{python}
    # Parallelized grouped ts_summary
    (
        df
            .groupby('symbol')
            .ts_summary(
                date_column = 'date',
                threads = 2,
                show_progress = True
            )
    )
    ```

    ```{python}
    # Polars DataFrame using the tk accessor
    import polars as pl


    pl_df = pl.from_pandas(df)

    pl_df.tk.ts_summary(date_column='date')
    ```
    """

    if engine not in ["pandas", "polars"]:
        raise ValueError(
            f"Supported engines are 'pandas' or 'polars'. Found {engine}. Please select an authorized engine."
        )

    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared = conversion.data

    if engine_resolved == "pandas":
        result = _ts_summary_pandas_dispatch(
            prepared,
            date_column=date_column,
            threads=threads,
            show_progress=show_progress,
        )
    else:
        result = _ts_summary_polars_dispatch(
            prepared,
            date_column=date_column,
            group_columns=list(conversion.group_columns or []),
        )

    if engine_resolved == "polars" and conversion.original_kind in (
        "pandas_df",
        "pandas_groupby",
    ):
        conversion.pandas_index = None

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored

    return restored


def _ts_summary(group: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Compute time series summary for a single group."""

    # Make sure date is sorted
    date = group.sort_values(by=date_column)[date_column]

    # Compute summary statistics
    date_summary = get_date_summary(date)
    frequency_summary = get_frequency_summary(date)
    diff_summary = get_diff_summary(date)
    diff_summary_num = get_diff_summary(date, numeric=True)

    # Combine summary statistics into a single DataFrame
    return pd.concat(
        [date_summary, frequency_summary, diff_summary, diff_summary_num], axis=1
    )


def _ts_summary_pandas_dispatch(
    prepared: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    threads: int,
    show_progress: bool,
) -> pd.DataFrame:
    if isinstance(prepared, pd.DataFrame):
        return _ts_summary(prepared, date_column)

    group_names = prepared.grouper.names
    threads_resolved = get_threads(threads)

    if threads_resolved == 1:
        result = progress_apply(
            prepared,
            func=_ts_summary,
            date_column=date_column,
            show_progress=show_progress,
            desc="TS Summarizing...",
        )
    else:
        result = parallel_apply(
            prepared,
            func=_ts_summary,
            date_column=date_column,
            threads=threads_resolved,
            show_progress=show_progress,
            desc="TS Summarizing...",
        )

    return result.reset_index(level=group_names)


def _ts_summary_polars_dispatch(
    prepared: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    group_columns: Sequence[str],
) -> pl.DataFrame:
    if isinstance(prepared, pl.DataFrame):
        return _ts_summary_polars(prepared, date_column)

    frame = prepared.df
    partitions = frame.partition_by(group_columns, maintain_order=True, as_dict=True)

    summaries: List[pl.DataFrame] = []
    for key, subgroup in partitions.items():
        summary = _ts_summary_polars(subgroup, date_column)
        key_tuple = key if isinstance(key, tuple) else (key,)
        for col, value in zip(group_columns, key_tuple):
            summary = summary.with_columns(pl.lit(value).alias(col))
        ordered_cols = list(group_columns) + [
            col for col in summary.columns if col not in group_columns
        ]
        summaries.append(summary.select(ordered_cols))

    if not summaries:
        return pl.DataFrame()

    return pl.concat(summaries, how="vertical")


def _ts_summary_polars(data: pl.DataFrame, date_column: str) -> pl.DataFrame:
    """Compute time series summary for a single group. Polars version."""

    frame = data.sort(date_column)
    date = frame[date_column]

    # Compute summary statistics
    date_summary = compute_date_summary_polars(date, output_type="polars")
    frequency_summary = pl.from_pandas(
        get_frequency_summary(date.to_pandas(), engine="pandas")
    )
    diff_summary = get_diff_summary_polars(date).cast(pl.Duration("ns"))
    diff_summary_num = get_diff_summary_polars(date, numeric=True).cast(pl.Float64)

    df = pl.concat(
        [date_summary, frequency_summary, diff_summary, diff_summary_num],
        how="horizontal",
    )

    if "date_tz" in df.columns:
        df = df.with_columns(pl.col("date_tz").cast(pl.Utf8()))

    return df


def get_diff_summary(idx: Union[pd.Series, pd.DatetimeIndex], numeric: bool = False):
    """
    Calculates summary statistics of the time differences between consecutive values in a datetime index.

    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter can be either a pandas Series or a pandas
        DateTimeIndex. It represents the index values for which you want to
        calculate the difference summary.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that indicates whether the
        input index should be treated as numeric or not.

        - If `numeric` is set to `True`, the index values are converted to
          integers representing the number of seconds since the Unix epoch
          (January 1, 1970).

        - If `numeric` is set to `False`, the index values are treated as
          datetime values. The default value of `numeric` is `False`.

    Returns
    -------
    pd.DataFrame
        The function `get_diff_summary` returns a pandas DataFrame containing
        summary statistics including:

        If `numeric` is set to `False`, the column names are:
        - `diff_min`: The minimum time difference between consecutive
                      observations in the time series as a timedelta.
        - `diff_q25`: The 25th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_median`: The median time difference between consecutive
                         observations in the time series as a timedelta.
        - `diff_mean`: The mean time difference between consecutive observations
                       in the time series as a timedelta.
        - `diff_q75`: The 75th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_max`: The maximum time difference between consecutive
                      observations in the time series as a timedelta.

        If `numeric` is set to `True`, the column names are:
        - `diff_min_seconds`: The minimum time difference between consecutive
                              observations in the time series in seconds.
        - `diff_q25_seconds`: The 25th percentile of the time difference between
                              consecutive observations in the time series in
                              seconds.
        - `diff_median_seconds`: The median time difference between consecutive
                                 observations in the time series in seconds.
        - `diff_mean_seconds`: The mean time difference between consecutive
                               observations in the time series in seconds.
        - `diff_q75_seconds`: The 75th percentile of the time difference between
                              consecutive observations in the time series in
                              seconds.
        - `diff_max_seconds`: The maximum time difference between consecutive
                              observations in the time series in seconds.

    """

    # common checks
    check_series_or_datetime(idx)

    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    if numeric:
        idx = idx.astype(np.int64) // 10**9

    date_diff = idx.diff()

    _diff_min = date_diff.min()
    _diff_q25 = date_diff.quantile(0.25)
    _diff_median = date_diff.median()
    _diff_mean = date_diff.mean()
    _diff_q75 = date_diff.quantile(0.75)
    _diff_max = date_diff.max()

    ret = pd.DataFrame(
        {
            "diff_min": [_diff_min],
            "diff_q25": [_diff_q25],
            "diff_median": [_diff_median],
            "diff_mean": [_diff_mean],
            "diff_q75": [_diff_q75],
            "diff_max": [_diff_max],
        }
    )

    if numeric:
        ret.columns = [
            "diff_min_seconds",
            "diff_q25_seconds",
            "diff_median_seconds",
            "diff_mean_seconds",
            "diff_q75_seconds",
            "diff_max_seconds",
        ]

    return ret


def get_diff_summary_polars(idx: pl.Series, numeric: bool = False):
    """
    Calculates summary statistics of the time differences between consecutive values in a datetime index.

    Parameters
    ----------
    idx : pl.Series
        The `idx` parameter can be either a pandas Series or a pandas
        DateTimeIndex. It represents the index values for which you want to
        calculate the difference summary.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that indicates whether the
        input index should be treated as numeric or not.

        - If `numeric` is set to `True`, the index values are converted to
          integers representing the number of seconds since the Unix epoch
          (January 1, 1970).

        - If `numeric` is set to `False`, the index values are treated as
          datetime values. The default value of `numeric` is `False`.

    Returns
    -------
    pl.DataFrame
        The function `get_diff_summary` returns a polars DataFrame containing
        summary statistics including:

        If `numeric` is set to `False`, the column names are:
        - `diff_min`: The minimum time difference between consecutive
                      observations in the time series as a timedelta.
        - `diff_q25`: The 25th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_median`: The median time difference between consecutive
                         observations in the time series as a timedelta.
        - `diff_mean`: The mean time difference between consecutive observations
                       in the time series as a timedelta.
        - `diff_q75`: The 75th percentile of the time difference between
                      consecutive observations in the time series as a timedelta.
        - `diff_max`: The maximum time difference between consecutive
                      observations in the time series as a timedelta.

        If `numeric` is set to `True`, the column names are:
        - `diff_min_seconds`: The minimum time difference between consecutive
                              observations in the time series in seconds.
        - `diff_q25_seconds`: The 25th percentile of the time difference between
                              consecutive observations in the time series in
                              seconds.
        - `diff_median_seconds`: The median time difference between consecutive
                                 observations in the time series in seconds.
        - `diff_mean_seconds`: The mean time difference between consecutive
                               observations in the time series in seconds.
        - `diff_q75_seconds`: The 75th percentile of the time difference between
                              consecutive observations in the time series in
                              seconds.
        - `diff_max_seconds`: The maximum time difference between consecutive
                              observations in the time series in seconds.

    """

    # common checks
    if not isinstance(idx, pl.Series):
        raise TypeError("Expected pl.Series, got {}.".format(type(idx)))

    keys = ["diff_min", "diff_q25", "diff_median", "diff_mean", "diff_q75", "diff_max"]
    if numeric:
        keys = map(lambda s: s + "_seconds", keys)
        idx = idx.dt.epoch(time_unit="s")

    date_diff = idx.diff()

    values = [
        date_diff.min(),
        date_diff.quantile(0.25, interpolation="linear"),
        date_diff.median(),
        date_diff.mean(),
        date_diff.quantile(0.75, interpolation="linear"),
        date_diff.max(),
    ]

    return pl.DataFrame(dict(zip(keys, values)))


def get_date_summary(
    idx: Union[pd.Series, pd.DatetimeIndex], engine: str = "pandas"
) -> pd.DataFrame:
    """
    Returns a summary of the date-related information, including the number of
    dates, the time zone, the start date, and the end date.

    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The parameter `idx` can be either a pandas Series or a pandas
        DateTimeIndex. It represents the dates or timestamps for which we want
        to generate a summary.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        generating a date summary. It can be either "pandas" or "polars".
        Default is "pandas".

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        - `date_n`: The number of dates in the index.
        - `date_tz`: The time zone of the dates in the index.
        - `date_start`: The first date in the index.
        - `date_end`: The last date in the index.

    Notes
    -----
    - When using the 'polars' engine, timezone information is derived from the
      pandas input before conversion, as Polars does not natively preserve it.
    """
    if not isinstance(idx, (pd.Series, pd.DatetimeIndex)):
        raise TypeError(
            f"Input must be of type pd.Series or pd.DatetimeIndex. Got {type(idx)}"
        )

    # Convert to Series if DatetimeIndex and extract timezone
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    if engine == "pandas":
        return compute_date_summary_pandas(idx)
    elif engine == "polars":
        # Extract timezone from pandas before conversion
        tz = idx.dt.tz
        pl_idx = (
            pl.Series("idx", idx.values)
            if isinstance(idx, pd.Series)
            else pl.from_pandas(idx)
        )
        return compute_date_summary_polars(pl_idx)  # No tz parameter needed
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


def compute_date_summary_pandas(idx: pd.Series) -> pd.DataFrame:
    """[Unchanged docstring]"""
    _n = len(idx)
    _tz = idx.dt.tz
    _date_start = idx.min()
    _date_end = idx.max()

    return pd.DataFrame(
        {
            "date_n": [_n],
            "date_tz": [_tz],
            "date_start": [_date_start],
            "date_end": [_date_end],
        }
    )


def compute_date_summary_polars(
    idx: pl.Series, output_type="pandas"
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Returns a summary of the date-related information, including the number of
    dates, the time zone, the start date, and the end date.

    Parameters
    ----------
    idx : pl.Series
        A Polars Series containing the dates or timestamps for which we want
        to generate a summary.
    output_type : str, optional
        The format of the output, either 'pandas' (default) or 'polars'.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        A DataFrame with the following columns:
        - `date_n`: The number of dates in the index.
        - `date_tz`: The time zone of the dates (derived from the original pandas input).
        - `date_start`: The first date in the index.
        - `date_end`: The last date in the index.
    """
    if output_type not in ["pandas", "polars"]:
        raise TypeError(f"Output type can only be pandas or polars. Got {output_type}")

    # Attempt to infer timezone from dtype if available, otherwise assume None
    tz = None
    if idx.dtype == pl.Datetime and hasattr(idx.dtype, "time_zone"):
        tz = idx.dtype.time_zone  # Polars 0.20+ supports time_zone in dtype

    return (
        pd.DataFrame(
            {
                "date_n": [len(idx)],
                "date_tz": [tz],
                "date_start": [idx.min()],
                "date_end": [idx.max()],
            }
        )
        if output_type == "pandas"
        else pl.DataFrame(
            {
                "date_n": pl.Series([len(idx)]),
                "date_tz": pl.Series([tz]),
                "date_start": pl.Series([idx.min()], dtype=pl.Datetime("ns")),
                "date_end": pl.Series([idx.max()], dtype=pl.Datetime("ns")),
            }
        )
    )

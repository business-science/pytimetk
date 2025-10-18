# Imports
import pandas as pd
import polars as pl
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
)
from pytimetk.utils.datetime_helpers import parse_end_date
from pytimetk.utils.dataframe_ops import (
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    resolve_pandas_groupby_frame,
)

try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore


# Function ----
@pf.register_groupby_method
@pf.register_dataframe_method
def filter_by_time(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    start_date: str = "start",
    end_date: str = "end",
    engine: str = "pandas",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Filters a DataFrame or GroupBy object based on a specified date range.

    This function filters data in a pandas DataFrame or a pandas GroupBy object
    by a given date range. It supports various date formats and can handle both
    DataFrame and GroupBy objects.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The data to be filtered. Supports both pandas and polars DataFrames / GroupBy
        objects. Grouped inputs are processed per group before the final result is returned.
    date_column : str
        The name of the column in `data` that contains date information.
        This column is used for filtering the data based on the date range.
    start_date : str
        The start date of the filtering range. The format of the date can be
        YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DD HH, YYYY-MM-DD HH:SS, or YYYY-MM-DD HH:MM:SS.
        Default: 'start', which will filter from the earliest date in the data.
    end_date : str
        The end date of the filtering range. It supports the same formats as
        `start_date`.
        Default: 'end', which will filter until the latest date in the data.
    engine : str, default = 'pandas'
        Computation engine. Use ``'pandas'``, ``'polars'``, or ``'cudf'``. The special value ``'auto'``
        infers the engine from the input data.

    Returns
    -------
    DataFrame
        Data containing rows within the specified date range. The concrete type matches the
        engine used.

    Raises
    ------
    ValueError
        If the provided date strings do not match any of the supported formats.

    Notes
    -----
    - The function uses pd.to_datetime to convert the start date
      (e.g. start_date = "2014" becomes "2014-01-01").
    - The function internally uses the `parse_end_date` function to convert the
      end dates (e.g. end_date = "2014" becomes "2014-12-31").


    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import datetime

    m4_daily_df = tk.datasets.load_dataset('m4_daily', parse_dates = ['date'])

    ```

    ```{python}
    # Example 1 - Filter by date

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014-07-03',
        end_date    = '2014-07-10'
    )

    df_filtered

    ```

    ```{python}
    # Example 2 - Filter by month.
    # Note: This will filter by the first day of the month.

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014-07',
        end_date    = '2014-09'
    )

    df_filtered

    ```

    ```{python}
    # Example 3 - Filter by year.
    # Note: This will filter by the first day of the year.

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014',
        end_date    = '2014'
    )

    df_filtered

    ```

    ```{python}
    # Example 4 - Filter by day/hour/minute/second
    # Here we'll use an hourly dataset, however this will also work for minute/second data

    # Load data and format date column appropriately
    m4_hourly_df = tk.datasets.load_dataset('m4_hourly', parse_dates = ['date'])

    df_filtered = tk.filter_by_time(
        data        = m4_hourly_df,
        date_column = "date",
        start_date  = '2015-07-01 12:00:00',
        end_date    = '2015-07-01 20:00:00'
    )

    df_filtered
    ```

    ```{python}
    # Example 5 - Combine year/month/day/hour/minute/second filters
    df_filtered = tk.filter_by_time(
        data        = m4_hourly_df,
        date_column = "date",
        start_date  = '2015-07-01',
        end_date    = '2015-07-29'
    )

    df_filtered

    ```

    ```{python}
    # Example 7 - Filter using the polars engine and tk accessor
    import polars as pl


    pl_df = pl.from_pandas(m4_daily_df)

    df_filtered = (
        pl_df
            .tk.filter_by_time(
                date_column = 'date',
                start_date  = '2014-07-03',
                end_date    = '2014-07-10'
            )
    )

    df_filtered

    ```

    ```{python}
    # Example 6 - Filter a GroupBy object

    df_filtered = (
        m4_hourly_df
            .groupby('id')
            .filter_by_time(
                date_column = "date",
                start_date  = '2015-07-01 12:00:00',
                end_date    = '2015-07-01 20:00:00'
            )
    )

    df_filtered
    ```

    """
    # Checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "cudf" and cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required for engine='cudf', but it is not installed."
        )

    conversion = convert_to_engine(data, engine_resolved)
    prepared = conversion.data

    if engine_resolved == "pandas":
        result = _filter_by_time_pandas(
            prepared,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
        )
    elif engine_resolved == "polars":
        result = _filter_by_time_polars(
            prepared,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
        )
    elif engine_resolved == "cudf":
        result = _filter_by_time_cudf(
            prepared,
            date_column=date_column,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")

    if engine_resolved == "polars" and conversion.original_kind in (
        "pandas_df",
        "pandas_groupby",
    ):
        conversion.pandas_index = None

    return restore_output_type(result, conversion)


# Monkey Patch the Method to Pandas Grouby Objects
def _filter_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    start_date: str,
    end_date: str,
):
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = resolve_pandas_groupby_frame(data)

    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Handle start/end dates and parsing
    if start_date == "start":
        start_date = df[date_column].min()
    if end_date == "end":
        end_date = df[date_column].max()

    if isinstance(start_date, str):
        start_date_parsed = pd.to_datetime(start_date)
    else:
        start_date_parsed = start_date

    if isinstance(end_date, str):
        end_date_parsed = parse_end_date(end_date)
    else:
        end_date_parsed = end_date

    # If the original index has a timezone, apply it to the future dates
    if df[date_column].dt.tz is not None:
        start_date_parsed = start_date_parsed.tz_localize(df[date_column].dt.tz)
        end_date_parsed = end_date_parsed.tz_localize(df[date_column].dt.tz)

    # Filter
    filtered_df = df[
        (df[date_column] >= start_date_parsed) & (df[date_column] <= end_date_parsed)
    ]

    # Return
    return filtered_df


def _filter_by_time_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    if date_column not in frame.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    frame = frame.with_columns(pl.col(date_column).cast(pl.Datetime("ns")))

    if start_date == "start":
        start_value = frame.select(pl.col(date_column).min()).item()
    else:
        start_value = pd.to_datetime(start_date)

    if end_date == "end":
        end_value = frame.select(pl.col(date_column).max()).item()
    else:
        end_value = parse_end_date(end_date)

    start_value = pd.Timestamp(start_value).to_pydatetime()
    end_value = pd.Timestamp(end_value).to_pydatetime()

    filtered = frame.filter(
        (pl.col(date_column) >= pl.lit(start_value))
        & (pl.col(date_column) <= pl.lit(end_value))
    )

    return filtered


def _filter_by_time_cudf(
    data: Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"],
    date_column: str,
    start_date: str,
    end_date: str,
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf filter_by_time backend.")

    if hasattr(data, "obj"):
        df = data.obj.copy(deep=True)
    else:
        df = data.copy(deep=True)

    if date_column not in df.columns:
        raise KeyError(f"{date_column} not found in DataFrame")

    df[date_column] = cudf.to_datetime(df[date_column])

    if start_date == "start":
        start_value = df[date_column].min()
        start_value = start_value.to_pandas() if hasattr(start_value, "to_pandas") else start_value
    else:
        start_value = pd.to_datetime(start_date)

    if end_date == "end":
        end_value = df[date_column].max()
        end_value = end_value.to_pandas() if hasattr(end_value, "to_pandas") else end_value
    else:
        end_value = parse_end_date(end_date)

    mask = (df[date_column] >= pd.to_datetime(start_value)) & (
        df[date_column] <= pd.to_datetime(end_value)
    )
    filtered_df = df.loc[mask]

    return filtered_df


# Utilities ----

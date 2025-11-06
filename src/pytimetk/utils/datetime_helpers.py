import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf

from datetime import datetime, timedelta
from dateutil import parser
from typing import Iterable, List, Sequence, Union
import re

from pytimetk.utils.checks import check_series_or_datetime
from pytimetk.utils.polars_helpers import pandas_to_polars_frequency
from pytimetk.utils.string_helpers import parse_freq_str

_HUMAN_DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>[-+]?\d*\.?\d+)?\s*(?P<unit>[a-zA-Z]+)\s*$"
)

_HUMAN_DURATION_UNITS = {
    # Timedelta-compatible units
    "s": "seconds",
    "sec": "seconds",
    "secs": "seconds",
    "second": "seconds",
    "seconds": "seconds",
    "ms": "milliseconds",
    "millisecond": "milliseconds",
    "milliseconds": "milliseconds",
    "us": "microseconds",
    "microsecond": "microseconds",
    "microseconds": "microseconds",
    "ns": "nanoseconds",
    "nanosecond": "nanoseconds",
    "nanoseconds": "nanoseconds",
    "m": "minutes",
    "min": "minutes",
    "mins": "minutes",
    "minute": "minutes",
    "minutes": "minutes",
    "h": "hours",
    "hr": "hours",
    "hrs": "hours",
    "hour": "hours",
    "hours": "hours",
    "d": "days",
    "day": "days",
    "days": "days",
    "w": "weeks",
    "wk": "weeks",
    "wks": "weeks",
    "week": "weeks",
    "weeks": "weeks",
    # DateOffset-based units
    "month": "months",
    "months": "months",
    "mon": "months",
    "mons": "months",
    "mo": "months",
    "q": "quarters",
    "quarter": "quarters",
    "quarters": "quarters",
    "y": "years",
    "yr": "years",
    "yrs": "years",
    "year": "years",
    "years": "years",
}


def parse_human_duration(
    value: Union[str, int, float, pd.Timedelta, np.timedelta64, timedelta, pd.DateOffset],
) -> Union[pd.Timedelta, pd.DateOffset]:
    """
    Convert human-friendly duration input into a pandas Timedelta or DateOffset.

    Parameters
    ----------
    value : str or numeric or timedelta-like
        Supported examples include: ``"30 minutes"``, ``"2 hours"``, ``"3 months"``,
        ``"1 year"``, ``pd.Timedelta("7D")``, ``datetime.timedelta(days=2)``.

    Returns
    -------
    Union[pd.Timedelta, pd.DateOffset]
        Returns a Timedelta for fixed-width units (seconds through weeks) and a
        DateOffset for calendar-aware units (months, quarters, years).

    Raises
    ------
    ValueError
        If the input cannot be parsed or represents an unsupported unit.

    Examples
    --------
    ```python
    import pytimetk as tk

    tk.parse_human_duration("45 minutes")
    tk.parse_human_duration("3 months")
    ```
    """
    if isinstance(value, pd.DateOffset):
        return value

    if isinstance(value, (pd.Timedelta, np.timedelta64, timedelta)):
        return pd.to_timedelta(value)

    if isinstance(value, (int, float)):
        # Interpret bare numerics as seconds to avoid silent mistakes.
        return pd.to_timedelta(value, unit="seconds")

    if not isinstance(value, str):
        raise ValueError(f"Unsupported duration type: {type(value)}")

    text = value.strip()
    if not text:
        raise ValueError("Duration string is empty.")

    match = _HUMAN_DURATION_PATTERN.match(text.lower())
    if not match:
        raise ValueError(
            f"Could not parse duration string '{value}'. Expected formats like '3 days' or '15 minutes'."
        )

    quantity_str = match.group("value")
    unit_key = match.group("unit")

    if unit_key not in _HUMAN_DURATION_UNITS:
        raise ValueError(
            f"Unit '{unit_key}' is not supported. Supported units include seconds, minutes, hours, "
            "days, weeks, months, quarters, and years."
        )

    canonical_unit = _HUMAN_DURATION_UNITS[unit_key]
    quantity = float(quantity_str) if quantity_str is not None else 1.0

    if canonical_unit in {"months", "quarters", "years"}:
        if not float(quantity).is_integer():
            raise ValueError(
                f"Duration '{value}' requires an integer quantity for calendar units."
            )
        quantity_int = int(quantity)
        if canonical_unit == "months":
            return pd.DateOffset(months=quantity_int)
        if canonical_unit == "quarters":
            return pd.DateOffset(months=quantity_int * 3)
        return pd.DateOffset(years=quantity_int)

    # Timedelta-based units
    return pd.to_timedelta(quantity, unit=canonical_unit)


def resolve_lag_sequence(
    lags: Union[str, int, Sequence[int], np.ndarray, range, slice],
    index: Union[pd.Series, pd.DatetimeIndex, Sequence],
    clamp: bool = True,
) -> np.ndarray:
    """
    Normalise lag specifications into a sorted numpy array of non-negative integers.

    Parameters
    ----------
    lags : str, int, Sequence[int], range, slice
        - String durations (e.g. ``"30 days"``, ``"3 months"``) are converted to the
          number of observation lags implied by ``index``.
        - Integers produce a ``range(0, lags)`` style sequence (inclusive).
        - Sequences/ranges/slices are materialised and sorted.
    index : array-like
        A date/time index used to translate duration strings into row counts. The
        index is sorted internally.
    clamp : bool, optional
        If ``True`` (default), lags greater than ``len(index) - 1`` are clipped.

    Returns
    -------
    np.ndarray
        Sorted array of lag integers starting at zero.

    Raises
    ------
    ValueError
        When the input cannot be interpreted or the index is empty.

    Examples
    --------
    ```python
    import numpy as np
    import pandas as pd
    import pytimetk as tk

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    tk.resolve_lag_sequence("3 days", idx)
    tk.resolve_lag_sequence([0, 2, 4], idx)
    ```
    """
    if index is None:
        raise ValueError("`index` must be provided to resolve lag sequences.")

    if isinstance(index, pd.Series):
        idx = pd.to_datetime(index.to_numpy(copy=False))
    elif isinstance(index, pd.DatetimeIndex):
        idx = pd.to_datetime(index)
    else:
        idx = pd.to_datetime(np.asarray(index))

    if idx.size == 0:
        raise ValueError("Cannot resolve lags from an empty index.")

    idx = np.sort(idx)
    max_possible_lag = max(idx.size - 1, 0)

    def _clamp(sequence: Iterable[int]) -> np.ndarray:
        arr = np.array([int(x) for x in sequence if int(x) >= 0], dtype=int)
        if arr.size == 0:
            return np.array([0], dtype=int)
        if clamp:
            arr = arr[arr <= max_possible_lag]
        return np.unique(arr)

    if isinstance(lags, str):
        duration = parse_human_duration(lags)
        start = idx[0]
        if isinstance(duration, pd.Timedelta):
            end = start + duration
        else:
            end = (pd.Timestamp(start) + duration).to_pydatetime()
        # Include rows up to the computed end point
        mask = idx <= np.datetime64(end)
        reach = int(mask.sum()) - 1
        reach = max(reach, 0)
        return _clamp(range(0, reach + 1))

    if isinstance(lags, slice):
        start = lags.start or 0
        stop = lags.stop if lags.stop is not None else max_possible_lag
        step = lags.step or 1
        return _clamp(range(int(start), int(stop) + 1, int(step)))

    if isinstance(lags, range):
        return _clamp(lags)

    if isinstance(lags, np.ndarray):
        return _clamp(lags.tolist())

    if isinstance(lags, Sequence) and not isinstance(lags, (str, bytes)):
        return _clamp(lags)

    if isinstance(lags, (int, np.integer)):
        upper = int(lags)
        if upper < 0:
            raise ValueError("`lags` must be non-negative.")
        return _clamp(range(0, upper + 1))

    raise ValueError(f"Unsupported lag specification: {lags!r}")


@pf.register_series_method
def floor_date(
    idx: Union[pd.Series, pd.DatetimeIndex],
    unit: str = "D",
    engine: str = "pandas",
) -> pd.Series:
    """
    Robust date flooring.

    The `floor_date` function takes a pandas Series of dates and returns a new Series
    with the dates rounded down to the specified unit. It's more robust than the
    pandas `floor` function, which does weird things with irregular frequencies
    like Month which are actually regular.

    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter is a pandas Series or pandas DatetimeIndex object that
        contains datetime values. It represents the dates that you want to round down.
    unit : str, optional
        The `unit` parameter in the `floor_date` function is a string that specifies
        the time unit to which the dates in the `idx` series should be rounded down.
        It has a default value of "D", which stands for day. Other possible values
        for the `unit` parameter could be.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for calculating
        the floor datetime. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library for
          calculating the floor datetime. This can be faster than using "pandas" for
          large datasets.

    Returns
    -------
    pd.Series
        The `floor_date` function returns a pandas Series object containing datetime64[ns] values.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd

    dates = pd.date_range("2020-01-01", "2020-01-10", freq="1H")
    dates
    ```

    ```{python}
    # Pandas fails to floor Month
    # dates.floor("M") # ValueError: <MonthEnd> is a non-fixed frequency

    # floor_date works as expected
    tk.floor_date(dates, unit="M", engine='pandas')
    ```
    """

    # Common checks
    check_series_or_datetime(idx)

    if engine == "pandas":
        return _floor_date_pandas(idx, unit)
    elif engine == "polars":
        return _floor_date_polars(idx, unit)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


def _floor_date_pandas(
    idx: Union[pd.Series, pd.DatetimeIndex],
    unit: str = "D",
) -> pd.Series:
    """
    Robust date flooring.
    """

    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    nm = idx.name

    # Fix for pandas bug: When unit is greater than the length of the index, the floor function returns the first day of the year
    days_in_sequence = idx.iloc[-1] - idx.iloc[0]
    days_in_unit = freq_to_timedelta(unit)

    if days_in_unit > days_in_sequence:
        # fill series with first value
        idx.iloc[1:] = idx.iloc[0]

    # Parse frequency string
    quantity, unit_2 = parse_freq_str(unit)
    quantity = int(quantity) if quantity else 1

    if quantity == 1:
        # DOESN'T WORK WITH MULTIPLES OF FREQUENCIES "2M" "3D" "4H"
        date = pd.Series(pd.PeriodIndex(idx.values, freq=unit).to_timestamp(), name=nm)
    else:
        if unit_2 == "M":
            # Floor to N-month intervals using vectorized operations
            floored_months = ((idx.dt.month - 1) // quantity) * quantity + 1
            date = pd.to_datetime(
                {"year": idx.dt.year, "month": floored_months, "day": 1}
            )
            date = pd.Series(date, name=nm)
        elif unit_2 == "Q":
            # Floor to N-quarter intervals using vectorized operations
            floored_months = ((idx.dt.month - 1) // (3 * quantity)) * (3 * quantity) + 1
            date = pd.to_datetime(
                {"year": idx.dt.year, "month": floored_months, "day": 1}
            )
            date = pd.Series(date, name=nm)
        elif unit_2 == "Y":
            # Floor to N-year intervals using vectorized operations
            floored_years = (idx.dt.year // quantity) * quantity
            date = pd.to_datetime({"year": floored_years, "month": 1, "day": 1})
            date = pd.Series(date, name=nm)
        else:
            # DOESN'T WORK WITH IRREGULAR FREQUENCIES (MONTH, QUARTER, YEAR, ETC)
            date = idx.dt.floor(unit)

    return date


def _floor_date_polars(
    idx: Union[pd.Series, pd.DatetimeIndex],
    unit: str = "D",
) -> pd.Series:
    """
    Robust date flooring.
    """

    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pl.Series(idx).alias("idx")

    date = (idx.dt.truncate(every=pandas_to_polars_frequency(unit))).to_pandas()

    return date


@pf.register_series_method
def ceil_date(
    idx: Union[pd.Series, pd.DatetimeIndex],
    unit: str = "D",
) -> pd.Series:
    """
    Robust date ceiling.

    The `ceil_date` function takes a pandas Series of dates and returns a new
    Series with the dates rounded up to the next specified unit. It's more
    robust than the pandas `ceil` function, which does weird things with
    irregular frequencies like Month which are actually regular.

    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter is a pandas Series or pandas DatetimeIndex object
        that contains datetime values. It represents the dates that you want to
        round down.
    unit : str, optional
        The `unit` parameter in the `ceil_date` function is a string that
        specifies the time unit to which the dates in the `idx` series should be
        rounded down. It has a default value of "D", which stands for day. Other
        possible values for the `unit` parameter could be

    Returns
    -------
    pd.Series
        The `ceil_date` function returns a pandas Series object containing
        datetime64[ns] values.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd

    dates = pd.date_range("2020-01-01", "2020-01-10", freq="1H")
    dates
    ```

    ```{python}
    # Pandas ceil fails on month
    # dates.ceil("M") # ValueError: <MonthEnd> is a non-fixed frequency

    # Works on Month
    tk.ceil_date(dates, unit="M")
    ```
    """
    # Common checks
    check_series_or_datetime(idx)

    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    # Convert to period
    date = floor_date(idx, unit=unit) + freq_to_dateoffset(unit)

    return date


def freq_to_dateoffset(freq_str):
    # Adjusted regex to account for potential absence of numeric part
    quantity, unit = parse_freq_str(freq_str)

    # Assume quantity of 1 if it's not explicitly provided
    quantity = int(quantity) if quantity else 1

    # Define a dictionary to map frequency units to DateOffset functions
    unit_to_offset = {
        "D": pd.DateOffset(days=quantity),
        "H": pd.DateOffset(hours=quantity),
        "T": pd.DateOffset(minutes=quantity),
        "min": pd.DateOffset(minutes=quantity),
        "S": pd.DateOffset(seconds=quantity),
        "L": pd.DateOffset(milliseconds=quantity),
        "U": pd.DateOffset(microseconds=quantity),
        "N": pd.DateOffset(nanoseconds=quantity),
        "Y": pd.DateOffset(years=quantity),
        "A": pd.DateOffset(years=quantity),
        "AS": pd.DateOffset(years=quantity),
        "YS": pd.DateOffset(years=quantity),
        "W": pd.DateOffset(weeks=quantity),
        "Q": pd.DateOffset(months=quantity * 3),
        "QS": pd.DateOffset(months=quantity * 3),
        "M": pd.DateOffset(months=quantity),
        "MS": pd.DateOffset(months=quantity),
    }

    # Get the DateOffset function for the given unit
    offset = unit_to_offset.get(unit)

    if offset is None:
        raise ValueError(f"Unsupported frequency unit: {unit}")

    return offset


def freq_to_timedelta(freq_str):
    unit_mapping = {
        "D": "days",
        "H": "hours",
        "T": "minutes",
        "min": "minutes",
        "S": "seconds",
        "L": "milliseconds",
        "U": "milliseconds",
        "N": "nanoseconds",
        "Y": "years",
        "A": "years",
        "AS": "years",
        "YS": "years",
        "W": "weeks",
        "Q": "quarters",
        "QS": "quarters",
        "M": "months",
        "MS": "months",
    }

    quantity, unit = parse_freq_str(freq_str)
    quantity = int(quantity) if quantity else 1

    if unit in unit_mapping:
        if unit_mapping[unit] == "years":
            return pd.Timedelta(days=quantity * 365.25)
        elif unit_mapping[unit] == "quarters":
            return pd.Timedelta(days=quantity * 3 * 30.44)
        elif unit_mapping[unit] == "months":
            return pd.Timedelta(days=quantity * 30.44)
        else:
            return pd.Timedelta(**{unit_mapping[unit]: quantity})
    else:
        raise ValueError(f"Unsupported frequency unit: {unit}")


def parse_end_date(date_str):
    date = pd.to_datetime(date_str)

    # Determine the granularity of the input and apply the appropriate offset
    if len(date_str) == 4:  # Year granularity
        end_date = date + pd.offsets.YearEnd()
    elif len(date_str) == 7:  # Month granularity
        end_date = date + pd.offsets.MonthEnd()
    elif len(date_str) == 10:  # Day granularity
        end_date = pd.Timestamp(date_str).replace(hour=23, minute=59, second=59)
    elif len(date_str) == 13:  # Hour granularity
        end_date = pd.Timestamp(date_str).replace(minute=59, second=59)
    elif len(date_str) == 16:  # Minute granularity
        end_date = pd.Timestamp(date_str).replace(second=59)
    elif len(date_str) == 19:  # Second Granularity
        end_date = date
    else:  # Day or finer granularity
        # Assuming you want to roll up to the end of the day
        end_date = pd.Timestamp(date_str).replace(hour=23, minute=59, second=59)

    return end_date


@pf.register_series_method
def week_of_month(
    idx: Union[pd.Series, pd.DatetimeIndex],
    engine: str = "pandas",
) -> pd.Series:
    """
    The "week_of_month" function calculates the week number of a given date
    within its month.

    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The parameter "idx" is a pandas Series object that represents a specific
        date for which you want to determine the week of the month.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        calculating the week of the month. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
        for calculating the week of the month. This can be faster than using
        "pandas" for large datasets.


    Returns
    -------
    pd.Series
        The week of the month for a given date.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd

    dates = pd.date_range("2020-01-01", "2020-02-28", freq="1D")
    dates
    ```

    ```{python}
    # Works on DateTimeIndex
    tk.week_of_month(dates, engine='pandas')
    ```

    ```{python}
    # Works on DateTimeIndex
    tk.week_of_month(dates, engine='polars')
    ```

    ```{python}
    # Works on Pandas Series
    dates.to_series().week_of_month()
    ```

    ```{python}
    # Works on Pandas Series
    dates.to_series().week_of_month(engine='polars')
    ```

    """
    # Common checks
    check_series_or_datetime(idx)

    if engine == "pandas":
        return _week_of_month_pandas(idx)
    elif engine == "polars":
        return _week_of_month_polars(idx)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


def _week_of_month_pandas(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    """
    The "week_of_month" function calculates the week number of a given date within its month.
    """

    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")

    ret = (idx.dt.day - 1) // 7 + 1

    ret = pd.Series(ret, name="week_of_month", index=idx.index)

    return ret


def _week_of_month_polars(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    """
    The "week_of_month" function calculates the week number of a given date within its month.
    """

    if isinstance(idx, pd.DatetimeIndex):
        idx = pl.Series(idx).alias("idx")
    elif isinstance(idx, pd.Series):
        idx = pl.Series(idx).alias("idx")
    else:
        raise ValueError("Input 'idx' must be a Pandas DatetimeIndex or Series.")

    ret = ((idx.dt.day() - 1) // 7 + 1).alias("week_of_month").to_pandas()

    return ret


@pf.register_series_method
def is_holiday(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.Series],
    country_name: str = "UnitedStates",
    country: str = None,
    engine: str = "pandas",
) -> pd.Series:
    """
    Check if a given list of dates are holidays for a specified country.

    Note: This function requires the `holidays` package to be installed.

    Parameters
    ----------
    idx : Union[str, datetime, List[Union[str, datetime]], pd.Series]
        The dates to check for holiday status.
    country_name (str, optional):
        The name of the country for which to check the holiday status. Defaults
        to 'UnitedStates' if not specified.
    country (str, optional):
        An alternative parameter to specify the country for holiday checking,
        overriding country_name.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        generating the boolean series. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
          for generating a boolean of holidays or not holidays. This can be
          faster than using "pandas" for long series.

    Returns:
    -------
    pd.Series:
        Series containing True if the date is a holiday, False otherwise.

    Raises:
    -------
    ValueError:
        If the specified country is not found in the holidays package.

    Examples:
    --------
    ```{python}
    import polars as pl
    import pytimetk as tk

    tk.is_holiday('2023-01-01', country_name='UnitedStates')
    ```

    ```{python}
    # List of dates
    tk.is_holiday(['2023-01-01', '2023-01-02', '2023-01-03'], country_name='UnitedStates')
    ```

    ```{python}
    # Polars Series
    tk.is_holiday(pl.Series(['2023-01-01', '2023-01-02', '2023-01-03']), country_name='UnitedStates')
    ```
    """

    if engine == "pandas":
        return _is_holiday_pandas(idx, country_name, country)
    elif engine == "polars":
        return _is_holiday_polars(idx, country_name, country)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


def _is_holiday_pandas(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.DatetimeIndex, pd.Series],
    country_name: str = "UnitedStates",
    country: str = None,
) -> pd.Series:
    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError(
            "The 'holidays' package is not installed. Please install it by running 'pip install holidays'."
        )

    if country:
        country_name = (
            country  # Override the default country_name with the provided one
        )

    # Find the country module from the holidays package
    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")

    if isinstance(idx, str) or isinstance(idx, datetime):
        idx = [idx]

    idx = pd.to_datetime(idx)  # Convert all dates to pd.Timestamp if not already

    # Check each date if it's a holiday and return the results as a Series
    ret = pd.Series(
        [date in country_module(years=date.year) for date in idx], name="is_holiday"
    )

    return ret


def _is_holiday_polars(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.Series],
    country_name: str = "UnitedStates",
    country: str = None,
) -> pd.Series:
    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError(
            "The 'holidays' package is not installed. Please install it by running 'pip install holidays'."
        )

    if country:
        country_name = (
            country  # Override the default country_name with the provided one
        )

    # Convert pl.date objects if they are strings
    if isinstance(idx, str):
        start_year, start_month, start_day = map(int, idx.split("-"))
        start = pl.date(start_year, start_month, start_day)
        end = start
        end_year, end_month, end_day = map(int, idx.split("-"))

    # Convert to pl.date objects DatetimeIndex object
    if isinstance(idx, pd.core.indexes.datetimes.DatetimeIndex):
        date_range = idx

        start_date = date_range[0]
        start_year, start_month, start_day = (
            start_date.year,
            start_date.month,
            start_date.day,
        )
        start = pl.date(start_year, start_month, start_day)

        end_date = date_range[-1]
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
        end = pl.date(end_year, end_month, end_day)

    # Convert list of strings to list of dates
    if isinstance(idx, list):
        dates = []
        for date_str in idx:
            date = pd.to_datetime(date_str).date()
            dates.append(date)
        start_date = dates[0]
        start_year, start_month, start_day = (
            start_date.year,
            start_date.month,
            start_date.day,
        )
        start = pl.date(start_year, start_month, start_day)

        end_date = dates[-1]
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
        end = pl.date(end_year, end_month, end_day)

    holidays_list = list(
        holidays.country_holidays(country_name, years=[start_year, end_year])
    )
    expr = pl.date_range(start, end)
    is_holiday = expr.is_in(holidays_list)

    ret = pl.select(is_holiday).to_series().to_pandas()

    return ret


def is_datetime_string(x: Union[str, pd.Series, pd.DatetimeIndex]) -> bool:
    if isinstance(x, pd.Series):
        x = x.values[0]

    if isinstance(x, pd.DatetimeIndex):
        x = x[0]

    try:
        parser.parse(str(x))
        return True
    except ValueError:
        return False


def detect_timeseries_columns(
    data: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    df = data.head(1)

    if verbose:
        print(df)

    return df.map(is_datetime_string)


def has_timeseries_columns(data: pd.DataFrame, verbose: bool = False) -> bool:
    if verbose:
        print(detect_timeseries_columns(data).iloc[0])

    return detect_timeseries_columns(data).iloc[0].any()


def get_timeseries_colname(data: pd.DataFrame, verbose: bool = False) -> str:
    if verbose:
        print(detect_timeseries_columns(data).iloc[0])

    return detect_timeseries_columns(data).iloc[0].idxmax()

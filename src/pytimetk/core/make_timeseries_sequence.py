import pandas as pd
import polars as pl
from datetime import datetime
import pandas_flavor as pf

from typing import List, Sequence, Union

import holidays


def _coerce_to_timestamp(
    value: Union[str, datetime, pd.Timestamp, pd.DatetimeIndex],
    pick: str,
) -> pd.Timestamp:
    if isinstance(value, pd.DatetimeIndex):
        if len(value) == 0:
            raise ValueError("DatetimeIndex inputs must contain at least one value.")
        base = value[0] if pick == "start" else value[-1]
        return pd.Timestamp(base)
    return pd.Timestamp(value)


def _build_holiday_filter(
    dates: pd.DatetimeIndex,
    remove_holidays: bool,
    country: Union[str, None],
) -> Sequence[pd.Timestamp]:
    if not remove_holidays or len(dates) == 0:
        return ()

    country_key = country or "UnitedStates"
    years = sorted(set(dates.year))
    holiday_map = holidays.country_holidays(country_key, years=years)
    return pd.to_datetime(list(holiday_map.keys()))


def _make_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    allowed_weekdays: List[int],
    label: str,
    remove_holidays: bool,
    country: Union[str, None],
    engine: str,
) -> Union[pd.Series, pl.Series]:
    start_ts = _coerce_to_timestamp(start_date, "start")
    end_ts = _coerce_to_timestamp(end_date, "end")

    if start_ts > end_ts:
        raise ValueError("`start_date` must be on or before `end_date`.")

    all_days = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
    filtered = all_days[all_days.dayofweek.isin(allowed_weekdays)]

    holidays_to_remove = _build_holiday_filter(filtered, remove_holidays, country)
    if len(holidays_to_remove) > 0:
        normalized_holidays = pd.to_datetime(holidays_to_remove).normalize()
        filtered = filtered[~filtered.normalize().isin(normalized_holidays)]

    if engine == "polars":
        return pl.Series(label, filtered.to_pydatetime())
    if engine == "pandas":
        return pd.Series(filtered, name=label)
    raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


@pf.register_series_method
def make_weekday_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None,
    engine: str = "pandas",
) -> Union[pd.Series, pl.Series]:
    """
    Generate a sequence of weekday dates within a specified date range,
    optionally excluding weekends and holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    sunday_to_thursday : bool, optional
        If True, generates a sequence with Sunday to Thursday weekdays (excluding
        Friday and Saturday). If False (default), generates a sequence with
        Monday to Friday weekdays.
    remove_holidays : bool, optional
        If True, excludes holidays (based on the specified country) from the
        generated sequence.
        If False (default), includes holidays in the sequence.
    country (str, optional):
        The name of the country for which to generate holiday-specific sequences.
        Defaults to None, which uses the United States as the default country.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        generating a weekday series. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
          for generating a weekday series. This can be faster than using
          "pandas" for large datasets.

    Returns
    -------
    Series
        A Series containing the generated weekday dates. The concrete type
        matches the requested engine.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    # United States has Monday to Friday as weekdays (excluding Saturday and
    # Sunday and holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15",
                              sunday_to_thursday = False,
                              remove_holidays    = True,
                              country            = 'UnitedStates',
                              engine             = 'pandas')
    ```

    ```{python}
    # Israel has Sunday to Thursday as weekdays (excluding Friday and Saturday
    # and Israel holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15",
                              sunday_to_thursday = True,
                              remove_holidays    = True,
                              country            = 'Israel',
                              engine             = 'pandas')
    ```

    ```{python}
    # Israel has Sunday to Thursday as weekdays (excluding Friday and Saturday
    # and Israel holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15",
                              sunday_to_thursday = True,
                              remove_holidays    = True,
                              country            = 'Israel',
                              engine             = 'polars')
    ```
    """

    weekday_range = [0, 1, 2, 3, 4]
    if sunday_to_thursday:
        weekday_range = [6, 0, 1, 2, 3]

    return _make_sequence(
        start_date=start_date,
        end_date=end_date,
        allowed_weekdays=weekday_range,
        label="Weekday Dates",
        remove_holidays=remove_holidays,
        country=country,
        engine=engine,
    )


@pf.register_series_method
def make_weekend_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None,
    engine: str = "pandas",
) -> Union[pd.Series, pl.Series]:
    """
    Generate a sequence of weekend dates within a specified date range,
    optionally excluding holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    friday_saturday (bool, optional):
        If True, generates a sequence with Friday and Saturday as weekends. If
        False (default), generates a sequence with Saturday and Sunday as
        weekends.
    remove_holidays : bool, optional
        If True, excludes holidays (based on the specified country) from the
        generated sequence.
        If False (default), includes holidays in the sequence.
    country (str, optional):
        The name of the country for which to generate holiday-specific sequences.
        Defaults to None, which uses the United States as the default country.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        generating a weekend series. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library
          for generating a weekend series. This can be faster than using
          "pandas" for large datasets.

    Returns
    -------
    Series
        A Series containing the generated weekend dates. The concrete type
        matches the requested engine.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    # United States has Saturday and Sunday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31",
                             friday_saturday = False,
                             remove_holidays = True,
                             country         = 'UnitedStates',
                             engine          = 'pandas')
    ```

    ```{python}
    # Saudi Arabia has Friday and Saturday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31",
                             friday_saturday = True,
                             remove_holidays = True,
                             country         = 'SaudiArabia',
                             engine          = 'pandas')
    ```

    ```{python}
    # Saudi Arabia has Friday and Saturday as weekends (polars engine)
    tk.make_weekend_sequence("2023-01-01", "2023-01-31",
                             friday_saturday = True,
                             remove_holidays = True,
                             country         = 'SaudiArabia',
                             engine          = 'polars')
    ```
    """

    weekend_range = [5, 6]  # Saturday=5, Sunday=6
    if friday_saturday:
        weekend_range = [4, 5]

    return _make_sequence(
        start_date=start_date,
        end_date=end_date,
        allowed_weekdays=weekend_range,
        label="Weekend Dates",
        remove_holidays=remove_holidays,
        country=country,
        engine=engine,
    )

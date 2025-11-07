import pandas as pd
import pandas_flavor as pf
from typing import Union, Optional, List

from pytimetk.core.frequency import get_frequency

from pytimetk.utils.checks import check_series_or_datetime
from pytimetk.utils.datetime_helpers import normalize_frequency_alias


@pf.register_series_method
def make_future_timeseries(
    idx: Union[str, List[str], pd.Series, pd.DatetimeIndex],
    length_out: int,
    freq: Optional[Union[str, pd.DateOffset]] = None,
    force_regular: bool = False,
) -> pd.Series:
    """
    Make future dates for a time series.

    The function `make_future_timeseries` takes a pandas Series or DateTimeIndex
    and generates a future sequence of dates based on the frequency of the input
    series.

    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter is the input time series data. It can be either a
        pandas Series or a pandas DateTimeIndex. It represents the existing dates
        in the time series.
    length_out : int
        The parameter `length_out` is an integer that represents the number of
        future dates to generate for the time series.
    freq : str or pd.DateOffset, optional
        Frequency of the future dates. When ``None``, the cadence is inferred
        from the input data (respecting ``force_regular``). Accepts pandas
        offsets or human-friendly strings e.g. ``"2 weeks"``.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether
        the frequency of the future dates should be forced to be regular. If
        `force_regular` is set to `True`, the frequency of the future dates will
        be forced to be regular. If `force_regular` is set to `False`, the
        frequency of the future dates will be inferred from the input data (e.g.
        business calendars might be used). The default value is `False`.

    Returns
    -------
    pd.Series
        A pandas Series object containing future dates.

    Examples
    --------

    ```{python}
    import pandas as pd
    import pytimetk as tk

    # Works with a single date (must provide a length out and frequency if only
    # 1 date is provided)
    tk.make_future_timeseries("2011-01-01", 5, "D")
    ```

    ```{python}
    # DateTimeIndex: Generate 5 future dates (with inferred frequency)

    dates = pd.Series(pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']))

    future_dates_dt = tk.make_future_timeseries(dates, 5)
    future_dates_dt
    ```

    ```{python}
    # Series: Generate 5 future dates
    pd.Series(future_dates_dt).make_future_timeseries(5)
    ```

    ```{python}
    # Hourly Frequency: Generate 5 future dates
    timestamps = ["2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 03:00", "2023-01-01 04:00", "2023-01-01 05:00"]

    dates = pd.to_datetime(timestamps)

    tk.make_future_timeseries(dates, 5)
    ```

    ```{python}
    # Monthly Frequency: Generate 4 future dates
    dates = pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"])

    tk.make_future_timeseries(dates, 4)
    ```

    ```{python}
    # Quarterly Frequency: Generate 4 future dates
    dates = pd.to_datetime(["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"])

    tk.make_future_timeseries(dates, 4)
    ```

    ```{python}
    # Irregular Dates: Business Days
    dates = pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-06"])

    tk.get_frequency(dates)

    tk.make_future_timeseries(dates, 4)
    ```

    ```{python}
    # Irregular Dates: Business Days (Force Regular)
    tk.make_future_timeseries(dates, 4, force_regular=True)
    ```
    """
    if isinstance(idx, str):
        idx = pd.to_datetime([idx])
    elif isinstance(idx, list):
        idx = pd.to_datetime(idx)

    check_series_or_datetime(idx)

    if isinstance(idx, pd.Series):
        series = idx.copy()
    elif isinstance(idx, pd.DatetimeIndex):
        series = pd.Series(idx, name=idx.name or "idx")
    else:
        series = pd.Series(idx, name="idx")

    series = pd.to_datetime(series)

    if len(series) < 2 and freq is None:
        raise ValueError("`freq` must be provided if `idx` contains only 1 date.")

    dt_index = pd.DatetimeIndex(series.values)

    freq_resolved = freq
    if freq_resolved is None:
        freq_resolved = get_frequency(dt_index, force_regular=force_regular)

    if isinstance(freq_resolved, str):
        freq_resolved = normalize_frequency_alias(freq_resolved)

    future_dates = pd.date_range(
        start=series.iloc[-1], periods=length_out + 1, freq=freq_resolved
    )[1:]

    return pd.Series(future_dates)

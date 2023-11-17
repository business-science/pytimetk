import pandas as pd
import numpy as np
import pandas_flavor as pf
from typing import Union, Optional, List

from pytimetk.core.frequency import get_frequency

from pytimetk.utils.checks import check_series_or_datetime


@pf.register_series_method
def make_future_timeseries(
    idx: Union[str, List[str], pd.Series, pd.DatetimeIndex],
    length_out: int,
    freq: Optional[str] = None,
    force_regular: bool = False,
) -> pd.Series:
    '''
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
    freq : str, optional
        The `frequency` parameter is a string that specifies the frequency of the 
        future dates. If `frequency` is set to `None`, the frequency of the future 
        dates will be inferred from the input data (e.g. business calendars might 
        be used). The default value is `None`.
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
    '''
    # Convert idx to Pandas DateTime Index if it's a string or list of strings
    if isinstance(idx, str):
        idx = pd.to_datetime([idx])
        
    if isinstance(idx, list):
        idx = pd.to_datetime(idx)
        
    # Check if idx is a Series or DatetimeIndex
    check_series_or_datetime(idx)
        
    if len(idx) < 2:
        if freq is None:
            raise ValueError("`freq` must be provided if `idx` contains only 1 date.")
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Create a DatetimeIndex from the provided dates
    dt_index = pd.DatetimeIndex(pd.Series(idx).values)
    
    # Determine the frequency
    if freq is None:
        freq = get_frequency(dt_index, force_regular=force_regular)  
    
    # Generate the next four periods (dates)
    future_dates = pd.date_range(
        start   = dt_index[-1], 
        periods = length_out +1, 
        freq    = freq
    )[1:]  # Exclude the first date as it's already in dt_index

    # If the original index has a timezone, apply it to the future dates
    if idx.dt.tz is not None:
        future_dates = future_dates.tz_localize(idx.dt.tz)
    
    ret = pd.Series(future_dates)
    
    return ret
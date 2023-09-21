
import pandas as pd
import numpy as np
import pandas_flavor as pf

from math import ceil
from warnings import warn



@pf.register_series_method
def floor_date(
    idx: pd.Series or pd.DatetimeIndex, 
    unit: str = "D",
) -> pd.Series:
    '''The `floor_date` function takes a pandas Series of dates and returns a new Series with the dates
    rounded down to the specified unit.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter is a pandas Series or pandas DatetimeIndex object that contains datetime values. It represents the dates that you want to round down.
    unit : str, optional
        The `unit` parameter in the `floor_date` function is a string that specifies the time unit to which the dates in the `idx` series should be rounded down. It has a default value of "D", which stands for day. Other possible values for the `unit` parameter could be
    
    Returns
    -------
        The `floor_date` function returns a pandas Series object.
    
    Examples
    --------
    ```{python}
    import timetk as tk
    import pandas as pd
    
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="1H")
    dates
    ```
    
    ```{python}
    # Works on DateTimeIndex
    tk.floor_date(dates, unit="D")
    ```
    
    ```{python}
    # Works on Pandas Series
    dates.to_series().floor_date(unit="D")
    ```
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Check if idx is a Series
    if not isinstance(idx, pd.Series):
        raise TypeError('idx must be a pandas Series or DatetimeIndex object')
    
    # Convert to period
    period = idx.dt.to_period(unit)

    try:
        date = period.dt.to_timestamp()
    except:
        warn("Failed attempt to convert to pandas timestamp. Returning pandas period instead.")
        date = period

    return date

@pf.register_series_method
def week_of_month(idx: pd.Series or pd.DatetimeIndex,) -> pd.Series:
    '''The "week_of_month" function calculates the week number of a given date within its month.
    
    Parameters
    ----------
    idx : pd.Series
        The parameter "idx" is a pandas Series object that represents a specific date for which you want to determine the week of the month.
    
    Returns
    -------
        The week of the month for a given date.
    
    '''
    
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    if not isinstance(idx, pd.Series):
        raise TypeError('idx must be a pandas Series or DatetimeIndex object')
    
    ret = (idx.dt.day - 1) // 7 + 1
    
    ret = pd.Series(
        ret, 
        name="week_of_month", 
        index = idx.index
    )
    
    return ret

import pandas as pd
import numpy as np
import pandas_flavor as pf

from math import ceil
from warnings import warn



@pf.register_series_method
def floor_date(
    idx: pd.Series, 
    unit: str = "D",
) -> pd.Series:
    """
    The `floor_date` function takes a pandas Series of dates and returns a new Series with the dates
    rounded down to the specified unit.
    
    :param idx: A pandas Series containing datetime values
    :type idx: pd.Series
    :param unit: The `unit` parameter in the `floor_date` function is a string that specifies the time
    unit to which the dates in the `x` series should be rounded down. It has a default value of "D",
    which stands for day. Other possible values for the `unit` parameter could be, defaults to D
    :type unit: str (optional)
    :return: The function `floor_date` returns a pandas Series object.
    """
    
    # Check pandas series
    
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
    
    # first_day = idx.replace(day=1)
  
    # day_of_month = idx.day
  
    # if(first_day.weekday() == 6):
    #     adjusted_dom = (1 + first_day.weekday()) / 7
    # else:
    #     adjusted_dom = day_of_month + first_day.weekday()
    
    # int(ceil(adjusted_dom/7.0))
    
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
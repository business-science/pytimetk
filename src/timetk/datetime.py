
import pandas as pd
import numpy as np
import pandas_flavor as pf
import typing

from warnings import warn



@pf.register_series_method
def floor_date(
    data: pd.Series, 
    unit: str = "D",
):
    """
    Takes a date-time object and time unit, and rounds it to the first value of the specified time unit.

    Args:
        x ([pandas.Series]): A pandas series containing pandas timestamps. 
        unit (str, optional): A pandas offset frequency. Defaults to "D".

    Returns:
        [pandas.Series]: [description]
    """

    # Check pandas series
    
    period = data.dt.to_period(unit)

    try:
        date = period.dt.to_timestamp()
    except:
        warn("Failed attempt to convert to pandas timestamp. Returning pandas period instead.")
        date = period

    return date


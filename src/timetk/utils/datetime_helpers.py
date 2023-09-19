
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
    The `floor_date` function takes a pandas Series of dates and returns a new Series with the dates
    rounded down to the specified unit.
    
    :param data: A pandas Series containing datetime values
    :type data: pd.Series
    :param unit: The `unit` parameter in the `floor_date` function is a string that specifies the time
    unit to which the dates in the `data` series should be rounded down. It has a default value of "D",
    which stands for day. Other possible values for the `unit` parameter could be, defaults to D
    :type unit: str (optional)
    :return: The function `floor_date` returns a pandas Series object.
    """
    
    # Check pandas series
    
    period = data.dt.to_period(unit)

    try:
        date = period.dt.to_timestamp()
    except:
        warn("Failed attempt to convert to pandas timestamp. Returning pandas period instead.")
        date = period

    return date


# Dependencies
import pandas as pd
import numpy as np
import datetime as dt
from math import ceil
import pandas_flavor as pf

@pf.register_dataframe_method
def week_of_month(date):
   first_day = date.replace(day=1)
 
   day_of_month = date.day
 
   if(first_day.weekday() == 6):
       adjusted_dom = (1 + first_day.weekday()) / 7
   else:
       adjusted_dom = day_of_month + first_day.weekday()
 
   return int(ceil(adjusted_dom/7.0))
 
@pf.register_dataframe_method
def get_timeseries_signature_date(x):
    """
    Engineers 30 different date and time based features from a single datetime.
 
    Args:
        x ([datetime64[ns]]): A datatime64[ns] dtype column.
 
    Returns:
        DataFrame: A pandas data frame that leverages all of the currently accessible time/date components and several others derived from some date math:
            - index_num: An int64 feature that captures the entire datetime as a numeric value to the second
            - year: The year of the datetime
            - year_iso: The iso year of the datetime
            - yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
            - yearend: Logical (0,1) indicating if last day of year (defined by frequency)
            - leapyear: Logical (0,1) indicating if the date belongs to a leap year
            - half: Half year of the date: Jan-Jun = 1, July-Dec = 2
            - quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
            - quarteryear: Quarter of the date + relative year
            - quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
            - quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
            - month: The month of the datetime
            - month_lbl: The month label of the datetime
            - monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
            - monthend: Logical (0,1) indicating if last day of month (defined by frequency)
            - yweek_iso: The week ordinal of the iso year
            - yweek: The week ordinal of the year
            - mweek: The week ordinal of the month
            - wday: The number of the day of the week with Monday=1, Sunday=6
            - wday_lbl: The day of the week label
            - mday: The day of the datetime
            - qday: The days of the relative quarter
            - yday: The ordinal day of year
            - weekend: Logical (0,1) indicating if the day is a weekend 
            - hour: The hour of the datetime
            - minute: The minutes of the datetime
            - second: The seconds of the datetime
            - msecond: The microseconds of the datetime
            - nsecond: The nanoseconds of the datetime
            - am_pm: Half of the day, AM = ante meridiem,  PM = post meridiem
    """
 
    # Date-Time Index Feature
    index_num    = x.apply(lambda x: x.value)
 
    # Yearly Features
    year         = x.dt.year
    year_iso     = x.dt.isocalendar().year
    yearstart    = x.dt.is_year_start.astype('uint8')
    yearend      = x.dt.is_year_end.astype('uint8')
    leapyear     = x.dt.is_leap_year.astype('uint8')
 
    # Semesterly Features
    half         = np.where(((x.dt.quarter == 1) | ((x.dt.quarter == 2))), 1, 2)
    half         = pd.Series(half)
 
    # Quarterly Features
    quarter      = x.dt.quarter
    quarteryear  = pd.PeriodIndex(x, freq = 'Q')
    quarteryear  = pd.Series(quarteryear)
    quarterstart = x.dt.is_quarter_start.astype('uint8')
    quarterend   = x.dt.is_quarter_end.astype('uint8')
 
    # Monthly Features
    month       = x.dt.month
    month_lbl   = x.dt.month_name()
    monthstart  = x.dt.is_month_start.astype('uint8')
    monthend    = x.dt.is_month_end.astype('uint8')
 
    # Weekly Features
    yweek_iso   = x.dt.isocalendar().week
    yweek       = x.dt.week
    mweek       = x.apply(week_of_month)
 
    # Daily Features
    wday        = x.dt.dayofweek + 1
    wday_lbl    = x.dt.day_name()
    mday        = x.dt.day
    qday        = (x - pd.PeriodIndex(x, freq ='Q').start_time).dt.days + 1
    yday        = x.dt.dayofyear
    weekend     = np.where((x.dt.dayofweek <= 5), 0, 1)
    weekend     = pd.Series(weekend)
 
    # Hourly Features
    hour        = x.dt.hour
 
    # Minute Features
    minute     = x.dt.minute
 
    # Second Features
    second     = x.dt.second
 
    # Microsecond Features
    msecond    = x.dt.microsecond
 
    # Nanosecond Features
    nsecond    = x.dt.nanosecond
 
    # AM/PM
    am_pm      = np.where((x.dt.hour <= 12), 'am', 'pm')
    am_pm      = pd.Series(am_pm)
 
    # Combine Series
    df = pd.concat([index_num, year, year_iso, yearstart, yearend,
                    leapyear, half, quarter, quarteryear, quarterstart,
                    quarterend, month, month_lbl, monthstart, monthend,
                    yweek_iso, yweek, mweek, wday, wday_lbl,
                    mday, qday, yday, weekend, hour,
                    minute, second, msecond, nsecond, am_pm
                    ], axis=1)
 
    # Get Date and Time Column Names
    column_names = ['index_num', 'year', 'year_iso', 'yearstart', 'yearend',
                    'leapyear', 'half', 'quarter', 'quarteryear', 'quarterstart',
                    'quarterend', 'month', 'month_lbl', 'monthstart', 'monthend',
                    'yweek_iso', 'yweek', 'mweek', 'wday', 'wday_lbl',
                    'mday', 'qday', 'yday', 'weekend', 'hour',
                    'minute', 'second', 'msecond', 'nsecond', 'am_pm']
 
    # Give Columns Proper Names
    df.columns = column_names
 
    return df
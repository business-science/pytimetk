# Dependencies
import pandas as pd
import numpy as np
import pandas_flavor as pf

from .utils.datetime_helpers import week_of_month


 
@pf.register_series_method
def get_timeseries_signature(idx: pd.Series or pd.DatetimeIndex) -> pd.DataFrame:
    '''The function `tk_get_timeseries_signature` engineers 29 different date and time based features from
    a single datetime. 
    
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
        - am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Parameters
    ----------
    idx : pd.Series
        idx is a pandas Series object containing datetime values.
    
    Returns
    -------
        The function `tk_get_timeseries_signature` returns a pandas DataFrame that contains 29 different date and time based features derived from a single datetime column.
        
    Examples
    --------
    import pandas as pd
    import timetk as tk
    
    dates = pd.date_range(start = '2019-01', end = '2019-03', freq = 'D')
    
    tk.get_timeseries_signature(dates)
    
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Check if idx is a Series
    if not isinstance(idx, pd.Series):
        raise TypeError('idx must be a pandas Series or DatetimeIndex object')
    
 
    # Date-Time Index Feature
    index_num    = idx.astype(np.int64) // 10**9
 
    # Yearly Features
    year         = idx.dt.year
    year_iso     = idx.dt.isocalendar().year
    yearstart    = idx.dt.is_year_start.astype('uint8')
    yearend      = idx.dt.is_year_end.astype('uint8')
    leapyear     = idx.dt.is_leap_year.astype('uint8')
 
    # Semesterly Features
    half         = np.where(((idx.dt.quarter == 1) | ((idx.dt.quarter == 2))), 1, 2)
    half         = pd.Series(half, index = idx.index)
 
    # Quarterly Features
    quarter      = idx.dt.quarter
    quarteryear  = pd.PeriodIndex(idx, freq = 'Q')
    quarteryear  = pd.Series(quarteryear, index = idx.index)
    quarterstart = idx.dt.is_quarter_start.astype('uint8')
    quarterend   = idx.dt.is_quarter_end.astype('uint8')
 
    # Monthly Features
    month       = idx.dt.month
    month_lbl   = idx.dt.month_name()
    monthstart  = idx.dt.is_month_start.astype('uint8')
    monthend    = idx.dt.is_month_end.astype('uint8')
 
    # Weekly Features
    yweek       = idx.dt.isocalendar().week
    mweek       = week_of_month(idx)
 
    # Daily Features
    wday        = idx.dt.dayofweek + 1
    wday_lbl    = idx.dt.day_name()
    mday        = idx.dt.day
    qday        = (idx - pd.PeriodIndex(idx, freq ='Q').start_time).dt.days + 1
    yday        = idx.dt.dayofyear
    weekend     = np.where((idx.dt.dayofweek <= 5), 0, 1)
    weekend     = pd.Series(weekend, index = idx.index)
 
    # Hourly Features
    hour        = idx.dt.hour
 
    # Minute Features
    minute     = idx.dt.minute
 
    # Second Features
    second     = idx.dt.second
 
    # Microsecond Features
    msecond    = idx.dt.microsecond
 
    # Nanosecond Features
    nsecond    = idx.dt.nanosecond
 
    # AM/PM
    am_pm      = np.where((idx.dt.hour <= 12), 'am', 'pm')
    am_pm      = pd.Series(am_pm, index = idx.index)
 
    # Combine Series
    df = pd.concat([index_num, year, year_iso, yearstart, yearend,
                    leapyear, half, quarter, quarteryear, quarterstart,
                    quarterend, month, month_lbl, monthstart, monthend,yweek, mweek, wday, wday_lbl,
                    mday, qday, yday, weekend, hour,
                    minute, second, msecond, nsecond, am_pm
                    ], axis=1)
 
    # Get Date and Time Column Names
    column_names = ['index_num', 'year', 'year_iso', 'yearstart', 'yearend',
                    'leapyear', 'half', 'quarter', 'quarteryear', 'quarterstart',
                    'quarterend', 'month', 'month_lbl', 'monthstart', 'monthend',
                    'yweek', 'mweek', 'wday', 'wday_lbl',
                    'mday', 'qday', 'yday', 'weekend', 'hour',
                    'minute', 'second', 'msecond', 'nsecond', 'am_pm']
 
    # Give Columns Proper Names
    df.columns = column_names
 
    return df

@pf.register_dataframe_method
def augment_timeseries_signature(
    data: pd.DataFrame, 
    date_column: str,
) -> pd.DataFrame:
    '''The function `augment_timeseries_signature` takes a DataFrame and a date column as input, calculates
    the timeseries signature of the date column, and returns the original DataFrame with the timeseries
    signature appended as additional columns.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the time series data.
    date_column : str
        The `date_column` parameter is a string that represents the name of the column in the `data`
    DataFrame that contains the dates for the time series data.
    
    Returns
    -------
        a pandas DataFrame that is the concatenation of the original data DataFrame and the ts_signature_df
    DataFrame.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk
    
    ( 
        tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
            .augment_timeseries_signature(date_column = 'order_date')
    )
    ```
    
    '''
    
    idx = data[date_column]
    
    ts_signature_df = idx.get_timeseries_signature()
    
    colnames = [date_column + "_" + item for item in ts_signature_df.columns]
    
    ts_signature_df.columns = colnames
    
    ret = pd.concat([data, ts_signature_df], axis = 1)
    
    return ret
    
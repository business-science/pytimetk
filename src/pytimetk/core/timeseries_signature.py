# Dependencies
import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.datetime_helpers import week_of_month
from pytimetk.utils.checks import check_series_or_datetime, check_dataframe_or_groupby, check_date_column

@pf.register_series_method
def get_timeseries_signature(
    data: Union[pd.Series, pd.DatetimeIndex],
    engine: str = 'pandas'
    ) -> pd.DataFrame:
    """
    Convert a timestamp to a set of 29 time series features.

    The function `get_timeseries_signature` engineers **29 different date and time based features** from a single datetime index `idx`: 

    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas Series of DatetimeIndex.

    engine : str, optional
        The `engine` parameter is used to specify the engine to use for augmenting datetime features. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library for feature generation. This is generally faster than using "pandas" for large datasets. 
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    dates = pd.date_range(start = '2019-01', end = '2019-03', freq = 'D')
    
    # Makes 29 new time series features from the dates
    tk.get_timeseries_signature(dates, engine='pandas').head()
    ```
    """
    # common checks
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Check if idx is a Series
    if not isinstance(idx, pd.Series):
        raise TypeError('idx must be a pandas Series or DatetimeIndex object')
    
    if engine == 'pandas':
        return _augment_timeseries_signature_pandas(data)
    elif engine == 'polars':
        return _augment_timeseries_signature_polars(data)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

# Monkey patch the method to Pandas Series objects
pd.Series.get_timeseries_signature = get_timeseries_signature

def _get_timeseries_signature_pandas(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.DataFrame:
    """
    Convert a timestamp to a set of 29 time series features.

    The function `get_timeseries_signature` engineers **29 different date and time based features** from a single datetime index `idx`: 

    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a Pandas Series of DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    dates = pd.date_range(start = '2019-01', end = '2019-03', freq = 'D')
    
    # Makes 29 new time series features from the dates
    tk.get_timeseries_signature_pandas(dates).head()
    ```
    """
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name='idx')

    data = pd.DataFrame({'idx': idx})

    # Date-Time Index Feature
    data['idx_index_num'] = idx.astype(np.int64) // 10**9

    # Yearly Features
    data['idx_year'] = idx.dt.year
    data['idx_year_iso'] = idx.dt.isocalendar().year
    data['idx_yearstart'] = idx.dt.is_year_start.astype('uint8')
    data['idx_yearend'] = idx.dt.is_year_end.astype('uint8')
    data['idx_leapyear'] = idx.dt.is_leap_year.astype('uint8')

    # Semesterly Features
    half = np.where(((idx.dt.quarter == 1) | ((idx.dt.quarter == 2))), 1, 2)
    data['idx_half'] = pd.Series(half, index=idx.index)

    # Quarterly Features
    data['idx_quarter'] = idx.dt.quarter
    quarteryear = pd.PeriodIndex(idx, freq='Q')
    data['idx_quarteryear'] = pd.Series(quarteryear, index=idx.index)
    data['idx_quarterstart'] = idx.dt.is_quarter_start.astype('uint8')
    data['idx_quarterend'] = idx.dt.is_quarter_end.astype('uint8')

    # Monthly Features
    data['idx_month'] = idx.dt.month
    data['idx_month_lbl'] = idx.dt.month_name()
    data['idx_monthstart'] = idx.dt.is_month_start.astype('uint8')
    data['idx_monthend'] = idx.dt.is_month_end.astype('uint8')

    # Weekly Features
    data['idx_yweek'] = idx.dt.isocalendar().week
    data['idx_mweek'] = week_of_month1(idx)

    # Daily Features
    data['idx_wday'] = idx.dt.dayofweek + 1
    data['idx_wday_lbl'] = idx.dt.day_name()
    data['idx_mday'] = idx.dt.day
    data['idx_qday'] = (idx.dt.tz_localize(None) - pd.PeriodIndex(idx.dt.tz_localize(None), freq='Q').start_time).dt.days + 1
    data['idx_yday'] = idx.dt.dayofyear
    weekend = np.where((idx.dt.dayofweek <= 5), 0, 1)
    data['idx_weekend'] = pd.Series(weekend, index=idx.index)

    # Hourly Features
    data['idx_hour'] = idx.dt.hour

    # Minute Features
    data['idx_minute'] = idx.dt.minute

    # Second Features
    data['idx_second'] = idx.dt.second

    # Microsecond Features
    data['idx_msecond'] = idx.dt.microsecond

    # Nanosecond Features
    data['idx_nsecond'] = idx.dt.nanosecond

    # AM/PM
    am_pm = np.where((idx.dt.hour <= 12), 'am', 'pm')
    data['idx_am_pm'] = pd.Series(am_pm, index=idx.index)

    return data

def _get_timeseries_signature_polars(data: Union[pd.Series, pd.DatetimeIndex]) -> pl.DataFrame:
    """
    Convert a timestamp to a set of 29 time series features.

    The function `get_timeseries_signature` engineers **29 different date and time based features** from a single datetime index `idx`: 

    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a Pandas Series of DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    dates = pd.date_range(start = '2019-01', end = '2019-03', freq = 'D')
    
    # Makes 29 new time series features from the dates
    tk.get_timeseries_signature_polars(dates).head()
    ```
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
        data.columns = ["idx"]

    if isinstance(data, pd.DatetimeIndex):
        data = pd.DataFrame(data, columns=["idx"])

    # Convert to Polars DataFrame
    df_pl = pl.DataFrame(data)

    df_pl = df_pl.with_columns(
        # Date-Time Index Feature
        (pl.col("idx").cast(pl.Int64) / 1_000_000_000).alias("idx_index_num"),

        # Yearly Features
        pl.col("idx").dt.year().alias("idx_year"),
        pl.col("idx").dt.iso_year().alias("idx_year_iso"),
        (pl.when((pl.col("idx").dt.month() == 1) & (pl.col("idx").dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_yearstart")
        ),
        (pl.when((pl.col("idx").dt.month() == 12) & (pl.col("idx").dt.day() == 31))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_yearend")
        ),
        pl.col("idx").dt.is_leap_year().cast(pl.Int8).alias("idx_leapyear"),

        # Semesterly Features
        (pl.when((pl.col("idx").dt.quarter() == 1) | (pl.col("idx").dt.quarter() == 2))
            .then(1)
            .otherwise(2)
            .alias("idx_half")
        ),

        # Quarterly Features
        pl.col("idx").dt.quarter().alias("idx_quarter"),
        pl.col("idx").dt.strftime("%Y").alias("quarteryear") + "Q" + pl.col("idx").dt.quarter().cast(pl.Utf8, strict=False),

        pl.when((pl.col("idx").dt.month().is_in([1, 4, 7, 10])) & (pl.col("idx").dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_quarterstart"),
        pl.when((pl.col("idx").dt.month().is_in([3, 12])) & (pl.col("idx").dt.day() == 31)
                | (pl.col("idx").dt.month().is_in([6, 9])) & (pl.col("idx").dt.day() == 30)
                )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_quarterend"),
        pl.col("idx").dt.month().alias("idx_month"),

        # Monthly Features
        pl.col("idx").dt.strftime(format="%B").alias("idx_month_lbl"),
        (pl.when((pl.col("idx").dt.month() == 1) & (pl.col("idx").dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_monthstart")
        ),
        pl.when((pl.col("idx").dt.month().is_in([1, 3, 5, 7, 8, 10, 12]) & (pl.col("idx").dt.day() == 31))
                | (pl.col("idx").dt.month().is_in([4, 6, 9, 11]) & (pl.col("idx").dt.day() == 30)
                | (pl.col("idx").dt.is_leap_year().cast(pl.Int8) == 0) & (pl.col("idx").dt.day() == 28)
                | (pl.col("idx").dt.is_leap_year().cast(pl.Int8) == 1) & (pl.col("idx").dt.day() == 29)
                ))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("idx_monthend"),

        # Weekly Features
        pl.col("idx").dt.week().alias("idx_yweek"),
        ((pl.col("idx").dt.day() - 1) // 7 + 1).alias("idx_mweek"),
        pl.col("idx").dt.weekday().alias("idx_wday"),
        pl.col("idx").dt.strftime(format="%A").alias("idx_wday_lbl"),

        # Daily Features
        pl.col("idx").dt.day().alias("idx_mday"),
        pl.when(pl.col("idx").dt.quarter() == 1)
            .then(pl.col("idx").dt.ordinal_day())
            .when(pl.col("idx").dt.quarter() == 2)
            .then(pl.col("idx").dt.ordinal_day() - 90)
            .when(pl.col("idx").dt.quarter() == 3)
            .then(pl.col("idx").dt.ordinal_day() - 181)
            .when(pl.col("idx").dt.quarter() == 4)
            .then(pl.col("idx").dt.ordinal_day() - 273)
            .alias("idx_qday"),
        pl.col("idx").dt.ordinal_day().alias("idx_yday"),
        (pl.when((pl.col("idx").dt.weekday() <= 5))
            .then(0)
            .otherwise(1)
            .alias("idx_weekend")
        ),

        # Hourly Features
        pl.col("idx").dt.hour().alias("idx_hour"),

        # Minute Features
        pl.col("idx").dt.minute().alias("idx_minute"),

        # Second Features
        pl.col("idx").dt.second().alias("idx_second"),

        # Microsecond Features
        pl.col("idx").dt.microsecond().alias("idx_msecond"),

        # Nanosecond Features
        pl.col("idx").dt.nanosecond().alias("idx_nanosecond"),

        # AM/PM
        (pl.when((pl.col("idx").dt.hour() <= 12))
            .then("am")
            .otherwise("pm")
            .alias("idx_am_pm")
        )
    )

    return df_pl.to_pandas()

@pf.register_dataframe_method
def augment_timeseries_signature(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    engine: str = 'pandas',
) -> pd.DataFrame:
    """
    The function `augment_timeseries_signature` takes a DataFrame and a date 
    column as input and returns the original DataFrame with the **29 different date 
    and time based features** added as new columns with the feature name based on 
    the date_column.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the time series data.
    date_column : str
        The `date_column` parameter is a string that represents the name of the date column in the `data` DataFrame.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for augmenting datetime features. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library for feature generation. This is generally faster than using "pandas" for large datasets. 
    
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    # Adds 29 new time series features as columns to the original DataFrame
    ( 
        tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
            .augment_timeseries_signature(date_column='order_date', engine ='pandas')
            .head()
    )
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)
    
    if engine == 'pandas':
        return _augment_timeseries_signature_pandas(data, date_column)
    elif engine == 'polars':
        return _augment_timeseries_signature_polars(data, date_column)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_timeseries_signature = augment_timeseries_signature

def _augment_timeseries_signature_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
) -> pd.DataFrame:
    """
    The function `augment_timeseries_signature` takes a DataFrame and a date 
    column as input and returns the original DataFrame with the **29 different date 
    and time based features** added as new columns with the feature name based on 
    the date_column.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the time series data.
    date_column : str
        The `date_column` parameter is a string that represents the name of the date column in the `data` DataFrame.
    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    # Adds 29 new time series features as columns to the original DataFrame
    ( 
        tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
            .augment_timeseries_signature_pandas(date_column='order_date')
            .head()
    )
    ```
    """
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
    
    idx = data[date_column]
    
    # Date-Time Index Feature
    data[date_column + "_index_num"] = idx.astype(np.int64) // 10**9

    # Yearly Features
    data[date_column + "_year"] = idx.dt.year
    data[date_column + "_year_iso"] = idx.dt.isocalendar().year
    data[date_column + "_yearstart"] = idx.dt.is_year_start.astype('uint8')
    data[date_column + "_yearend"] = idx.dt.is_year_end.astype('uint8')
    data[date_column + "_leapyear"] = idx.dt.is_leap_year.astype('uint8')

    # Semesterly Features
    half = np.where(((idx.dt.quarter == 1) | ((idx.dt.quarter == 2))), 1, 2)
    half = pd.Series(half, index=idx.index)
    data[date_column + "_half"] = half

    # Quarterly Features
    data[date_column + "_quarter"] = idx.dt.quarter
    quarteryear = pd.PeriodIndex(idx, freq='Q')
    quarteryear = pd.Series(quarteryear, index=idx.index)
    data[date_column + "_quarteryear"] = quarteryear
    data[date_column + "_quarterstart"] = idx.dt.is_quarter_start.astype('uint8')
    data[date_column + "_quarterend"] = idx.dt.is_quarter_end.astype('uint8')

    # Monthly Features
    data[date_column + "_month"] = idx.dt.month
    data[date_column + "_month_lbl"] = idx.dt.month_name()
    data[date_column + "_monthstart"] = idx.dt.is_month_start.astype('uint8')
    data[date_column + "_monthend"] = idx.dt.is_month_end.astype('uint8')

    # Weekly Features
    data[date_column + "_yweek"] = idx.dt.isocalendar().week
    data[date_column + "_mweek"] = week_of_month(idx)

    # Daily Features
    data[date_column + "_wday"] = idx.dt.dayofweek + 1
    data[date_column + "_wday_lbl"] = idx.dt.day_name()
    data[date_column + "_mday"] = idx.dt.day
    data[date_column + "_qday"] = (idx.dt.tz_localize(None) - pd.PeriodIndex(idx.dt.tz_localize(None), freq='Q').start_time).dt.days + 1
    data[date_column + "_yday"] = idx.dt.dayofyear
    weekend = np.where((idx.dt.dayofweek <= 5), 0, 1)
    weekend = pd.Series(weekend, index=idx.index)
    data[date_column + "_weekend"] = weekend

    # Hourly Features
    data[date_column + "_hour"] = idx.dt.hour

    # Minute Features
    data[date_column + "_minute"] = idx.dt.minute

    # Second Features
    data[date_column + "_second"] = idx.dt.second

    # Microsecond Features
    data[date_column + "_msecond"] = idx.dt.microsecond

    # Nanosecond Features
    data[date_column + "_nsecond"] = idx.dt.nanosecond

    # AM/PM
    am_pm = np.where((idx.dt.hour <= 12), 'am', 'pm')
    am_pm = pd.Series(am_pm, index=idx.index)
    data[date_column + "_am_pm"] = am_pm

    return data

def _augment_timeseries_signature_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str
) -> pd.DataFrame:
    """
    The function `augment_timeseries_signature` takes a DataFrame and a date 
    column as input and returns the original DataFrame with the **29 different date 
    and time based features** added as new columns with the feature name based on 
    the date_column.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the time series data.
    date_column : str
        The `date_column` parameter is a string that represents the name of the date column in the `data` DataFrame.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with 29 datetime features added to it.

    - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
    - _year: The year of the datetime
    - _year_iso: The iso year of the datetime
    - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
    - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
    - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
    - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
    - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
    - _quarteryear: Quarter of the date + relative year
    - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
    - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
    - _month: The month of the datetime
    - _month_lbl: The month label of the datetime
    - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
    - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
    - _yweek: The week ordinal of the year
    - _mweek: The week ordinal of the month
    - _wday: The number of the day of the week with Monday=1, Sunday=6
    - _wday_lbl: The day of the week label
    - _mday: The day of the datetime
    - _qday: The days of the relative quarter
    - _yday: The ordinal day of year
    - _weekend: Logical (0,1) indicating if the day is a weekend 
    - _hour: The hour of the datetime
    - _minute: The minutes of the datetime
    - _second: The seconds of the datetime
    - _msecond: The microseconds of the datetime
    - _nsecond: The nanoseconds of the datetime
    - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    pd.set_option('display.max_columns', None)
    
    # Adds 29 new time series features as columns to the original DataFrame
    ( 
        tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
            .augment_timeseries_signature_polars(date_column='order_date')
            .head()
    )
    ```
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.apply(lambda x: x)
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")
    
    # Perform the feature engineering using Polars
    df_pl = pl.DataFrame(pandas_df)

    df_pl = df_pl.with_columns(
        # Date-Time Index Feature
        (pl.col(date_column).cast(pl.Int64) / 1_000_000_000).suffix("_index_num"),

        # Yearly Features
        pl.col(date_column).dt.year().suffix("_year"),
        pl.col(date_column).dt.iso_year().suffix("_year_iso"),
        (pl.when((pl.col(date_column).dt.month() == 1) & (pl.col(date_column).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_yearstart")
        ),
        (pl.when((pl.col(date_column).dt.month() == 12) & (pl.col(date_column).dt.day() == 31))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_yearend")
        ),
        pl.col(date_column).dt.is_leap_year().cast(pl.Int8).suffix("_leapyear"),

        # Semesterly Features
        (pl.when((pl.col(date_column).dt.quarter() == 1) | (pl.col(date_column).dt.quarter() == 2))
            .then(1)
            .otherwise(2)
            .suffix("_half")
        ),

        # Quarterly Features
        pl.col(date_column).dt.quarter().suffix("_quarter"),
        pl.col(date_column).dt.strftime("%Y").alias("quarteryear") + "Q" + pl.col(date_column).dt.quarter().cast(pl.Utf8, strict=False),

        pl.when((pl.col(date_column).dt.month().is_in([1,4,7,10])) & (pl.col(date_column).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_quarterstart"),
        pl.when((pl.col(date_column).dt.month().is_in([3,12])) & (pl.col(date_column).dt.day() == 31)
                                                        |
                (pl.col(date_column).dt.month().is_in([6,9])) & (pl.col(date_column).dt.day() == 30)
                )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_quarterend"),
        pl.col(date_column).dt.month().suffix("_month"),

        # Monthly Features
        pl.col(date_column).dt.strftime(format="%B").suffix("_month_lbl"),
        (pl.when((pl.col(date_column).dt.month() == 1) & (pl.col(date_column).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_monthstart")
        ),
        pl.when((pl.col(date_column).dt.month().is_in([1,3,5,7,8,10,12])) & (pl.col(date_column).dt.day() == 31)
                                                                   |
                (pl.col(date_column).dt.month().is_in([4,6,9,11])) & (pl.col(date_column).dt.day() == 30)
                                                            |
                (pl.col(date_column).dt.is_leap_year().cast(pl.Int8) == 0) & (pl.col(date_column).dt.day() == 28)
                                                                    |
                (pl.col(date_column).dt.is_leap_year().cast(pl.Int8) == 1) & (pl.col(date_column).dt.day() == 29)
                )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .suffix("_monthend"),

        # Weekly Features
        pl.col(date_column).dt.week().suffix("_yweek"),
        ((pl.col(date_column).dt.day() - 1) // 7 + 1).suffix("_mweek"),
        pl.col(date_column).dt.weekday().suffix("_wday"),
        pl.col(date_column).dt.strftime(format="%A").suffix("_wday_lbl"),

        # Daily Features
        pl.col(date_column).dt.day().suffix("_mday"),
        pl.when(pl.col(date_column).dt.quarter() == 1)
          .then(pl.col(date_column).dt.ordinal_day())
          .when(pl.col(date_column).dt.quarter() == 2)
          .then(pl.col(date_column).dt.ordinal_day() - 90)
          .when(pl.col(date_column).dt.quarter() == 3)
          .then(pl.col(date_column).dt.ordinal_day() - 181)
          .when(pl.col(date_column).dt.quarter() == 4)
          .then(pl.col(date_column).dt.ordinal_day() - 273).suffix("_qday"),
        pl.col(date_column).dt.ordinal_day().suffix("_yday"),
        (pl.when((pl.col(date_column).dt.weekday() <= 5))
          .then(0)
          .otherwise(1)
          .suffix("_weekend")
        ),

        # Hourly Features
        pl.col(date_column).dt.hour().suffix("_hour"),

        # Minute Features
        pl.col(date_column).dt.minute().suffix("_minute"),

        # Second Features
        pl.col(date_column).dt.second().suffix("_second"),

        # Microsecond Features
        pl.col(date_column).dt.microsecond().suffix("_msecond"),

        # Nanosecond Features
        pl.col(date_column).dt.nanosecond().suffix("_nanosecond"),

        # AM/PM
        (pl.when((pl.col(date_column).dt.hour() <= 12))
            .then("am")
            .otherwise("pm")
            .suffix("_am_pm")
        )
    )
    
    return df_pl.to_pandas()


# Dependencies
import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.datetime_helpers import week_of_month
from pytimetk.utils.checks import check_series_or_datetime, check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage

@pf.register_dataframe_method
def augment_timeseries_signature(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    reduce_memory: bool = False,
    engine: str = 'pandas',
) -> pd.DataFrame:
    """
    The function `augment_timeseries_signature` takes a DataFrame and a date 
    column as input and returns the original DataFrame with the **29 different 
    date and time based features** added as new columns with the feature name 
    based on the date_column.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the time series 
        data.
    date_column : str
        The `date_column` parameter is a string that represents the name of the 
        date column in the `data` DataFrame.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting datetime features. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for feature generation. This is generally faster than using "pandas" 
          for large datasets. 
    
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
    
    df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
    ```
    
    ```{python}
    # Adds 29 new time series features as columns to the original DataFrame (pandas engine)
    ( 
        df
            .augment_timeseries_signature(date_column='order_date', engine ='pandas')
            .glimpse()
    )
    ```
    
    ```{python}
    # Adds 29 new time series features as columns to the original DataFrame (polars engine)
    ( 
        df
            .augment_timeseries_signature(date_column='order_date', engine ='polars')
            .glimpse()
    )
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = pd.concat(
            [
                data, 
                data[date_column].get_timeseries_signature(engine=engine).drop(date_column, axis=1)
            ],
            axis=1)
    elif engine == 'polars':
        
        df_pl = pl.DataFrame(data)
        
        df_pl = _polars_timeseries_signature(df_pl, date_column = date_column)
        
        ret = df_pl.to_pandas()
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_timeseries_signature = augment_timeseries_signature

@pf.register_series_method
def get_timeseries_signature(
    idx: Union[pd.Series, pd.DatetimeIndex],
    reduce_memory: bool = False,
    engine: str = 'pandas'
    ) -> pd.DataFrame:
    """
    Convert a timestamp to a set of 29 time series features.

    The function `get_timeseries_signature` engineers **29 different date and 
    time based features** from a single datetime index `idx`: 

    Parameters
    ----------
    idx : pd.DataFrame
        The `idx` parameter is a pandas Series of DatetimeIndex.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting datetime features. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for feature generation. This is generally faster than using "pandas" 
          for large datasets. 
    
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
    
    dates = pd.date_range(start = '2019-01', end = '2019-03', freq = 'D')
    ```
    
    ```{python}
    # Makes 29 new time series features from the dates
    tk.get_timeseries_signature(dates, engine='pandas').glimpse()
    ```
    
    ```{python}
    tk.get_timeseries_signature(dates, engine='polars').glimpse()
    ```
    ```{python}
    pd.Series(dates, name = "date").get_timeseries_signature(engine='pandas').glimpse()
    ```
    
    ```{python}
    pd.Series(dates, name = "date").get_timeseries_signature(engine='polars').glimpse()
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
        ret = _get_timeseries_signature_pandas(idx)
    elif engine == 'polars':
        ret = _get_timeseries_signature_polars(idx)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
    
    return ret

# Monkey patch the method to Pandas Series objects
pd.Series.get_timeseries_signature = get_timeseries_signature

def _get_timeseries_signature_pandas(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.DataFrame:

    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name='idx')

    if idx.name is None:
        idx.name = 'idx'
    
    data = idx.to_frame()
    name = idx.name

    data = _pandas_timeseries_signature(data, date_column = name)

    return data

def _get_timeseries_signature_polars(idx: Union[pd.Series, pd.DatetimeIndex]) -> pl.DataFrame:
    
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name='idx')

    if idx.name is None:
        idx.name = 'idx'
    
    data = idx.to_frame()
    name = idx.name

    # Convert to Polars DataFrame
    df_pl = pl.DataFrame(data)

    # Helper function that works with polars objects
    df_pl = _polars_timeseries_signature(df_pl, date_column = name)

    return df_pl.to_pandas()



# UTILITIES
# ---------

def _pandas_timeseries_signature(data: pd.DataFrame, date_column: str) -> pd.DataFrame:
    
    name = date_column
    idx = data[name]
    
    # Date-Time Index Feature
    data[f'{name}_index_num'] = idx.astype(np.int64) // 10**9

    # Yearly Features
    data[f'{name}_year'] = idx.dt.year
    data[f'{name}_year_iso'] = idx.dt.isocalendar().year
    data[f'{name}_yearstart'] = idx.dt.is_year_start.astype('uint8')
    data[f'{name}_yearend'] = idx.dt.is_year_end.astype('uint8')
    data[f'{name}_leapyear'] = idx.dt.is_leap_year.astype('uint8')

    # Semesterly Features
    half = np.where(((idx.dt.quarter == 1) | ((idx.dt.quarter == 2))), 1, 2)
    data[f'{name}_half'] = pd.Series(half, index=idx.index)

    # Quarterly Features
    data[f'{name}_quarter'] = idx.dt.quarter
    quarteryear = pd.PeriodIndex(idx, freq='Q')
    data[f'{name}_quarteryear'] = pd.Series(quarteryear, index=idx.index).dt.strftime('%YQ%q')
    data[f'{name}_quarterstart'] = idx.dt.is_quarter_start.astype('uint8')
    data[f'{name}_quarterend'] = idx.dt.is_quarter_end.astype('uint8')

    # Monthly Features
    data[f'{name}_month'] = idx.dt.month
    data[f'{name}_month_lbl'] = idx.dt.month_name()
    data[f'{name}_monthstart'] = idx.dt.is_month_start.astype('uint8')
    data[f'{name}_monthend'] = idx.dt.is_month_end.astype('uint8')

    # Weekly Features
    data[f'{name}_yweek'] = idx.dt.isocalendar().week
    data[f'{name}_mweek'] = week_of_month(idx)

    # Daily Features
    data[f'{name}_wday'] = idx.dt.dayofweek + 1
    data[f'{name}_wday_lbl'] = idx.dt.day_name()
    data[f'{name}_mday'] = idx.dt.day
    data[f'{name}_qday'] = (idx.dt.tz_localize(None) - pd.PeriodIndex(idx.dt.tz_localize(None), freq='Q').start_time).dt.days + 1
    data[f'{name}_yday'] = idx.dt.dayofyear
    weekend = np.where((idx.dt.dayofweek <= 5), 0, 1)
    data[f'{name}_weekend'] = pd.Series(weekend, index=idx.index)

    # Hourly Features
    data[f'{name}_hour'] = idx.dt.hour

    # Minute Features
    data[f'{name}_minute'] = idx.dt.minute

    # Second Features
    data[f'{name}_second'] = idx.dt.second

    # Microsecond Features
    data[f'{name}_msecond'] = idx.dt.microsecond

    # Nanosecond Features
    data[f'{name}_nsecond'] = idx.dt.nanosecond

    # AM/PM
    am_pm = np.where((idx.dt.hour <= 12), 'am', 'pm')
    data[f'{name}_am_pm'] = pd.Series(am_pm, index=idx.index)
    
    return data
    

def _polars_timeseries_signature(data: pl.DataFrame, date_column: str) -> pl.DataFrame:
    
    df_pl = data
    name = date_column
    
    df_pl = df_pl.with_columns(
        # Date-Time Index Feature
        (pl.col(name).cast(pl.Int64) / 1_000_000_000).alias(f"{name}_index_num"),

        # Yearly Features
        pl.col(name).dt.year().alias(f"{name}_year"),
        pl.col(name).dt.iso_year().alias(f"{name}_year_iso"),
        (pl.when((pl.col(name).dt.month() == 1) & (pl.col(name).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_yearstart")
        ),
        (pl.when((pl.col(name).dt.month() == 12) & (pl.col(name).dt.day() == 31))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_yearend")
        ),
        pl.col(name).dt.is_leap_year().cast(pl.Int8).alias(f"{name}_leapyear"),

        # Semesterly Features
        (pl.when((pl.col(name).dt.quarter() == 1) | (pl.col(name).dt.quarter() == 2))
            .then(1)
            .otherwise(2)
            .alias(f"{name}_half")
        ),

        # Quarterly Features
        pl.col(name).dt.quarter().alias(f"{name}_quarter"),
        pl.col(name).dt.strftime("%Y").alias(f"{name}_quarteryear") + "Q" + pl.col(name).dt.quarter().cast(pl.Utf8, strict=False),

        pl.when((pl.col(name).dt.month().is_in([1, 4, 7, 10])) & (pl.col(name).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_quarterstart"),
        pl.when((pl.col(name).dt.month().is_in([3, 12])) & (pl.col(name).dt.day() == 31)
                | (pl.col(name).dt.month().is_in([6, 9])) & (pl.col(name).dt.day() == 30)
                )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_quarterend"),
        pl.col(name).dt.month().alias(f"{name}_month"),

        # Monthly Features
        pl.col(name).dt.strftime(format="%B").alias(f"{name}_month_lbl"),
        (pl.when((pl.col(name).dt.month() == 1) & (pl.col(name).dt.day() == 1))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_monthstart")
        ),
        pl.when((pl.col(name).dt.month().is_in([1, 3, 5, 7, 8, 10, 12]) & (pl.col(name).dt.day() == 31))
                | (pl.col(name).dt.month().is_in([4, 6, 9, 11]) & (pl.col(name).dt.day() == 30)
                | (pl.col(name).dt.is_leap_year().cast(pl.Int8) == 0) & (pl.col(name).dt.day() == 28)
                | (pl.col(name).dt.is_leap_year().cast(pl.Int8) == 1) & (pl.col(name).dt.day() == 29)
                ))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias(f"{name}_monthend"),

        # Weekly Features
        pl.col(name).dt.week().alias(f"{name}_yweek"),
        ((pl.col(name).dt.day() - 1) // 7 + 1).alias(f"{name}_mweek"),
        pl.col(name).dt.weekday().alias(f"{name}_wday"),
        pl.col(name).dt.strftime(format="%A").alias(f"{name}_wday_lbl"),

        # Daily Features
        pl.col(name).dt.day().alias(f"{name}_mday"),
        pl.when(pl.col(name).dt.quarter() == 1)
            .then(pl.col(name).dt.ordinal_day())
            .when(pl.col(name).dt.quarter() == 2)
            .then(pl.col(name).dt.ordinal_day() - 90)
            .when(pl.col(name).dt.quarter() == 3)
            .then(pl.col(name).dt.ordinal_day() - 181)
            .when(pl.col(name).dt.quarter() == 4)
            .then(pl.col(name).dt.ordinal_day() - 273)
            .alias(f"{name}_qday"),
        pl.col(name).dt.ordinal_day().alias(f"{name}_yday"),
        (pl.when((pl.col(name).dt.weekday() <= 5))
            .then(0)
            .otherwise(1)
            .alias(f"{name}_weekend")
        ),

        # Hourly Features
        pl.col(name).dt.hour().alias(f"{name}_hour"),

        # Minute Features
        pl.col(name).dt.minute().alias(f"{name}_minute"),

        # Second Features
        pl.col(name).dt.second().alias(f"{name}_second"),

        # Microsecond Features
        pl.col(name).dt.microsecond().alias(f"{name}_msecond"),

        # Nanosecond Features
        pl.col(name).dt.nanosecond().alias(f"{name}_nsecond"),

        # AM/PM
        pl.when(pl.col(name).dt.hour() <= 12).then(pl.lit("am")).otherwise(pl.lit("pm")).alias(f"{name}_am_pm"),
        
    )
    
    return df_pl

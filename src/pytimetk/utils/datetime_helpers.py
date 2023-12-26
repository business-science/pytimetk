import pandas as pd
import numpy as np
import polars as pl
import pandas_flavor as pf

import re
from datetime import datetime
from dateutil import parser
from warnings import warn
from typing import Union, List

from pytimetk.utils.checks import check_series_or_datetime
from pytimetk.utils.polars_helpers import pandas_to_polars_frequency
from pytimetk.utils.string_helpers import parse_freq_str

try: 
    import holidays
except ImportError:
    pass

@pf.register_series_method
def floor_date(
    idx: Union[pd.Series, pd.DatetimeIndex], 
    unit: str = "D",
    engine: str = 'pandas',
    ) -> pd.Series:
    '''
    Robust date flooring.
    
    The `floor_date` function takes a pandas Series of dates and returns a new Series 
    with the dates rounded down to the specified unit. It's more robust than the 
    pandas `floor` function, which does weird things with irregular frequencies 
    like Month which are actually regular.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter is a pandas Series or pandas DatetimeIndex object that 
        contains datetime values. It represents the dates that you want to round down.
    unit : str, optional
        The `unit` parameter in the `floor_date` function is a string that specifies 
        the time unit to which the dates in the `idx` series should be rounded down. 
        It has a default value of "D", which stands for day. Other possible values 
        for the `unit` parameter could be.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for calculating 
        the floor datetime. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library for 
          calculating the floor datetime. This can be faster than using "pandas" for 
          large datasets. 
    
    Returns
    -------
    pd.Series 
        The `floor_date` function returns a pandas Series object containing datetime64[ns] values.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="1H")
    dates
    ```
    
    ```{python}
    # Pandas fails to floor Month
    # dates.floor("M") # ValueError: <MonthEnd> is a non-fixed frequency
    
    # floor_date works as expected
    tk.floor_date(dates, unit="M", engine='pandas')
    ```
    '''

    # Common checks
    check_series_or_datetime(idx)

    if engine == 'pandas':
        return _floor_date_pandas(idx, unit)
    elif engine == 'polars':
        return _floor_date_polars(idx, unit)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _floor_date_pandas(
    idx: Union[pd.Series, pd.DatetimeIndex], 
    unit: str = "D",
    ) -> pd.Series:
    '''
    Robust date flooring.
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
        
    nm = idx.name
    
    # Fix for pandas bug: When unit is greater than the length of the index, the floor function returns the first day of the year
    days_in_sequence = idx.iloc[-1] - idx.iloc[0]
    days_in_unit = freq_to_timedelta(unit)
    
    if days_in_unit > days_in_sequence:
        # fill series with first value
        idx.iloc[1:] = idx.iloc[0]
    
    # Parse frequency string
    quantity, unit_2 = parse_freq_str(unit)
    quantity = int(quantity) if quantity else 1
    
    if quantity == 1:
        # DOESN'T WORK WITH MULTIPLES OF FREQUENCIES "2M" "3D" "4H" 
        date = pd.Series(
            pd.PeriodIndex(idx.values, freq = unit).to_timestamp(),
            name=nm
        ) 
    else:
        if unit_2 == "M":
            # Floor to N-month intervals using vectorized operations
            floored_months = ((idx.dt.month - 1) // quantity) * quantity + 1
            date = pd.to_datetime({
                'year': idx.dt.year,
                'month': floored_months,
                'day': 1
            })
            date = pd.Series(date, name=nm)
        elif unit_2 == "Q":
            # Floor to N-quarter intervals using vectorized operations
            floored_months = ((idx.dt.month - 1) // (3*quantity) ) * (3*quantity) + 1            
            date = pd.to_datetime({
                'year': idx.dt.year,
                'month': floored_months,
                'day': 1
            })
            date = pd.Series(date, name=nm)
        elif unit_2 == "Y":
            # Floor to N-year intervals using vectorized operations
            floored_years = (idx.dt.year // quantity) * quantity
            date = pd.to_datetime({
                'year': floored_years,
                'month': 1,
                'day': 1
            })
            date = pd.Series(date, name=nm)
        else:
            # DOESN'T WORK WITH IRREGULAR FREQUENCIES (MONTH, QUARTER, YEAR, ETC)
            date = idx.dt.floor(unit)    

    return date

def _floor_date_polars(
    idx: Union[pd.Series, pd.DatetimeIndex], 
    unit: str = "D",
) -> pd.Series:
    '''
    Robust date flooring.
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pl.Series(idx).alias('idx')
    
    date = (idx
            .dt
            .truncate(every=pandas_to_polars_frequency(unit))
            ).to_pandas()

    return date

@pf.register_series_method
def ceil_date(
    idx: Union[pd.Series, pd.DatetimeIndex], 
    unit: str = "D",
) -> pd.Series:
    '''
    Robust date ceiling.
    
    The `ceil_date` function takes a pandas Series of dates and returns a new 
    Series with the dates rounded up to the next specified unit. It's more 
    robust than the pandas `ceil` function, which does weird things with 
    irregular frequencies like Month which are actually regular.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter is a pandas Series or pandas DatetimeIndex object 
        that contains datetime values. It represents the dates that you want to 
        round down.
    unit : str, optional
        The `unit` parameter in the `ceil_date` function is a string that 
        specifies the time unit to which the dates in the `idx` series should be 
        rounded down. It has a default value of "D", which stands for day. Other 
        possible values for the `unit` parameter could be
    
    Returns
    -------
    pd.Series 
        The `ceil_date` function returns a pandas Series object containing 
        datetime64[ns] values.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="1H")
    dates
    ```
    
    ```{python}
    # Pandas ceil fails on month
    # dates.ceil("M") # ValueError: <MonthEnd> is a non-fixed frequency
    
    # Works on Month
    tk.ceil_date(dates, unit="M")
    ```
    '''
    # Common checks
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Convert to period
    date = floor_date(idx, unit=unit) + freq_to_dateoffset(unit)

    return date

def freq_to_dateoffset(freq_str):
    # Adjusted regex to account for potential absence of numeric part
    quantity, unit = parse_freq_str(freq_str)
    
    # Assume quantity of 1 if it's not explicitly provided
    quantity = int(quantity) if quantity else 1
    
    # Define a dictionary to map frequency units to DateOffset functions
    unit_to_offset = {
        'D'  : pd.DateOffset(days=quantity),
        'H'  : pd.DateOffset(hours=quantity),
        'T'  : pd.DateOffset(minutes=quantity),
        'min': pd.DateOffset(minutes=quantity),
        'S'  : pd.DateOffset(seconds=quantity),
        'L'  : pd.DateOffset(milliseconds=quantity),
        'U'  : pd.DateOffset(microseconds=quantity),
        'N'  : pd.DateOffset(nanoseconds=quantity),
        'Y'  : pd.DateOffset(years=quantity),
        'A'  : pd.DateOffset(years=quantity),
        'AS' : pd.DateOffset(years=quantity),
        'YS' : pd.DateOffset(years=quantity),
        'W'  : pd.DateOffset(weeks=quantity),
        'Q'  : pd.DateOffset(months=quantity*3),
        'QS' : pd.DateOffset(months=quantity*3),
        'M'  : pd.DateOffset(months=quantity),
        'MS' : pd.DateOffset(months=quantity)
    }
    
    # Get the DateOffset function for the given unit
    offset = unit_to_offset.get(unit)
    
    if offset is None:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    
    return offset

def freq_to_timedelta(freq_str):
    unit_mapping = {
        'D'  : 'days',
        'H'  : 'hours',
        'T'  : 'minutes',
        'min': 'minutes',
        'S'  : 'seconds',
        'L'  : 'milliseconds',
        'U'  : 'milliseconds',
        'N'  : 'nanoseconds',
        'Y'  : 'years',
        'A'  : 'years',
        'AS' : 'years',
        'YS' : 'years',
        'W'  : 'weeks',
        'Q'  : 'quarters',
        'QS' : 'quarters',
        'M'  : 'months',
        'MS' : 'months',
    }

    quantity, unit = parse_freq_str(freq_str)
    quantity = int(quantity) if quantity else 1

    if unit in unit_mapping:
        if unit_mapping[unit] == 'years':
            return pd.Timedelta(days=quantity*365.25)
        elif unit_mapping[unit] == 'quarters':
            return pd.Timedelta(days=quantity*3*30.44)
        elif unit_mapping[unit] == 'months':
            return pd.Timedelta(days=quantity*30.44)
        else:
            return pd.Timedelta(**{unit_mapping[unit]: quantity})
    else:
        raise ValueError(f"Unsupported frequency unit: {unit}")

def parse_end_date(date_str):
    
    date = pd.to_datetime(date_str)
    
    # Determine the granularity of the input and apply the appropriate offset
    if len(date_str) == 4:  # Year granularity
        end_date = date + pd.offsets.YearEnd()
    elif len(date_str) == 7:  # Month granularity
        end_date = date + pd.offsets.MonthEnd()
    elif len(date_str) == 10:  # Day granularity
        end_date = pd.Timestamp(date_str).replace(hour=23, minute=59, second=59)
    elif len(date_str) == 13:  # Hour granularity
        end_date = pd.Timestamp(date_str).replace(minute=59, second=59)
    elif len(date_str) == 16:  # Minute granularity
        end_date = pd.Timestamp(date_str).replace(second=59)
    elif len(date_str) == 19:  # Second Granularity
        end_date = date
    else:  # Day or finer granularity
        # Assuming you want to roll up to the end of the day
        end_date = pd.Timestamp(date_str).replace(hour=23, minute=59, second=59)

    return end_date
    

@pf.register_series_method
def week_of_month(
    idx: Union[pd.Series, pd.DatetimeIndex],
    engine: str = 'pandas',
    ) -> pd.Series:
    '''
    The "week_of_month" function calculates the week number of a given date 
    within its month.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The parameter "idx" is a pandas Series object that represents a specific
        date for which you want to determine the week of the month.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        calculating the week of the month. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
        for calculating the week of the month. This can be faster than using 
        "pandas" for large datasets. 

    
    Returns
    -------
    pd.Series
        The week of the month for a given date.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    dates = pd.date_range("2020-01-01", "2020-02-28", freq="1D")
    dates
    ```
    
    ```{python}
    # Works on DateTimeIndex
    tk.week_of_month(dates, engine='pandas')
    ```
    
    ```{python}
    # Works on DateTimeIndex
    tk.week_of_month(dates, engine='polars')
    ```
    
    ```{python}
    # Works on Pandas Series
    dates.to_series().week_of_month()
    ```
    
    ```{python}
    # Works on Pandas Series
    dates.to_series().week_of_month(engine='polars')
    ```
    
    '''
    # Common checks
    check_series_or_datetime(idx)

    if engine == 'pandas':
        return _week_of_month_pandas(idx)
    elif engine == 'polars':
        return _week_of_month_polars(idx)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _week_of_month_pandas(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    '''
    The "week_of_month" function calculates the week number of a given date within its month.
    '''
 
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    ret = (idx.dt.day - 1) // 7 + 1
    
    ret = pd.Series(
        ret, 
        name="week_of_month", 
        index = idx.index
    )
    
    return ret

def _week_of_month_polars(idx: Union[pd.Series, pd.DatetimeIndex]) -> pd.Series:
    '''
    The "week_of_month" function calculates the week number of a given date within its month.
    '''
    
    if isinstance(idx, pd.DatetimeIndex):
        idx = pl.Series(idx).alias('idx')
    elif isinstance(idx, pd.Series):
        idx = pl.Series(idx).alias('idx')
    else:
        raise ValueError("Input 'idx' must be a Pandas DatetimeIndex or Series.")
        
    ret = (
        ((idx.dt.day() - 1) // 7 + 1)
        .alias('week_of_month')
        .to_pandas()
    )
    
    return ret

@pf.register_series_method
def is_holiday(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.Series],
    country_name: str = 'UnitedStates',
    country: str = None,
    engine: str = 'pandas'
) -> pd.Series:
    """
    Check if a given list of dates are holidays for a specified country.
    
    Note: This function requires the `holidays` package to be installed.

    Parameters
    ----------
    idx : Union[str, datetime, List[Union[str, datetime]], pd.Series]
        The dates to check for holiday status.
    country_name (str, optional):
        The name of the country for which to check the holiday status. Defaults 
        to 'UnitedStates' if not specified.
    country (str, optional):
        An alternative parameter to specify the country for holiday checking, 
        overriding country_name.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating the boolean series. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating a boolean of holidays or not holidays. This can be 
          faster than using "pandas" for long series. 

    Returns:
    -------
    pd.Series:
        Series containing True if the date is a holiday, False otherwise.

    Raises:
    -------
    ValueError:
        If the specified country is not found in the holidays package.

    Examples:
    --------
    ```{python}
    import polars as pl
    import pytimetk as tk
    
    tk.is_holiday('2023-01-01', country_name='UnitedStates')
    ```
    
    ```{python}
    # List of dates
    tk.is_holiday(['2023-01-01', '2023-01-02', '2023-01-03'], country_name='UnitedStates')
    ```
    
    ```{python}
    # Polars Series
    tk.is_holiday(pl.Series(['2023-01-01', '2023-01-02', '2023-01-03']), country_name='UnitedStates')
    ```
    """
    
    if engine == 'pandas':
        return _is_holiday_pandas(idx, country_name , country)
    elif engine == 'polars':
        return _is_holiday_polars(idx, country_name, country)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _is_holiday_pandas(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.DatetimeIndex, pd.Series],
    country_name: str = 'UnitedStates',
    country: str = None
) -> pd.Series:

    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError("The 'holidays' package is not installed. Please install it by running 'pip install holidays'.")

    if country:
        country_name = country  # Override the default country_name with the provided one

    # Find the country module from the holidays package
    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")
    
    if isinstance(idx, str) or isinstance(idx, datetime):
        idx = [idx]
    
    idx = pd.to_datetime(idx)  # Convert all dates to pd.Timestamp if not already
    
    # Check each date if it's a holiday and return the results as a Series
    ret = pd.Series([date in country_module(years=date.year) for date in idx], name='is_holiday')
    
    return ret

def _is_holiday_polars(
    idx: Union[str, datetime, List[Union[str, datetime]], pd.Series],
    country_name: str = 'UnitedStates',
    country: str = None
) -> pd.Series:

    # This function requires the holidays package to be installed
    try:
        import holidays
    except ImportError:
        raise ImportError("The 'holidays' package is not installed. Please install it by running 'pip install holidays'.")

    if country:
        country_name = country  # Override the default country_name with the provided one
    
    # Convert pl.date objects if they are strings
    if isinstance(idx, str):
        start_year, start_month, start_day = map(int,idx.split("-"))
        start = pl.date(start_year, start_month, start_day)
        end = start
        end_year, end_month, end_day = map(int,idx.split("-"))

    # Convert to pl.date objects DatetimeIndex object
    if isinstance(idx, pd.core.indexes.datetimes.DatetimeIndex):
        date_range = idx

        start_date = date_range[0]
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
        start = pl.date(start_year, start_month, start_day)

        end_date   = date_range[-1]
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
        end = pl.date(end_year, end_month, end_day)
        
    # Convert list of strings to list of dates
    if isinstance(idx, list):
        dates = []
        for date_str in idx:
            date = pd.to_datetime(date_str).date()
            dates.append(date)
        start_date = dates[0]
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
        start = pl.date(start_year, start_month, start_day)

        end_date   = dates[-1]
        end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
        end = pl.date(end_year, end_month, end_day)
        
    holidays_list = list(holidays.country_holidays(country_name, years=[start_year,end_year]))
    expr = pl.date_range(start, end)
    is_holiday = expr.is_in(holidays_list)

    ret = pl.select(is_holiday).to_series().to_pandas() 

    return ret

def is_datetime_string(x: Union[str, pd.Series, pd.DatetimeIndex]) -> bool:
    
    if isinstance(x, pd.Series):
        x = x.values[0]
    
    if isinstance(x, pd.DatetimeIndex):
        x = x[0]
    
    try:
        parser.parse(str(x))
        return True
    except ValueError:
        return False
    
def detect_timeseries_columns(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    
    df = data.head(1)
    
    if verbose:
        print(df)
    
    return df.map(is_datetime_string)

def has_timeseries_columns(data: pd.DataFrame, verbose: bool = False) -> bool:
    
    if verbose:
        print(detect_timeseries_columns(data).iloc[0])
    
    return detect_timeseries_columns(data).iloc[0].any()

def get_timeseries_colname(data: pd.DataFrame, verbose: bool = False) -> str:
    
    if verbose:
        print(detect_timeseries_columns(data).iloc[0])
        
    return detect_timeseries_columns(data).iloc[0].idxmax()

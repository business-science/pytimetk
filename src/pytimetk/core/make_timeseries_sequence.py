import pandas as pd
import polars as pl
from datetime import datetime, timedelta
import pandas_flavor as pf

from pytimetk.utils.datetime_helpers import is_holiday
from typing import Union

import holidays

@pf.register_series_method
def make_weekday_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None,
    engine: str = 'pandas'
) -> pd.Series:
    """
    Generate a sequence of weekday dates within a specified date range, 
    optionally excluding weekends and holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    sunday_to_thursday : bool, optional
        If True, generates a sequence with Sunday to Thursday weekdays (excluding 
        Friday and Saturday). If False (default), generates a sequence with 
        Monday to Friday weekdays.
    remove_holidays : bool, optional 
        If True, excludes holidays (based on the specified country) from the 
        generated sequence.
        If False (default), includes holidays in the sequence.
    country (str, optional): 
        The name of the country for which to generate holiday-specific sequences. 
        Defaults to None, which uses the United States as the default country.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating a weekday series. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating a weekday series. This can be faster than using 
          "pandas" for large datasets. 

    Returns
    -------
    pd.Series
        A Series containing the generated weekday dates.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    # United States has Monday to Friday as weekdays (excluding Saturday and 
    # Sunday and holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15", 
                              sunday_to_thursday = False, 
                              remove_holidays    = True, 
                              country            = 'UnitedStates',
                              engine             = 'pandas')
    ```
    
    ```{python}   
    # Israel has Sunday to Thursday as weekdays (excluding Friday and Saturday 
    # and Israel holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15", 
                              sunday_to_thursday = True, 
                              remove_holidays    = True, 
                              country            = 'Israel',
                              engine             = 'pandas')
    ```
    
    ```{python}   
    # Israel has Sunday to Thursday as weekdays (excluding Friday and Saturday 
    # and Israel holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15", 
                              sunday_to_thursday = True, 
                              remove_holidays    = True, 
                              country            = 'Israel',
                              engine             = 'polars')
    ```
    """

    if engine == 'pandas':
        return _make_weekday_sequence_pandas(start_date, end_date, sunday_to_thursday, remove_holidays, country)
    elif engine == 'polars':
        return _make_weekday_sequence_polars(start_date, end_date, sunday_to_thursday, remove_holidays, country)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _make_weekday_sequence_pandas(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.Series:
    # Convert start_date and end_date to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Create a list to store weekday dates
    weekday_dates = []

    # Define the default weekday range (Monday to Friday)
    weekday_range = [0, 1, 2, 3, 4]

    # If Sunday to Thursday schedule is specified, adjust the weekday range
    if sunday_to_thursday:
        weekday_range = [0, 1, 2, 3, 6]  # Sunday to Thursday

    # Generate weekday dates within the date range
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() in weekday_range:
            if not remove_holidays or not is_holiday(current_date, country=country).values[0]:  # Check for holidays if specified
                weekday_dates.append(current_date)
        current_date += timedelta(days=1)

    # Convert the list of weekday dates to a DataFrame
    df = pd.DataFrame({'Weekday Dates': weekday_dates})

    return df['Weekday Dates']

def _make_weekday_sequence_polars(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.Series:

    # Convert start_date and end_date to pl.Date objects if they are strings
    if isinstance(start_date, str):
        start_year, start_month, start_day = map(int,start_date.split("-"))
        start = pl.date(start_year, start_month, start_day)
    if isinstance(end_date, str):
        end_year, end_month, end_day = map(int,end_date.split("-"))
        end = pl.date(end_year, end_month, end_day)

    # Convert start_date and end_date to pl.Date objects if they are Pandas datetime
    if isinstance(start_date, pd._libs.tslibs.timestamps.Timestamp):
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
        start = pl.date(start_year, start_month, start_day)

    if isinstance(end_date, pd._libs.tslibs.timestamps.Timestamp):
      end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
      end = pl.date(end_year, end_month, end_day)
    
    # Create a list to store weekday dates
    weekday_dates = []

    # Define the default weekday range (Monday to Friday)
    weekday_range = [1, 2, 3, 4, 5]
    
    # If sunday_to_thursday is True, redefine the weekday range to Sunday to Thursday
    if sunday_to_thursday:
        weekday_range = [0, 1, 2, 3, 4]
    
    # Generate a sequence of dates within the specified date range
    expr = pl.date_range(start, end)
    
    # Filter out weekends if sunday_to_thursday is True
    expr = expr.filter(expr.dt.weekday().is_in(weekday_range))
    
    # Filter out holidays if remove_holidays is True
    if remove_holidays:
        if country:
            holidays_list = list(holidays.country_holidays(country, years=[start_year,end_year]))
        else:
            holidays_list = list(holidays.country_holidays("US", years=[start_year,end_year]))
        expr = expr.filter(~expr.is_in(holidays_list))
    
    # Convert the resulting expression to a DataFrame and return it
    df = pl.select(expr).to_pandas().squeeze()

    return df.rename('Weekday Dates')

@pf.register_series_method
def make_weekend_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None,
    engine: str = 'pandas'
) -> pd.Series:
    """
    Generate a sequence of weekend dates within a specified date range, 
    optionally excluding holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    friday_saturday (bool, optional): 
        If True, generates a sequence with Friday and Saturday as weekends.If 
        False (default), generates a sequence with Saturday and Sunday as 
        weekends.
    remove_holidays : bool, optional 
        If True, excludes holidays (based on the specified country) from the 
        generated sequence.
        If False (default), includes holidays in the sequence.
    country (str, optional): 
        The name of the country for which to generate holiday-specific sequences. 
        Defaults to None, which uses the United States as the default country.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating a weekend series. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating a weekend series. This can be faster than using 
          "pandas" for large datasets. 

    Returns
    -------
    pd.Series
        A Series containing the generated weekday dates.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    # United States has Saturday and Sunday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31", 
                             friday_saturday = False, 
                             remove_holidays = True, 
                             country         = 'UnitedStates',
                             engine          = 'pandas')
    ```
    
    ```{python}
    # Saudi Arabia has Friday and Saturday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31", 
                             friday_saturday = True, 
                             remove_holidays = True, 
                             country         = 'SaudiArabia',
                             engine          = 'pandas')
    ```
    
     # Saudi Arabia has Friday and Saturday as weekends, polars engine
    tk.make_weekend_sequence("2023-01-01", "2023-01-31", 
                             friday_saturday = True, 
                             remove_holidays = True, 
                             country         = 'SaudiArabia',
                             engine          = '')
    ```
    """
    
    if engine == 'pandas':
        return _make_weekend_sequence_pandas(start_date, end_date, friday_saturday, remove_holidays, country)
    elif engine == 'polars':
        return _make_weekend_sequence_polars(start_date, end_date, friday_saturday, remove_holidays, country)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _make_weekend_sequence_pandas(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.DataFrame:

    # Convert start_date and end_date to datetime objects if they are strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # Create a list to store weekend dates
    weekend_dates = []

    # Define the default weekend range (Saturday and Sunday)
    weekend_range = [5, 6]

    # If Friday and Saturday schedule is specified, adjust the weekend range
    if friday_saturday:
        weekend_range = [4, 5]  # Friday and Saturday

    # Generate weekend dates within the date range
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() in weekend_range:
            if not remove_holidays or not is_holiday(current_date, country=country).values[0]:  # Check for holidays if specified
                weekend_dates.append(current_date)
        current_date += timedelta(days=1)

    # Convert the list of weekend dates to a DataFrame
    df = pd.DataFrame({'Weekend Dates': weekend_dates})

    return df['Weekend Dates']



def _make_weekend_sequence_polars(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.Series:

    # Convert start_date and end_date to pl.Date objects if they are strings
    if isinstance(start_date, str):
        start_year, start_month, start_day = map(int,start_date.split("-"))
        start = pl.date(start_year, start_month, start_day)
    if isinstance(end_date, str):
        end_year, end_month, end_day = map(int,end_date.split("-"))
        end = pl.date(end_year, end_month, end_day)

    # Convert start_date and end_date to pl.Date objects if they are Pandas datetime
    if isinstance(start_date, pd._libs.tslibs.timestamps.Timestamp):
        start_year, start_month, start_day = start_date.year, start_date.month, start_date.day
        start = pl.date(start_year, start_month, start_day)

    if isinstance(end_date, pd._libs.tslibs.timestamps.Timestamp):
      end_year, end_month, end_day = end_date.year, end_date.month, end_date.day
      end = pl.date(end_year, end_month, end_day)
    
    # Create a list to store weekend dates
    weekday_dates = []

    # Define the default weekend range (Saturday and Sunday)
    weekend_range = [6, 7]
    
    # If Friday and Saturday schedule is specified, adjust the weekend range
    if friday_saturday:
        weekday_range = [5, 6] # Friday and Saturday
    
    # Generate a sequence of dates within the specified date range
    expr = pl.date_range(start, end)
    
    # Filter out weekends if sunday_to_thursday is True
    expr = expr.filter(expr.dt.weekday().is_in(weekend_range))
    
    # Filter out holidays if remove_holidays is True
    if remove_holidays:
        if country:
            holidays_list = list(holidays.country_holidays(country, years=[start_year,end_year]))
        else:
            holidays_list = list(holidays.country_holidays("US", years=[start_year,end_year]))
        expr = expr.filter(~expr.is_in(holidays_list))
    
    # Convert the resulting expression to a DataFrame and return it
    df = pl.select(expr).to_pandas().squeeze()

    return df.rename('Weekend Dates')

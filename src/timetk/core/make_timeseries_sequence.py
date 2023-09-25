import pandas as pd
from datetime import datetime, timedelta
import pandas_flavor as pf

from timetk.utils.datetime_helpers import is_holiday
from typing import Union

@pf.register_series_method
def make_weekday_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.DataFrame:
    """
    Generate a sequence of weekday dates within a specified date range, optionally excluding weekends and holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    sunday_to_thursday : bool, optional
        If True, generates a sequence with Sunday to Thursday weekdays (excluding Friday and Saturday). If False (default), generates a sequence with Monday to Friday weekdays.
    remove_holidays : bool, optional 
        If True, excludes holidays (based on the specified country) from the generated sequence.
        If False (default), includes holidays in the sequence.
    country (str, optional): 
        The name of the country for which to generate holiday-specific sequences. Defaults to None, which uses the United States as the default country.

    Returns
    -------
    pd.Series
        A Series containing the generated weekday dates.

    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk
    
    # United States has Monday to Friday as weekdays (excluding Saturday and Sunday and holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15", sunday_to_thursday=False, remove_holidays=True, country='UnitedStates')
    ```
    
    ```{python}   
    # Israel has Sunday to Thursday as weekdays (excluding Friday and Saturday and Israel holidays)
    tk.make_weekday_sequence("2023-01-01", "2023-01-15", sunday_to_thursday=True, remove_holidays=True, country='Israel')
    ```
    """
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


@pf.register_series_method
def make_weekend_sequence(
    start_date: Union[str, datetime, pd.DatetimeIndex],
    end_date: Union[str, datetime, pd.DatetimeIndex],
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.DataFrame:
    """
    Generate a sequence of weekend dates within a specified date range, optionally excluding holidays.

    Parameters
    ----------
    start_date : str or datetime or pd.DatetimeIndex
        The start date of the date range.
    end_date : str or datetime or pd.DatetimeIndex
        The end date of the date range.
    friday_saturday (bool, optional): 
        If True, generates a sequence with Friday and Saturday as weekends.If False (default), generates a sequence with Saturday and Sunday as weekends.
    remove_holidays (bool, optional): 
        If True, excludes holidays (based on the specified country) from the generated sequence. If False (default), includes holidays in the sequence.
    country (str, optional): 
        The name of the country for which to generate holiday-specific sequences. Defaults to None, which uses the United States as the default country.

    Returns
    -------
    pd.Series
        A Series containing the generated weekday dates.

    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk

    # United States has Saturday and Sunday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31", friday_saturday=False, remove_holidays=True, country='UnitedStates')
    ```
    
    ```{python}
    # Saudi Arabia has Friday and Saturday as weekends
    tk.make_weekend_sequence("2023-01-01", "2023-01-31", friday_saturday=True, remove_holidays=True, country='SaudiArabia')
    ```
    """
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



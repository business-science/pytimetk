import pandas as pd
from datetime import datetime, timedelta
import holidays
import math
import pandas_flavor as pf

@pf.register_series_method
def tk_make_weekday_sequence(
    start_date: datetime,
    end_date: datetime,
    sunday_to_thursday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.DataFrame:
    """
    Generate a sequence of weekday dates within a specified date range, optionally excluding weekends and holidays.

    Parameters
    ----------
        start_date (datetime): The start date of the date range.
        end_date (datetime): The end date of the date range.
        sunday_to_thursday (bool, optional): 
            If True, generates a sequence with Sunday to Thursday weekdays (excluding Friday and Saturday).
            If False (default), generates a sequence with Monday to Friday weekdays.
        remove_holidays (bool, optional): 
            If True, excludes holidays (based on the specified country) from the generated sequence.
            If False (default), includes holidays in the sequence.
        country (str, optional): 
            The name of the country for which to generate holiday-specific sequences.
            Defaults to None, which uses the United States as the default country.

    Returns
    -------
        pandas.DataFrame: A DataFrame containing the generated weekday dates.

    Examples
    --------
    ```{python}
    import pandas as pd
    import datetime
    import timetk as tk
        
    start_date_us = datetime(2023, 1, 1)
    end_date_us = datetime(2023, 1, 15)
    tk_make_weekday_sequence(start_date_us, end_date_us, sunday_to_thursday=False, remove_holidays=True, country='UnitedStates')
    ```
    ```{python}   
    start_date_il = datetime(2023, 1, 1)
    end_date_il = datetime(2023, 1, 15)
    tk_make_weekday_sequence(start_date_il, end_date_il, sunday_to_thursday=True, remove_holidays=True, country='Israel')
    ```
    """
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
            if not remove_holidays or not is_holiday(current_date, country=country):  # Check for holidays if specified
                weekday_dates.append(current_date)
        current_date += timedelta(days=1)

    # Convert the list of weekday dates to a DataFrame
    df = pd.DataFrame({'Weekday Dates': weekday_dates})

    return df

def is_holiday(
    date: datetime,
    country_name: str = 'UnitedStates',
    country: str = None
) -> bool:
    """
    Check if a given date is a holiday for a specified country.

    Args:
        date (datetime): The date to check for holiday status.
        country_name (str, optional): 
            The name of the country for which to check the holiday status.
            Defaults to 'UnitedStates' if not specified.
        country (str, optional): 
            An alternative parameter to specify the country for holiday checking, overriding country_name.

    Returns:
        bool: True if the date is a holiday, False otherwise.
        
    Raises:
        ValueError: If the specified country is not found in the holidays package.
    """
    if country:
        country_name = country  # Override the default country_name with the provided one

    # Extract the year from the date
    year = date.year

    # Retrieve the corresponding country's module using regular expressions
    for key in holidays.__dict__.keys():
        if key.lower() == country_name.lower():
            country_module = holidays.__dict__[key]
            break
    else:
        raise ValueError(f"Country '{country_name}' not found in holidays package.")

    # Check if the date is a holiday for the specified year and country
    if date in country_module(years=year):
        return True
    else:
        return False

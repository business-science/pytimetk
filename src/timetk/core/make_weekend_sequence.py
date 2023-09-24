import pandas as pd
from datetime import datetime, timedelta
import holidays
import math
import pandas_flavor as pf

@pf.register_series_method
def tk_make_weekend_sequence(
    start_date: datetime,
    end_date: datetime,
    friday_saturday: bool = False,
    remove_holidays: bool = False,
    country: str = None
) -> pd.DataFrame:
    """
    Generate a sequence of weekend dates within a specified date range, optionally excluding holidays.

    Parameters
    ----------
        start_date (datetime): The start date of the date range.
        end_date (datetime): The end date of the date range.
        friday_saturday (bool, optional): 
            If True, generates a sequence with Friday and Saturday as weekends.
            If False (default), generates a sequence with Saturday and Sunday as weekends.
        remove_holidays (bool, optional): 
            If True, excludes holidays (based on the specified country) from the generated sequence.
            If False (default), includes holidays in the sequence.
        country (str, optional): 
            The name of the country for which to generate holiday-specific sequences.
            Defaults to None, which uses the United States as the default country.

    Returns
    -------
        pandas.DataFrame: A DataFrame containing the generated weekend dates.

    Examples
    --------
    ```{python}
    import pandas as pd
    import datetime
    import timetk as tk

    start_date_us = datetime(2023, 1, 1)
    end_date_us = datetime(2023, 1, 15)
    tk_make_weekend_sequence(start_date_us, end_date_us, friday_saturday=False, remove_holidays=True, country='UnitedStates')
    ```
    
    ```{python}
    start_date_sa = datetime(2023, 1, 1)
    end_date_sa = datetime(2023, 1, 15)
    tk_make_weekend_sequence(start_date_sa, end_date_sa, friday_saturday=True, remove_holidays=True, country='SaudiArabia')
    ```
    """
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
            if not remove_holidays or not is_holiday(current_date, country=country):  # Check for holidays if specified
                weekend_dates.append(current_date)
        current_date += timedelta(days=1)

    # Convert the list of weekend dates to a DataFrame
    df = pd.DataFrame({'Weekend Dates': weekend_dates})

    return df

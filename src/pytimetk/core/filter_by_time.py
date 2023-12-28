
# Imports
import pandas as pd
import datetime
import pandas_flavor as pf
from typing import Union, Callable, Tuple, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.datetime_helpers import parse_end_date

# Function ----
@pf.register_dataframe_method
def filter_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    start_date: str = "start",
    end_date: str = "end",
    engine: str = 'pandas'
):

    """
    Filters a DataFrame or GroupBy object based on a specified date range.

    This function filters data in a pandas DataFrame or a pandas GroupBy object
    by a given date range. It supports various date formats and can handle both
    DataFrame and GroupBy objects.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The data to be filtered. It can be a pandas DataFrame or a pandas
        GroupBy object.
    date_column : str
        The name of the column in `data` that contains date information.
        This column is used for filtering the data based on the date range.
    start_date : str
        The start date of the filtering range. The format of the date can be
        YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DD HH, YYYY-MM-DD HH:SS, or YYYY-MM-DD HH:MM:SS.
        Default: 'start', which will filter from the earliest date in the data.
    end_date : str
        The end date of the filtering range. It supports the same formats as
        `start_date`.
        Default: 'end', which will filter until the latest date in the data.
    engine : str, default = 'pandas'
        The engine to be used for filtering the data. Currently, only 'pandas'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the filtered data within the specified
        date range.

    Raises
    ------
    ValueError
        If the provided date strings do not match any of the supported formats.

    Notes
    -----
    - The function uses pd.to_datetime to convert the start date 
      (e.g. start_date = "2014" becomes "2014-01-01").
    - The function internally uses the `parse_end_date` function to convert the
      end dates (e.g. end_date = "2014" becomes "2014-12-31").
    

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import datetime

    m4_daily_df = tk.datasets.load_dataset('m4_daily', parse_dates = ['date'])

    ```

    ```{python}
    # Example 1 - Filter by date

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014-07-03',
        end_date    = '2014-07-10'
    )

    df_filtered

    ```

    ```{python}
    # Example 2 - Filter by month.
    # Note: This will filter by the first day of the month.

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014-07',
        end_date    = '2014-09'
    )

    df_filtered

    ```

    ```{python}
    # Example 3 - Filter by year.
    # Note: This will filter by the first day of the year.

    df_filtered = tk.filter_by_time(
        data        = m4_daily_df,
        date_column = 'date',
        start_date  = '2014',
        end_date    = '2014'
    )

    df_filtered

    ```

    ```{python}
    # Example 4 - Filter by day/hour/minute/second
    # Here we'll use an hourly dataset, however this will also work for minute/second data

    # Load data and format date column appropriately
    m4_hourly_df = tk.datasets.load_dataset('m4_hourly', parse_dates = ['date'])

    df_filtered = tk.filter_by_time(
        data        = m4_hourly_df,
        date_column = "date",
        start_date  = '2015-07-01 12:00:00',
        end_date    = '2015-07-01 20:00:00'
    )

    df_filtered
    ```

    ```{python}
    # Example 5 - Combine year/month/day/hour/minute/second filters
    df_filtered = tk.filter_by_time(
        data        = m4_hourly_df,
        date_column = "date",
        start_date  = '2015-07-01',
        end_date    = '2015-07-29'
    )

    df_filtered

    ```

    ```{python}
    # Example 6 - Filter a GroupBy object

    df_filtered = (
        m4_hourly_df
            .groupby('id')
            .filter_by_time(
                date_column = "date",
                start_date  = '2015-07-01 12:00:00',
                end_date    = '2015-07-01 20:00:00'
            )
    )

    df_filtered
    ```

    """
    # Checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    # Engine
    if engine == 'pandas':
        return _filter_by_time_pandas(data, date_column, start_date, end_date)
    else:
        raise ValueError("Invalid engine. Current supported engines: 'pandas'")


# Monkey Patch the Method to Pandas Grouby Objects
pd.core.groupby.generic.DataFrameGroupBy.filter_by_time = filter_by_time

def _filter_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    start_date: str,
    end_date: str
):
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
        
    df = data.copy()  
    df[date_column] = pd.to_datetime(df[date_column])      
    
    # Handle start/end dates and parsing
    if start_date == 'start':
        start_date = df[date_column].min()
    if end_date == 'end':
        end_date = df[date_column].max()
    
    if isinstance(start_date, str):
        start_date_parsed = pd.to_datetime(start_date)
    else:
        start_date_parsed = start_date
        
    if isinstance(end_date, str):
        end_date_parsed = parse_end_date(end_date)
    else:
        end_date_parsed = end_date

    # If the original index has a timezone, apply it to the future dates
    if df[date_column].dt.tz is not None:
        start_date_parsed = start_date_parsed.tz_localize(df[date_column].dt.tz)
        end_date_parsed = end_date_parsed.tz_localize(df[date_column].dt.tz)
    
    
    # Filter
    filtered_df = df[(df[date_column] >= start_date_parsed) & (df[date_column] <= end_date_parsed)]

    # Return
    return filtered_df




# Utilities ----




# Imports
import pandas as pd
import datetime
import pandas_flavor as pf
from typing import Union, Callable, Tuple, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

# Function ----
@pf.register_dataframe_method
def filter_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    start_date: str,
    end_date: str,
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
        YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-MM-DD HH:MM:SS.
    end_date : str
        The end date of the filtering range. It supports the same formats as
        `start_date`.
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
    - The function internally uses the `parse_date` function to convert the
      string dates to datetime objects based on their format.
    - For GroupBy objects, the filtering is applied to each group individually.
    - The function ensures that the original data is not modified by creating
      a copy of the DataFrame before filtering.

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
            date_column = 'date,
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
            date_column = 'date,
            start_date  = '2014',
            end_date    = '2014'
        )

        df_filtered

      ```

      ```{python}
      # Example 4 - Filter by day/hour/minute/second
      # Here we'll use an hourly dataset, however this will also work for minute/second data

      # Load data and format date column appropriately
      m4_hourly_df = (
          tk.datasets.load_dataset('m4_hourly', parse_dates = ['date'])
           .assign(date = lambda x: x['date'].dt.strftime('%Y-%m-%d %H:%M:%S').astype('datetime64[ns]'))
      )

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
          end_date    = '2015-08'
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


# Monkey Patch the Method to Pandas Grouby Objecs
pd.core.groupby.generic.DataFrameGroupBy.filter_by_time = filter_by_time

def _filter_by_time_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    start_date: str,
    end_date: str
):

    # Function to parse dates based on length
    def parse_date(date_str):
        if len(date_str) == 4:
            return datetime.datetime.strptime(date_str, '%Y')
        elif len(date_str) == 7:  # Year-Month
            return datetime.datetime.strptime(date_str, '%Y-%m')
        elif len(date_str) == 10:  # Year-Month-Day
            return datetime.datetime.strptime(date_str, '%Y-%m-%d')
        elif len(date_str) == 19:  # Full datetime
            return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

        else: raise ValueError("Invalid date format. Supported formats: YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS")

    # Parsed Dates
    start_date_parsed = parse_date(start_date)
    end_date_parsed = parse_date(end_date)

    # Handle Dataframe
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        filtered_df = df[(df[date_column] >= start_date_parsed) & (df[date_column] <= end_date_parsed)]


    # Handle Groupby
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        def filter_group(group):
            group[date_column] = pd.to_datetime(group[date_column])
            return group[(group[date_column] >= start_date_parsed) & (group[date_column] <= end_date_parsed)]

        filtered_df = data.apply(filter_group)

    # Return
    return filtered_df




# Utilities ----



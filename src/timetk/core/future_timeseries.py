import pandas as pd
import pandas_flavor as pf

@pf.register_series_method
def make_future_timeseries(
    idx: pd.Series or pd.DateTimeIndex,
    length_out: int,
) -> pd.Series:
    '''Make future dates for a time series.
    
    The function `make_future_timeseries` takes a pandas Series or DateTimeIndex and generates a future sequence of dates based on the frequency of the input series.
    
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter is the input time series data. It can be either a pandas Series or a pandas DateTimeIndex. It represents the existing dates in the time series.
    length_out : int
        The parameter `length_out` is an integer that represents the number of future dates to generate in the time series.
    
    Returns
    -------
    pd.Series
        A pandas Series object containing future dates.
    
    Examples
    --------
    
    ```{python}
    import pandas as pd
    import timetk as tk
    
    dates = pd.Series(pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']))
    dates
    ```
    
    ```{python}
    # DateTimeIndex: Generate 5 future dates
    future_dates_dt = tk.make_future_timeseries(dates, 5)
    future_dates_dt
    ```
    
    ```{python}
    # Series: Generate 5 future dates
    pd.Series(future_dates_dt).make_future_timeseries(5)
    ```
    
    ```{python}
    timestamps = ["2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 03:00", "2023-01-01 04:00", "2023-01-01 05:00"]
    
    dates = pd.to_datetime(timestamps)
    
    tk.make_future_timeseries(dates, 5)
    ```
    
    ```{python}
    # Monthly Frequency: Generate 4 future dates
    dates = pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01", "2021-04-01"])

    tk.make_future_timeseries(dates, 4)
    ```
    
    ```{python}
    # Quarterly Frequency: Generate 4 future dates
    dates = pd.to_datetime(["2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01"])

    tk.make_future_timeseries(dates, 4)
    ```
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Check if idx is a Series
    if not isinstance(idx, pd.Series):
        raise TypeError('idx must be a pandas Series or DatetimeIndex object')
    
    # Create a DatetimeIndex from the provided dates
    dt_index = pd.DatetimeIndex(pd.Series(idx).values)
    
    # Determine the frequency
    frequency = dt_index.inferred_freq  
    
    # Generate the next four periods (dates)
    future_dates = pd.date_range(
        start   = dt_index[-1], 
        periods = length_out +1, 
        freq    = frequency
    )[1:]  # Exclude the first date as it's already in dt_index

    ret = pd.Series(future_dates)
    
    return ret


@pf.register_dataframe_method
def future_frame(
    data: pd.DataFrame,
    date_column: str, 
    length_out: int, 
    bind_data: bool = False
) -> pd.DataFrame:
    
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    # Prep groups
    group_names = None
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.set_index(date_column).groupby(group_names)
    
    # TODO - Finish this function
    
    return None
    
    
    
    
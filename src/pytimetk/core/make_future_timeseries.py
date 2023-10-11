import pandas as pd
import numpy as np
import pandas_flavor as pf
from typing import Union, Optional, List

from pytimetk.core.frequency import get_frequency

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column,  check_series_or_datetime

from pytimetk.utils.parallel_helpers import conditional_tqdm

@pf.register_series_method
def make_future_timeseries(
    idx: Union[str, List[str], pd.Series, pd.DatetimeIndex],
    length_out: int,
    freq: Optional[str] = None,
    force_regular: bool = False,
) -> pd.Series:
    '''Make future dates for a time series.
    
    The function `make_future_timeseries` takes a pandas Series or DateTimeIndex and generates a future sequence of dates based on the frequency of the input series.
    
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter is the input time series data. It can be either a pandas Series or a pandas DateTimeIndex. It represents the existing dates in the time series.
    length_out : int
        The parameter `length_out` is an integer that represents the number of future dates to generate for the time series.
    freq : str, optional
        The `frequency` parameter is a string that specifies the frequency of the future dates. If `frequency` is set to `None`, the frequency of the future dates will be inferred from the input data (e.g. business calendars might be used). The default value is `None`.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether the frequency of the future dates should be forced to be regular. If `force_regular` is set to `True`, the frequency of the future dates will be forced to be regular. If `force_regular` is set to `False`, the frequency of the future dates will be inferred from the input data (e.g. business calendars might be used). The default value is `False`.
    
    Returns
    -------
    pd.Series
        A pandas Series object containing future dates.
    
    Examples
    --------
    
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    # Works with a single date (must provide a length out and frequency if only 1 date is provided)
    tk.make_future_timeseries("2011-01-01", 5, "D")
    ```
    
    ```{python}
    # DateTimeIndex: Generate 5 future dates (with inferred frequency)
    
    dates = pd.Series(pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']))

    future_dates_dt = tk.make_future_timeseries(dates, 5)
    future_dates_dt
    ```
    
    ```{python}
    # Series: Generate 5 future dates
    pd.Series(future_dates_dt).make_future_timeseries(5)
    ```
    
    ```{python}
    # Hourly Frequency: Generate 5 future dates
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
    
    ```{python}
    # Irregular Dates: Business Days
    dates = pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05", "2021-01-06"])
    
    tk.get_frequency(dates)
    
    tk.make_future_timeseries(dates, 4)
    ```
    
    ```{python}
    # Irregular Dates: Business Days (Force Regular)    
    tk.make_future_timeseries(dates, 4, force_regular=True)
    ```
    '''
    # Convert idx to Pandas DateTime Index if it's a string or list of strings
    if isinstance(idx, str):
        idx = pd.to_datetime([idx])
        
    if isinstance(idx, list):
        idx = pd.to_datetime(idx)
        
    # Check if idx is a Series or DatetimeIndex
    check_series_or_datetime(idx)
        
    if len(idx) < 2:
        if freq is None:
            raise ValueError("`freq` must be provided if `idx` contains only 1 date.")
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    # Create a DatetimeIndex from the provided dates
    dt_index = pd.DatetimeIndex(pd.Series(idx).values)
    
    # Determine the frequency
    if freq is None:
        freq = get_frequency(dt_index, force_regular=force_regular)  
    
    # Generate the next four periods (dates)
    future_dates = pd.date_range(
        start   = dt_index[-1], 
        periods = length_out +1, 
        freq    = freq
    )[1:]  # Exclude the first date as it's already in dt_index

    # If the original index has a timezone, apply it to the future dates
    if idx.dt.tz is not None:
        future_dates = future_dates.tz_localize(idx.dt.tz)
    
    ret = pd.Series(future_dates)
    
    return ret


@pf.register_dataframe_method
def future_frame(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str, 
    length_out: int,
    freq: Optional[str] = None, 
    force_regular: bool = False,
    bind_data: bool = True,
    threads: int = 1,
    show_progress: bool = True
) -> pd.DataFrame:
    '''Extend a DataFrame or GroupBy object with future dates.
    
    The `future_frame` function extends a given DataFrame or GroupBy object with future dates based on a specified length, optionally binding the original data.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to extend with future dates.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to generate future dates.
    freq : str, optional
    length_out : int
        The `length_out` parameter specifies the number of future dates to be added to the DataFrame.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether the frequency of the future dates should be forced to be regular. If `force_regular` is set to `True`, the frequency of the future dates will be forced to be regular. If `force_regular` is set to `False`, the frequency of the future dates will be inferred from the input data (e.g. business calendars might be used). The default value is `False`.
    bind_data : bool, optional
        The `bind_data` parameter is a boolean flag that determines whether the extended data should be concatenated with the original data or returned separately. If `bind_data` is set to `True`, the extended data will be concatenated with the original data using `pd.concat`. If `bind_data` is set to `False`, the extended data will be returned separately. The default value is `True`.
    threads : int
        The `threads` parameter specifies the number of threads to use for parallel processing. If `threads` is set to `None`, it will use all available processors. If `threads` is set to `-1`, it will use all available processors as well.
    show_progress : bool, optional
        A boolean parameter that determines whether to display progress using tqdm. If set to True, progress will be displayed. If set to False, progress will not be displayed.
    
    Returns
    -------
    pd.DataFrame
        An extended DataFrame with future dates.
        
    See Also
    --------
    make_future_timeseries: Generate future dates for a time series.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    df = tk.load_dataset('m4_hourly', parse_dates = ['date'])
    df
    ```
    
    ```{python}
    # Extend the data for a single time series group by 12 hours
    extended_df = (
        df
            .query('id == "H10"')
            .future_frame(
                date_column = 'date', 
                length_out  = 12
            )
            .assign(id = lambda x: x['id'].ffill())
    )
    extended_df
    ```
    
    ```{python}
    # Extend the data for each group by 12 hours
    extended_df = (
        df
            .groupby('id')
            .future_frame(
                date_column = 'date', 
                length_out  = 12
            )
    )    
    extended_df
    ```
    
    ```{python}
    # Same as above, but just return the extended data with bind_data=False
    extended_df = (
        df
            .groupby('id')
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                bind_data   = False # Returns just future data
            )
    )    
    extended_df
    ```
    
    ```{python}
     # Working with irregular dates: Business Days (Stocks Data)
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
    df
    ```
    
    ```{python}
    # Allow irregular future dates (i.e. business days)
    extended_df = (
        df
            .groupby('symbol')
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                force_regular = False, # Allow irregular future dates (i.e. business days)),
                bind_data   = True
            )
    )    
    extended_df
    ```
    
    ```{python}
    # Force regular: Include Weekends
    extended_df = (
        df
            .groupby('symbol')
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                force_regular = True, # Force regular future dates (i.e. include weekends)),
                bind_data   = True
            )
    )    
    extended_df
    ```
    '''
    
    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    
    if isinstance(data, pd.DataFrame):
        
        ts_series = data[date_column]
            
        new_dates = make_future_timeseries(
            idx=ts_series, 
            length_out=length_out, 
            force_regular=force_regular
        )
        
        new_rows = pd.DataFrame({date_column: new_dates})
        
        if bind_data:
            extended_df = pd.concat([data, new_rows], axis=0, ignore_index=True)
        else:
            extended_df = new_rows
        
        return extended_df       
    
    
    # GROUPED EXTENSION - If data is a GroupBy object, extend with future dates by group
    
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        
        # If freq is None, infer the frequency from the first series in the data
        if freq is None:
            
            label_of_first_group = list(data.groups.keys())[0]
            first_group = data.get_group(label_of_first_group)
            
            freq = get_frequency(first_group[date_column].sort_values(), force_regular=force_regular)
        
        # Use agg to get the last date of each group in a vectorized manner
        last_dates_df = data.agg({date_column: 'max'}).reset_index()
        
        future_dates_list = []
        
        iterable = conditional_tqdm(last_dates_df.iterrows(), total=len(last_dates_df), display=show_progress)
        
        for _, row in iterable:
            future_dates = make_future_timeseries(
                idx=pd.Series(row[date_column]), 
                length_out=length_out,
                freq=freq, 
                force_regular=force_regular
            )
            
            future_dates_df = pd.DataFrame({date_column: future_dates})
            
            for group_name in group_names:
                future_dates_df[group_name] = row[group_name]
            
            future_dates_list.append(future_dates_df)
        
        future_dates_df = pd.concat(future_dates_list, axis=0).reset_index(drop=True)
        
        if bind_data:
            extended_df = pd.concat([data.obj, future_dates_df], axis=0).reset_index(drop=True)
        else:
            extended_df = future_dates_df
            
        return extended_df
    
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.future_frame = future_frame

# UTILITIES ------------------------------------------------------------------
def _parallel_group_extension(group_data, date_column, length_out, freq, force_regular, group_names):
    
    last_date = group_data[date_column].max()
    
    future_dates = make_future_timeseries(
        idx=pd.Series(last_date),
        length_out=length_out,
        freq=freq,
        force_regular=force_regular
    )

    future_dates_df = pd.DataFrame({date_column: future_dates})

    for group_name in group_names:
        future_dates_df[group_name] = group_data[group_name].iloc[0]

    return future_dates_df
import pandas as pd
import numpy as np
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
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str, 
    length_out: int, 
    bind_data: bool = True,
) -> pd.DataFrame:
    '''Extend a DataFrame or GroupBy object with future dates.
    
    The `future_frame` function extends a given DataFrame or GroupBy object with future dates based on a specified length, optionally binding the original data.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to extend with future dates.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to generate future dates.
    length_out : int
        The `length_out` parameter specifies the number of future dates to be added to the DataFrame.
    bind_data : bool, optional
        The `bind_data` parameter is a boolean flag that determines whether the extended data should be concatenated with the original data or returned separately. If `bind_data` is set to `True`, the extended data will be concatenated with the original data using `pd.concat`. If `bind_data` is set to `False`, the extended data will be returned separately. The default value is `True`.
    
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
    import timetk as tk
    
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
    # Same as above, but just return the extended data
    extended_df = (
        df
            .groupby('id')
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                bind_data   = False
            )
    )
    
    extended_df
    ```
    
    '''
    
    # Check if data is a Pandas DataFrame or GroupBy object
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    
    if isinstance(data, pd.DataFrame):
        
        ts_series = data[date_column]
            
        new_dates = make_future_timeseries(ts_series, length_out)
        
        new_rows = pd.DataFrame({date_column: new_dates})
        
        if bind_data:
            ret = pd.concat(
                [data, new_rows], 
                axis         = 0,
                ignore_index = True
            )
        else:
            ret = new_rows
            
        extended_df = ret        
    
    
    # GROUPED EXTENSION - If data is a GroupBy object, extend with future dates by group
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj
        
        # Define a function to extend each group
        def extend_group(group_df):
            
            ts_series = group_df[date_column]
            
            new_dates = make_future_timeseries(ts_series, length_out)
            
            new_rows = pd.DataFrame({date_column: new_dates})
            for group_name in group_names:
                new_rows[group_name] = group_df[group_name].iloc[0]       
            
            if bind_data:
                ret = pd.concat(
                    [group_df, new_rows], 
                    axis         = 0,
                    ignore_index = True
                )
            else:
                ret = new_rows
            
            return ret 
        
        # Group by 'group' column and apply the function to each group
        extended_df = (
            data
                .groupby(
                    group_names, 
                    as_index   = False, 
                    group_keys = False
                )
                .apply(extend_group)
        )   
    
    return extended_df
    
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.future_frame = future_frame
    
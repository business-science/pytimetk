import pandas as pd
import pandas_flavor as pf
import numpy as np

from timetk.utils import get_pandas_frequency

    
@pf.register_dataframe_method
def ts_summary(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str,
):
    '''Computes summary statistics for a time series data, either for the entire dataset or grouped by a specific column.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter can be either a Pandas DataFrame or a Pandas DataFrameGroupBy object. It represents the data that you want to summarize.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to compute summary statistics for the time series data.
    
    Returns
    -------
    pd.DataFrame
        The `ts_summary` function returns a summary of time series data. The summary includes the following statistics:
        - `date_n`: The number of observations in the time series.
        - `date_tz`: The time zone of the time series.
        - `date_start`: The first date in the time series.
        - `date_end`: The last date in the time series.
        - `freq_inferred_unit`: The inferred frequency of the time series from `pandas`.
        - `freq_median_timedelta`: The median time difference between consecutive observations in the time series.
        - `freq_median_scale`: The median time difference between consecutive observations in the time series, scaled to a common unit.
        - `freq_median_unit`: The unit of the median time difference between consecutive observations in the time series.
        - `diff_min`: The minimum time difference between consecutive observations in the time series as a timedelta.
        - `diff_q25`: The 25th percentile of the time difference between consecutive observations in the time series as a timedelta.
        - `diff_median`: The median time difference between consecutive observations in the time series as a timedelta.
        - `diff_mean`: The mean time difference between consecutive observations in the time series as a timedelta.
        - `diff_q75`: The 75th percentile of the time difference between consecutive observations in the time series as a timedelta.
        - `diff_max`: The maximum time difference between consecutive observations in the time series as a timedelta.
        - `diff_min_seconds`: The minimum time difference between consecutive observations in the time series in seconds.
        - `diff_q25_seconds`: The 25th percentile of the time difference between consecutive observations in the time series in seconds.
        - `diff_median_seconds`: The median time difference between consecutive observations in the time series in seconds.
        - `diff_mean_seconds`: The mean time difference between consecutive observations in the time series in seconds.
        - `diff_q75_seconds`: The 75th percentile of the time difference between consecutive observations in the time series in seconds.
        - `diff_max_seconds`: The maximum time difference between consecutive observations in the time series in seconds.
        
    
    Examples
    --------
    ```{python}
    import timetk as tk
    import pandas as pd
    
    dates = pd.to_datetime(["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10"])
    df = pd.DataFrame(dates, columns = ["date"])
    
    df.ts_summary(date_column = 'date')
    ```
    
    ```{python}
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
     
    df.groupby('symbol').ts_summary(date_column = 'date') 
    ```
    
    ```{python}
    # See also:
    tk.get_date_summary(df['date'])
    
    tk.get_frequency_summary(df['date'])
    
    tk.get_diff_summary(df['date'])
    
    tk.get_diff_summary(df['date'], numeric = True)

    ```
    '''
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
        
    if isinstance(data, pd.DataFrame):
        
        df = data.copy()        
        df.sort_values(by=[date_column], inplace=True)
        
        date = df[date_column]

        # Compute summary statistics
        date_summary = get_date_summary(date)
        frequency_summary = get_frequency_summary(date)
        diff_summary = get_diff_summary(date)
        diff_summary_num = get_diff_summary(date, numeric = True)
        
        # Combine summary statistics into a single DataFrame
        return pd.concat([date_summary, frequency_summary, diff_summary, diff_summary_num], axis = 1)

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        data = data.obj
        
        df = data.copy()
        
        df.sort_values(by=[*group_names, date_column], inplace=True)
        
        # Get unique group combinations
        groups = df[group_names].drop_duplicates()
        
        summary_dfs = []
        for idx, group in groups.iterrows():
            mask = (df[group_names] == group).all(axis=1)
            group_df = df[mask]
            
            date = group_df[date_column]
            
            # Compute summary statistics
            date_summary = get_date_summary(date)
            frequency_summary = get_frequency_summary(date)
            diff_summary = get_diff_summary(date)
            diff_summary_num = get_diff_summary(date, numeric = True)
            
            unique_id = ' | '.join(group.values)
            
            # Combine summary statistics into a single DataFrame
            summary_df = pd.concat([pd.Series(unique_id, name="unique_id"),date_summary, frequency_summary, diff_summary, diff_summary_num], axis = 1)
            
            # Append to list of summary DataFrames
            summary_dfs.append(summary_df)
            
        return pd.concat(summary_dfs, axis = 0).reset_index(drop=True)
        
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.ts_summary = ts_summary
        
  
    
def get_diff_summary(idx: pd.Series or pd.DateTimeIndex, numeric: bool = False):
    '''Calculates summary statistics of the time differences between consecutive values in a datetime index.
    
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter can be either a pandas Series or a pandas DateTimeIndex. It represents the index values for which you want to calculate the difference summary.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that indicates whether the input index should be treated as numeric or not. 
        
        - If `numeric` is set to `True`, the index values are converted to integers representing the number of seconds since the Unix epoch (January 1, 1970). 
        
        - If `numeric` is set to `False`, the index values are treated as datetime values. The default value of `numeric` is `False`.
    
    Returns
    -------
    pd.DataFrame
        The function `get_diff_summary` returns a pandas DataFrame containing summary statistics including:
        
        If `numeric` is set to `False`, the column names are:
        - `diff_min`: The minimum time difference between consecutive observations in the time series as a timedelta.
        - `diff_q25`: The 25th percentile of the time difference between consecutive observations in the time series as a timedelta.
        - `diff_median`: The median time difference between consecutive observations in the time series as a timedelta.
        - `diff_mean`: The mean time difference between consecutive observations in the time series as a timedelta.
        - `diff_q75`: The 75th percentile of the time difference between consecutive observations in the time series as a timedelta.
        - `diff_max`: The maximum time difference between consecutive observations in the time series as a timedelta.
        
        If `numeric` is set to `True`, the column names are:
        - `diff_min_seconds`: The minimum time difference between consecutive observations in the time series in seconds.
        - `diff_q25_seconds`: The 25th percentile of the time difference between consecutive observations in the time series in seconds.
        - `diff_median_seconds`: The median time difference between consecutive observations in the time series in seconds.
        - `diff_mean_seconds`: The mean time difference between consecutive observations in the time series in seconds.
        - `diff_q75_seconds`: The 75th percentile of the time difference between consecutive observations in the time series in seconds.
        - `diff_max_seconds`: The maximum time difference between consecutive observations in the time series in seconds.

    '''
    
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    if numeric:
        idx = idx.astype(np.int64) // 10**9

    date_diff = idx.diff()

    _diff_min = date_diff.min()
    _diff_q25 = date_diff.quantile(0.25)
    _diff_median = date_diff.median()
    _diff_mean = date_diff.mean()
    _diff_q75 = date_diff.quantile(0.75)
    _diff_max = date_diff.max()
    
    ret = pd.DataFrame({
        "diff_min": [_diff_min],
        "diff_q25": [_diff_q25],
        "diff_median": [_diff_median],
        "diff_mean": [_diff_mean],
        "diff_q75": [_diff_q75],
        "diff_max": [_diff_max]
    })
    
    if numeric:
        ret.columns = ["diff_min_seconds", "diff_q25_seconds", "diff_median_seconds", "diff_mean_seconds", "diff_q75_seconds", "diff_max_seconds"]
    
    return ret
    
    
def get_date_summary(idx: pd.Series or pd.DateTimeIndex):
    '''Returns a summary of the date-related information, including the number of dates, the time zone, the start
    date, and the end date.
    
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The parameter `idx` can be either a pandas Series or a pandas DateTimeIndex. It represents the dates or timestamps for which we want to generate a summary.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns: 
        - `date_n`: The number of dates in the index.
        - `date_tz`: The time zone of the dates in the index.
        - `date_start`: The first date in the index.
        - `date_end`: The last date in the index.
    
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    _n = len(idx)
    _tz = idx.dt.tz
    _date_start = idx.min()
    _date_end = idx.max()
    
    return pd.DataFrame({
        "date_n": [_n], 
        "date_tz": [_tz], # "America/New_York
        "date_start": [_date_start],
        "date_end": [_date_end],
    })  
   
    
    
def get_frequency_summary(idx: pd.Series or pd.DateTimeIndex):  
    '''Returns a summary including the inferred frequency, median time difference, scale, and unit.
    
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter is either a `pd.Series` or a `pd.DateTimeIndex`. It represents the index of a pandas DataFrame or Series, which contains datetime values.
    
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        - `freq_inferred_unit`: The inferred frequency of the time series from `pandas`.
        - `freq_median_timedelta`: The median time difference between consecutive observations in the time series.
        - `freq_median_scale`: The median time difference between consecutive observations in the time series, scaled to a common unit.
        - `freq_median_unit`: The unit of the median time difference between consecutive observations in the time series.
    
    '''
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    _freq_inferred = get_pandas_frequency(idx)
    
    _freq_median = idx.diff().median()
    
    _freq_median_seconds = _freq_median.total_seconds()
    
    if _freq_median_seconds < 60:
        _unit = "S"
        _scale = _freq_median_seconds
    elif _freq_median_seconds < 3600:
        _unit = "M"
        _scale = _freq_median_seconds / 60
    elif _freq_median_seconds < 86400:
        _unit = "H"
        _scale = _freq_median_seconds / 3600
    elif _freq_median_seconds < 604800:
        _unit = "D"
        _scale = _freq_median_seconds / 86400
    elif _freq_median_seconds < 604800:
        _unit = "W"
        _scale = _freq_median_seconds / 604800
    elif _freq_median_seconds < 2678400:
        _unit = "M"
        _scale = np.round(_freq_median_seconds / 2678400,0).astype("int")
    elif _freq_median_seconds < 7948800:
        _unit = "Q"
        _scale = np.round(_freq_median_seconds / 7948800,0).astype("int")
    else: 
        _unit = "Y"
        _scale = np.round(_freq_median_seconds / 31795200,1).astype("int")
    
    return pd.DataFrame({
        "freq_inferred_unit": [_freq_inferred],
        "freq_median_timedelta": [_freq_median],
        "freq_median_scale": [_scale],
        "freq_median_unit": [_unit],
    })
    
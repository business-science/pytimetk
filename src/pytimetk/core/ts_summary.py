import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union


    
@pf.register_dataframe_method
def ts_summary(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
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
        - If grouped data is provided, the returned data will contain the grouping columns first.
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
    import pytimetk as tk
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

            
            # Get group columns
            id_df = pd.DataFrame(group).T.reset_index(drop=True)
            # print(id_df)

            date = group_df[date_column]
            
            # Compute summary statistics
            date_summary = get_date_summary(date)
            frequency_summary = get_frequency_summary(date)
            diff_summary = get_diff_summary(date)
            diff_summary_num = get_diff_summary(date, numeric = True)
            
            unique_id = ' | '.join(group.values)
            
            # Combine summary statistics into a single DataFrame
            summary_df = pd.concat([id_df, date_summary, frequency_summary, diff_summary, diff_summary_num], axis = 1)
            
            # Append to list of summary DataFrames
            summary_dfs.append(summary_df)
            
        return pd.concat(summary_dfs, axis = 0).reset_index(drop=True)
        
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.ts_summary = ts_summary
        
  
    
def get_diff_summary(idx: Union[pd.Series, pd.DatetimeIndex], numeric: bool = False):
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
    
    
def get_date_summary(idx: Union[pd.Series, pd.DatetimeIndex]):
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
   
    
    
def get_frequency_summary(idx: Union[pd.Series, pd.DatetimeIndex]):  
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
    
    # Use time series frequency table
    _freq_table = timeseries_unit_frequency_table().set_index('unit')
    
    def lookup_freq(unit, type='freq'):
        return _freq_table.loc[unit, type]
    
    
    if _freq_median_seconds < lookup_freq('min'):
        _unit = "S"
        _scale = _freq_median_seconds
    elif _freq_median_seconds < lookup_freq('hour'):
        _unit = "T"
        _scale = _freq_median_seconds / lookup_freq('min')
    elif _freq_median_seconds < lookup_freq('day'):
        _unit = "H"
        _scale = _freq_median_seconds / lookup_freq('hour')
    elif _freq_median_seconds < lookup_freq('week'):
        _unit = "D"
        _scale = _freq_median_seconds / lookup_freq('day')
    elif _freq_median_seconds < lookup_freq('month', 'freq_min'):
        _unit = "W"
        _scale = _freq_median_seconds / lookup_freq('week')
    elif _freq_median_seconds < lookup_freq('quarter', 'freq_min'):
        _unit = "M"
        _scale = np.round(_freq_median_seconds / lookup_freq('month'),1)
    elif _freq_median_seconds < lookup_freq('year', 'freq_min'):
        _unit = "Q"
        _scale = np.round(_freq_median_seconds / lookup_freq('quarter'),1)
    else: 
        _unit = "Y"
        _scale = np.round(_freq_median_seconds / lookup_freq('year'),1)
        
    # CHECK TO SWITCH DAYS
    if _unit in ['M', 'Q', 'Y']:
        remainder = _scale - int(_scale)
        if 0.1 <= remainder <= 0.9:
            # Switch to days
            _scale = float(_freq_median.days)
            _unit = "D"
        
    ret = pd.DataFrame({
        "freq_inferred_unit": [_freq_inferred],
        "freq_median_timedelta": [_freq_median],
        "freq_median_scale": [_scale],
        "freq_median_unit": [_unit],
    })
    
    return ret
    

@pf.register_series_method
def get_frequency(idx: Union[pd.Series, pd.DatetimeIndex], force_regular: bool = False) -> str:
    '''
    Get the frequency of a pandas Series or DatetimeIndex.
    
    The function `get_frequency` first attempts to get a pandas inferred frequency. If the inferred frequency is None, it will attempt calculate the frequency manually. If the frequency cannot be determined, the function will raise a ValueError.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter can be either a `pd.Series` or a `pd.DatetimeIndex`. It represents the index or the time series data for which we want to determine the frequency.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether to force the frequency to be regular. If set to `True`, the function will convert irregular frequencies to their regular counterparts. For example, if the inferred frequency is 'B' (business days), it will be converted to 'D' (calendar days). The default value is `False`.
    
    Returns
    -------
    str
        The frequency of the given pandas series or datetime index.
    
    '''
    
    if isinstance(idx, pd.Series):
        idx = idx.values
        
    freq = get_pandas_frequency(idx, force_regular)
    
    if freq is None:
        freq = get_manual_frequency(idx)
    
    return freq

@pf.register_series_method
def get_manual_frequency(idx: Union[pd.Series, pd.DatetimeIndex]) -> str:
    
    freq_summary_df = get_frequency_summary(idx)
    
    freq_median_scale = freq_summary_df['freq_median_scale'].values[0]
    freq_median_unit = freq_summary_df['freq_median_unit'].values[0]
    freq_median_timedelta = freq_summary_df['freq_median_timedelta'].values[0]
    
    number = freq_median_scale
    remainder = number - int(number)
    
    # IRREGULAR FREQUENCIES (MONTH AND QUARTER)
    if freq_median_unit in ['M', 'Q', 'Y']:
        if 0.1 <= remainder <= 0.9:
            # Switch to days
            days = freq_median_timedelta.astype('timedelta64[D]').astype(int)
            
            freq_alias = f'{days}D'
        else:
            # Switch to Start
            if isinstance(idx, pd.Series):
                idx = idx.values
            if idx[0].day == 1:
                freq_alias = f'{int(number)}{freq_median_unit.upper()}S'
            else:
                freq_alias = f'{int(number)}{freq_median_unit.upper()}'
    else:
        freq_alias = f'{int(number)}{freq_median_unit.upper()}'
        
    return freq_alias
    
    

@pf.register_series_method
def get_pandas_frequency(idx: Union[pd.Series, pd.DatetimeIndex], force_regular: bool = False) -> str:
    '''
    Get the frequency of a pandas Series or DatetimeIndex.
    
    The function `get_pandas_frequency` takes a Pandas Series or DatetimeIndex as input and returns the inferred frequency of the index, with an option to force regular frequency.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter can be either a `pd.Series` or a `pd.DatetimeIndex`. It represents the index or the time series data for which we want to determine the frequency.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether to force the frequency to be regular. If set to `True`, the function will convert irregular frequencies to their regular counterparts. For example, if the inferred frequency is 'B' (business days), it will be converted to 'D' (calendar days). The default value is `False`.
    
    Returns
    -------
    str
        The frequency of the given pandas series or datetime index.
    
    '''
   
    if isinstance(idx, pd.Series):
        idx = idx.values
        
    _len = len(idx)
    if _len > 10:
        _len = 10
    
    dt_index = pd.DatetimeIndex(idx[0:_len])
    
    freq = dt_index.inferred_freq
    
    # if freq is None:
    #         raise ValueError("The frequency could not be detectied.")
    
    if force_regular:
        if freq == 'B':
            freq = 'D'
        if freq == 'BM':
            freq = 'M'
        if freq == 'BQ':
            freq = 'Q'
        if freq == 'BA':
            freq = 'A'
        if freq == 'BY':
            freq = 'Y'
        if freq == 'BMS':
            freq = 'MS'
        if freq == 'BQS':
            freq = 'QS'
        if freq == 'BYS':
            freq = 'YS'
        if freq == 'BAS':
            freq = 'AS'
        
    
    return freq

def timeseries_unit_frequency_table(wide_format: bool = False) -> pd.DataFrame:
    '''The function `timeseries_unit_frequency_table` returns a pandas DataFrame with units of time and
    their corresponding frequencies in seconds.
    
    Returns
    -------
    pd.DataFrame
        a pandas DataFrame that contains two columns: "unit" and "freq". The "unit" column contains the units of time (seconds, minutes, hours, etc.), and the "freq" column contains the corresponding frequencies in seconds for each unit.
    
    '''
    
    _freq_table = pd.DataFrame({
        "unit" : ["sec", "min", "hour", "day", "week", "month", "quarter", "year"],
        "freq" : [0, 60, 3600, 86400, 604800, 2678400, 7948800, 31622400],
        "freq_min": [0, 60, 3600, 86400, 604800, 2419200, 7689600, 31536000],
        "freq_max": [0, 60, 3600, 86400, 604800, 2678400, 8035200, 31622400],
    })
    
    if wide_format:
        _freq_table = _freq_table.set_index('unit').T
    
    return _freq_table
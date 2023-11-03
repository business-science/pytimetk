import pandas as pd
import pandas_flavor as pf
import numpy as np
import polars as pl

from typing import Union

from pytimetk.utils.checks import check_series_or_datetime
from pytimetk.utils.datetime_helpers import floor_date

def get_frequency_summary(
        idx: Union[pd.Series, pd.DatetimeIndex],
        force_regular: bool = False
):  
    '''
    More robust version of pandas inferred frequency.
        
    Parameters
    ----------
    idx : pd.Series or pd.DateTimeIndex
        The `idx` parameter is either a `pd.Series` or a `pd.DateTimeIndex`. It 
        represents the index of a pandas DataFrame or Series, which contains 
        datetime values.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether 
        to force the frequency to be regular. If set to `True`, the function 
        will convert irregular frequencies to their regular counterparts. For 
        example, if the inferred frequency is 'B' (business days), it will be 
        converted to 'D' (calendar days). The default value is `False`.
        
    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the following columns:
        - `freq_inferred_unit`: The inferred frequency of the time series from `pandas`.
        - `freq_median_timedelta`: The median time difference between consecutive 
           observations in the time series.
        - `freq_median_scale`: The median time difference between consecutive 
           observations in the time series, scaled to a common unit.
        - `freq_median_unit`: The unit of the median time difference between 
           consecutive observations in the time series.
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
        
    dates = pd.date_range(start = '2020-01-01', end = '2020-01-10', freq = 'D')
        
    tk.get_frequency_summary(dates)
    ```
        
    ```{python}
    # pandas inferred frequency fails
    dates = pd.to_datetime(["2021-01-01", "2021-02-01"])
        
    # Returns None
    dates.inferred_freq == None
        
    # Returns '1MS'
    tk.get_frequency_summary(dates)
        
    ``` 
    '''
    
    # common checks
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    _freq_inferred = _get_pandas_frequency(idx, force_regular = force_regular)
    
    _freq_median = idx.diff().median()
    
    _freq_median_seconds = _freq_median.total_seconds()
    
    # Use time series frequency table
    _table = timeseries_unit_frequency_table().set_index('unit')
    
    def lookup_freq(unit, type='freq'):
        return _table.loc[unit, type]
    
    
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
        
    # SWITCH DAYS IF REMAINDER IS BETWEEN 0.1 AND 0.9
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
def get_frequency(idx: Union[pd.Series, pd.DatetimeIndex], force_regular: bool = False, numeric: bool = False) -> str:
    '''
    Get the frequency of a pandas Series or DatetimeIndex.
    
    The function `get_frequency` first attempts to get a pandas inferred 
    frequency. If the inferred frequency is None, it will attempt calculate the 
    frequency manually. If the frequency cannot be determined, the function will 
    raise a ValueError.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter can be either a `pd.Series` or a `pd.DatetimeIndex`. 
        It represents the index or the time series data for which we want to 
        determine the frequency.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether 
        to force the frequency to be regular. If set to `True`, the function 
        will convert irregular frequencies to their regular counterparts. For 
        example, if the inferred frequency is 'B' (business days), it will be 
        converted to 'D' (calendar days). The default value is `False`.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that indicates whether a 
        numeric value for the median timestamps per pandas frequency or the 
        pandas string frequency alias.
    
    Returns
    -------
    str
        The frequency of the given pandas series or datetime index.
    
    '''
    # common checks
    check_series_or_datetime(idx)
    
    if len(idx) <2:
        raise ValueError("Cannot determine frequency with less than 2 timestamps. Please provide a timeseries with at least 2 timestamps.")
    
    if isinstance(idx, pd.Series):
        idx = idx.values
        
    freq = _get_pandas_frequency(idx, force_regular)
    
    if freq is None:
        freq = _get_manual_frequency(idx)
        
    # Convert to numeric
    if numeric:
        freq = _get_median_timestamps(idx, freq)
    
    return freq

def timeseries_unit_frequency_table(
    wide_format: bool = False,
    engine: str = 'pandas'
    ) -> pd.DataFrame:
    '''
    The function `timeseries_unit_frequency_table` returns a pandas DataFrame 
    with units of time and their corresponding frequencies in seconds.
 
    Parameters
    ----------
    wide_format : bool, optional
        The wide_format parameter determines the format of the output table. If 
        wide_format is set to True, the table will be transposed.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating the timeseries unit frequency table. It can be either "pandas" 
        or "polars". 
            
            - The default value is "pandas".
            
            - When "polars", the function will internally use the `polars` library 
            for generating a timeseries unit frequency table. 

    Returns
    -------
    pd.DataFrame
        a pandas DataFrame that contains two columns: "unit" and "freq". The 
        "unit" column contains the units of time (seconds, minutes, hours, etc.), 
        and the "freq" column contains the corresponding frequencies in seconds 
        for each unit.


    Examples
    --------
    ```{python}
    import pytimetk as tk
    
    tk.timeseries_unit_frequency_table()
    ```
    
    '''
    if engine == 'pandas':
        return _timeseries_unit_frequency_table_pandas(wide_format)
    elif engine == 'polars':
        return _timeseries_unit_frequency_table_polars(wide_format)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _timeseries_unit_frequency_table_pandas(wide_format: bool = False) -> pd.DataFrame:
    
    _table = pd.DataFrame({
        "unit" : ["sec", "min", "hour", "day", "week", "month", "quarter", "year"],
        "freq" : [0, 60, 3600, 86400, 604800, 2678400, 7948800, 31622400],
        "freq_min": [0, 60, 3600, 86400, 604800, 2419200, 7776000, 31536000],
        "freq_max": [0, 60, 3600, 86400, 604800, 2678400, 7948800, 31622400],
    })
    
    if wide_format:
        _table = _table.set_index('unit').T
    
    return _table

def _timeseries_unit_frequency_table_polars(wide_format: bool = False) -> pd.DataFrame:

    _table = pl.DataFrame({
        "unit" :    ["sec", "min", "hour", "day", "week", "month", "quarter", "year"],
        "freq" :    [0, 60, 3600, 86400, 604800, 2678400, 7948800, 31622400],
        "freq_min": [0, 60, 3600, 86400, 604800, 2419200, 7776000, 31536000],
        "freq_max": [0, 60, 3600, 86400, 604800, 2678400, 7948800, 31622400],
    })
    
    if wide_format:
        col_names = _table.columns
        _table = _table.transpose().with_columns(pl.Series(name='unit', values=col_names))
        _table.columns=_table.iter_rows().__next__()
        _table = _table.slice(1)
        _table = _table[['unit'] + [col for col in _table.columns if col != 'unit']]
        
    return _table.to_pandas()

def time_scale_template(
    wide_format: bool = False,
    engine: str = 'pandas'
    ) -> pd.DataFrame:
    '''
    The function `time_scale_template` returns a table with time scale 
    information in either wide or long format.
    
    Parameters
    -------
    wide_format : bool, optional
        The wide_format parameter determines the format of the output table. If 
        wide_format is set to True, the table will be transposed.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating a date summary. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating the time scale information. 

    Examples
    --------
    ```{python}
    import pytimetk as tk
    
    tk.time_scale_template()
    ```
    
    '''
    if engine == 'pandas':
        return _time_scale_template_pandas(wide_format)
    elif engine == 'polars':
        return _time_scale_template_polars(wide_format)
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")

def _time_scale_template_pandas(wide_format: bool = False):
    
    _table = pd.DataFrame({
        "median_unit": ["S", "T", "H", "D", "W", "M", "Q", "Y"],
        "seasonal_period" : ["1H", "1D", "1D", "1W", "1Q", "1Y", "1Y", "5Y"],
        "trend_period" : ["12H", "14D", "1M", "1Q", "1Y", "5Y", "10Y", "30Y"],
    })
    
    if wide_format:
        _table = _table.set_index('median_unit').T
    
    return _table

def _time_scale_template_polars(wide_format: bool = False):

    _table = pl.DataFrame({
        "median_unit": ["S", "T", "H", "D", "W", "M", "Q", "Y"],
        "seasonal_period" : ["1H", "1D", "1D", "1W", "1Q", "1Y", "1Y", "5Y"],
        "trend_period" : ["12H", "14D", "1M", "1Q", "1Y", "5Y", "10Y", "30Y"],
    })
    
    if wide_format:
        col_names = _table.columns
        _table = _table.transpose().with_columns(pl.Series(name='median_unit', values=col_names))
        _table.columns=_table.iter_rows().__next__()
        _table = _table.slice(1)
        _table = _table[['median_unit'] + [col for col in _table.columns if col != 'median_unit']]
        
    return _table.to_pandas().set_index('median_unit')

 
@pf.register_series_method
def get_seasonal_frequency(
    idx: Union[pd.Series, pd.DatetimeIndex],
    force_regular: bool = False,
    numeric: bool = False,
    engine: str = 'pandas'
):
    '''
    The `get_seasonal_frequency` function returns the seasonal period of a given 
    time series or datetime index.
    
    Parameters
    ----------
    idx : Union[pd.Series, pd.DatetimeIndex]
        The `idx` parameter can be either a pandas Series or a pandas 
        DatetimeIndex. It represents the time index for which you want to 
        calculate the seasonal frequency.
    force_regular : bool, optional
        force_regular is a boolean parameter that determines whether to force 
        the frequency to be regular. If set to True, the function will try to 
        find a regular frequency even if the data is irregular. If set to False, 
        the function will return the actual frequency of the data.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that determines whether the 
        output should be in numeric format or a string Pandas Frequency Alias. 
        If `numeric` is set to `True`, the output will be a numeric representation 
        of the seasonal period. If `numeric` is set to `False` (default), the 
        output will
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating a date summary. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating the time scale information. 
    
    Returns
    -------
        The function `get_seasonal_frequency` returns the seasonal period based 
        on the input index. If the index is a `pd.DatetimeIndex`, it is converted 
        to a `pd.Series` with the name "idx". The function then calculates the 
        summary frequency of the index using the `get_frequency_summary` function. 
        It determines the scale and unit of the frequency and adjusts the unit if 
        the scale is
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='MS')
    
    tk.get_seasonal_frequency(dates)
    ```
    '''
    
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    summary_freq = get_frequency_summary(idx, force_regular = force_regular)
    
    scale = summary_freq['freq_median_scale'].values[0]
    unit = summary_freq['freq_median_unit'].values[0]
    
    unit = unit[0] # Get first letter if "MS", "QS", "AS", etc.
    
    if unit == "D":
        if scale > 1:
            if scale > 360:
                unit = "Y"
            elif scale > 31:
                unit = "Q" 
            else:
                unit = "M"  
    
    def _lookup_seasonal_period(unit):
        return time_scale_template(wide_format = True, engine = engine)[unit]['seasonal_period']
    
    _period = _lookup_seasonal_period(unit)
    
    if numeric:
        _period = _get_median_timestamps(idx, _period)
    
    return _period
    
@pf.register_series_method
def get_trend_frequency(
    idx: Union[pd.Series, pd.DatetimeIndex],
    force_regular: bool = False,
    numeric: bool = False,
    engine: str = 'pandas'
) -> str:
    '''
    The `get_trend_frequency` function returns the trend period of a given time 
    series or datetime index.
    
    Parameters
    ----------
    idx : Union[pd.Series, pd.DatetimeIndex]
        The `idx` parameter can be either a pandas Series or a pandas 
        DatetimeIndex. It represents the time index for which you want to 
        calculate the trend frequency.
    force_regular : bool, optional
        force_regular is a boolean parameter that determines whether to force the 
        frequency to be regular. If set to True, the function will try to find a 
        regular frequency even if the data is irregular. If set to False, the 
        function will return the actual frequency of the data.
    numeric : bool, optional
        The `numeric` parameter is a boolean flag that determines whether the 
        output should be in numeric format or a string Pandas Frequency Alias. 
        If `numeric` is set to `True`, the output will be a numeric representation 
        of the trend period. If `numeric` is set to `False` (default), the output 
        will
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        generating a date summary. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for generating the time scale information. 
    
    Returns
    -------
        The function `get_trend_frequency` returns the trend period based on the 
        input index. If the index is a `pd.DatetimeIndex`, it is converted to a 
        `pd.Series` with the name "idx". The function then calculates the summary 
        frequency of the index using the `get_frequency_summary` function. It 
        determines the scale and unit of the frequency and adjusts the unit if 
        the scale is
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq='MS')
    
    tk.get_trend_frequency(dates)    
    ```
    '''
    
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    summary_freq = get_frequency_summary(idx, force_regular = force_regular)
    
    scale = summary_freq['freq_median_scale'].values[0]
    unit = summary_freq['freq_median_unit'].values[0]
    
    if unit == "D":
        if scale > 1:
            if scale > 360:
                unit = "Y"
            elif scale > 31:
                unit = "Q" 
            else:
                unit = "M"  
    
    def _lookup_trend_period(unit):
        return time_scale_template(wide_format=True, engine = engine)[unit]['trend_period']
    
    _period = _lookup_trend_period(unit)
    
    if numeric:
        _period = _get_median_timestamps(idx, _period)
    
    return _period
    
def _get_median_timestamps(idx, period):
    
    check_series_or_datetime(idx)
    
    # If idx is a DatetimeIndex, convert to Series
    if isinstance(idx, pd.DatetimeIndex):
        idx = pd.Series(idx, name="idx")
    
    idx_floor = floor_date(idx, period) 
    idx_floor.name = 'idx_floor'
    
    df = pd.DataFrame(idx_floor)
    
    df['n'] = 1
    
    ret = df.groupby('idx_floor').sum()
    
    return ret.median().values[0]
    
# UTILITIES ---------------------------------------------------------------

def _get_manual_frequency(idx: Union[pd.Series, pd.DatetimeIndex]) -> str:
    '''
    This is an internal function and not meant to be called directly.
    
    Parameters
    ----------
    idx : Union[pd.Series, pd.DatetimeIndex]
        The `idx` parameter can be either a pandas Series or a pandas DatetimeIndex.
    
    Returns
    -------
        a string representing the frequency alias.
    
    '''
    
    
    # common checks
    check_series_or_datetime(idx)
    
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
    
def _get_pandas_frequency(idx: Union[pd.Series, pd.DatetimeIndex], force_regular: bool = False) -> str:
    '''
    This is an internal function and not meant to be called directly.
    
    Parameters
    ----------
    idx : pd.Series or pd.DatetimeIndex
        The `idx` parameter can be either a `pd.Series` or a `pd.DatetimeIndex`. 
        It represents the index or the time series data for which we want to 
        determine the frequency.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether 
        to force the frequency to be regular. If set to `True`, the function will 
        convert irregular frequencies to their regular counterparts. For example, 
        if the inferred frequency is 'B' (business days), it will be converted 
        to 'D' (calendar days). The default value is `False`.
    
    Returns
    -------
    str
        The frequency of the given pandas series or datetime index.
    
    '''
    if isinstance(idx, pd.Series):
        idx = idx.values
        
    if isinstance(idx, pd.DatetimeIndex):
        dt_index = idx
    else:
        _len = min(len(idx), 10)
        dt_index = pd.DatetimeIndex(idx[0:_len])
    
    freq = dt_index.inferred_freq
    
    if force_regular and freq:
        irregular_to_regular = {
            'A-DEC': 'Y',
            'Q-DEC': 'Q',
            'W-SUN': 'W',
            'B'    : 'D',
            'BM'   : 'M',
            'BQ'   : 'Q',
            'BA'   : 'A',
            'BY'   : 'Y',
            'BMS'  : 'MS',
            'BQS'  : 'QS',
            'BYS'  : 'YS',
            'BAS'  : 'AS'
        }
        freq = irregular_to_regular.get(freq, freq)
    
    return freq

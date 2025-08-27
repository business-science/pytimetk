import pandas as pd
from pandas.testing import assert_series_equal
import pytimetk as tk

def test_floor_date():
    
    dates = pd.DatetimeIndex(['2022-01-04', '2022-01-05', '2022-01-06'])
    
    # Week Flooring
    result = tk.floor_date(dates, 'W')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-03', '2022-01-03', '2022-01-03'])), check_freq=False, check_names=False)
    
    # Month
    result = tk.floor_date(dates, 'M')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01'])), check_freq=False, check_names=False)
    
    # Quarter   
    result = tk.floor_date(dates, 'Q')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01'])), check_freq=False, check_names=False)
    
    # Year
    result = tk.floor_date(dates, 'Y')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01'])), check_freq=False, check_names=False)
    
def test_floor_datetime():
    
    datetimes = pd.DatetimeIndex(['2022-01-04 01:00:01', '2022-01-04 01:00:02', '2022-01-04 01:00:03'])
    
    #  Minute
    result = tk.floor_date(datetimes, 'min')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-04 01:00:00', '2022-01-04 01:00:00', '2022-01-04 01:00:00'])), check_freq=False, check_names=False)
    
    # Hour
    result = tk.floor_date(datetimes, 'H')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-04 01:00:00', '2022-01-04 01:00:00', '2022-01-04 01:00:00'])), check_freq=False, check_names=False)
    
    # Day
    result = tk.floor_date(datetimes, 'D')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-04', '2022-01-04', '2022-01-04'])), check_freq=False, check_names=False)
    
    # Week
    result = tk.floor_date(datetimes, 'W')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-03', '2022-01-03', '2022-01-03'])), check_freq=False, check_names=False)
    
    # Month
    result = tk.floor_date(datetimes, 'M')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-01', '2022-01-01', '2022-01-01'])), check_freq=False, check_names=False)
    
    
def test_ceil_date():
    
    
    dates = pd.DatetimeIndex(['2022-01-04', '2022-01-05', '2022-01-06'])
    
    # Week 
    result = tk.ceil_date(dates, 'W')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-01-10', '2022-01-10', '2022-01-10'])), check_freq=False, check_names=False)
    
    # Month
    result = tk.ceil_date(dates, 'M')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-02-01', '2022-02-01', '2022-02-01'])), check_freq=False, check_names=False)
    
    # Quarter   
    result = tk.ceil_date(dates, 'Q')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2022-04-01', '2022-04-01', '2022-04-01'])), check_freq=False, check_names=False)
    
    # Year
    result = tk.ceil_date(dates, 'Y')
    
    assert_series_equal(result, pd.Series(pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-01'])), check_freq=False, check_names=False)
    
def test_floor_date_robust_examples():
    
    # Flooring with multiple hours
    dates = pd.date_range("2011-01-01", "2011-01-02", freq = "1H")
    
    result = tk.floor_date(dates, "4H")
    
    result = pd.DatetimeIndex(result.unique())
    
    expect = pd.DatetimeIndex(
        ['2011-01-01 00:00:00', 
         '2011-01-01 04:00:00',
        '2011-01-01 08:00:00', 
        '2011-01-01 12:00:00',
        '2011-01-01 16:00:00', 
        '2011-01-01 20:00:00',
        '2011-01-02 00:00:00']
    )
    
    assert result.equals(expect)
    
    # Yearly Flooring with Multiple Years
    
    dates = pd.date_range("2011", "2025", freq = "1Q")
    
    result = tk.floor_date(dates, "5Y")
    
    result = pd.DatetimeIndex(result.unique())
    
    expect = pd.DatetimeIndex(['2010-01-01', '2015-01-01','2020-01-01'])
    
    assert result.equals(expect)
    
    # Flooring with multiple months
    
    dates = pd.date_range("2011-02-01", "2012-02-01", freq = "1MS")
    
    result = tk.floor_date(dates, "2M")
    
    result = pd.DatetimeIndex(result.unique())
    
    expect = pd.DatetimeIndex(
        ['2011-01-01', 
         '2011-03-01', 
         '2011-05-01', 
         '2011-07-01',
         '2011-09-01', 
         '2011-11-01', 
         '2012-01-01']
    )
    
    assert result.equals(expect)
    
    # Flooring with multiple quarters
    
    dates = pd.date_range("2011-02-01", "2012-02-01", freq = "1MS")
    
    result = tk.floor_date(dates, "2Q")
    
    result = pd.DatetimeIndex(result.unique())
    
    expect = pd.DatetimeIndex(
        ['2011-01-01', '2011-07-01', '2012-01-01']
    )
    
    assert result.equals(expect)
    
    
    
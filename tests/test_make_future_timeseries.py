import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
import pytimetk as tk


def test_make_future_timeseries_daily():
    
    # Daily Test    
    result_1 = tk.make_future_timeseries("2011-01-01", 5, freq="D")
        
    result_2 = tk.make_future_timeseries(["2010-12-31", "2011-01-01"], 5)
    
    expect = pd.date_range("2011-01-02", periods=5, freq="D")
    expect = pd.Series(expect)
    
    assert_series_equal(result_1, expect)
    assert_series_equal(result_2, expect)
    
    # Override daily frequency
    result_3 = tk.make_future_timeseries(["2011-01-01", "2011-02-01"], 4, "H")
    
    expect = pd.Series(pd.date_range("2011-02-01 01:00:00", periods = 4, freq="H"))
    
    assert_series_equal(result_3, expect)

def test_make_future_timeseries_monthly():
    
    # Monthly Test    
    result_1 = tk.make_future_timeseries("2011-01-01", 5, "MS")
        
    result_2 = tk.make_future_timeseries(["2010-12-01", "2011-01-01"], 5)
    
    expect = pd.date_range("2011-02-01", periods=5, freq="MS")
    expect = pd.Series(expect)
    
    assert_series_equal(result_1, expect)
    assert_series_equal(result_2, expect)
    
    # Override monthly frequency
    result_3 = tk.make_future_timeseries(["2011-01-01", "2011-02-01"], 4, "D")
    
    expect = pd.Series(pd.date_range("2011-02-02", periods = 4, freq="D"))
    
    assert_series_equal(result_3, expect)

def test_make_future_timeseries_quarterly():
    
    # Quarter end
    result_1 = tk.make_future_timeseries("2011-01-01", 5, "Q")
    
    expect = pd.Series(pd.date_range("2011-06-30", periods = 5, freq = "Q"))
    
    assert_series_equal(result_1, expect)

    # Quarter start
    result_2 = tk.make_future_timeseries(["2011-01-01", "2011-04-01"], 5)
    
    expect = pd.Series(pd.date_range("2011-06-30", periods = 5, freq = "QS"))
    
    assert_series_equal(result_2, expect)

    # Override quarterly frequency
    result_3 = tk.make_future_timeseries(["2011-01-01", "2011-04-01"], 4, "D")
    
    expect = pd.Series(pd.date_range("2011-04-02", periods = 4, freq="D"))
    
    assert_series_equal(result_3, expect)
    
def test_make_future_timeseries_compound_freq():
    
    # Hourly + Minute Test    
    result_1 = tk.make_future_timeseries("2011-01-01", 5, freq="1H30min")
    
    # TODO : WORK ON COMPOUND FREQUENCIES
    # result_2 = tk.make_future_timeseries(["2010-12-31 22:30:00", "2011-01-01 00:00:00"], 5)
    # tk.make_future_timeseries(["2010-12-31 21:00:00", "2010-12-31 22:30:00", "2011-01-01 00:00:00"], 5)
    
    expect = pd.date_range("2011-01-01 01:30:00", periods=5, freq="1H30min")
    expect = pd.Series(expect)
    
    assert_series_equal(result_1, expect)
    # assert_series_equal(result_2, expect)

    
    
    
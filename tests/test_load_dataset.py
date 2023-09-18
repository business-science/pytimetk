import numpy as np
import pandas as pd
import pytest
import timetk

def test_load_dataset_wrong_name():
    """Test if load_dataset raises an error when the name is wrong"""
        
    with pytest.raises(ValueError):
        timetk.load_dataset("wrong_name")

def test_load_dataset():
    """Test if load_dataset is working"""
    
    # Options: ['bike_sales_sample', 'bike_sharing_daily', 'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_weekly', 'm4_yearly', 'taylor_30_min', 'walmart_sales_weekly', 'wikipedia_traffic_daily']
    
    # m4_daily
    data = timetk.load_dataset("m4_daily")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
    
    assert data.shape == (9743, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_hourly
    
    data = timetk.load_dataset("m4_hourly")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
        
    assert data.shape == (3060, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_weekly
    
    data = timetk.load_dataset("m4_weekly")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
        
    assert data.shape == (2295, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_monthly
    
    data = timetk.load_dataset("m4_monthly")
    
    assert data.shape == (1574, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # TODO: Add the rest of the datasets
    # m4_quarterly
    # m4_yearly
    # 'bike_sales_sample', 'bike_sharing_daily', 'taylor_30_min', 'walmart_sales_weekly', 'wikipedia_traffic_daily'
        
    
        
    
    

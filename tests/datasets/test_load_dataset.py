import numpy as np
import pandas as pd
import pytest
import pytimetk

def test_load_dataset_wrong_name():
    """Test if load_dataset raises an error when the name is wrong"""
        
    with pytest.raises(ValueError):
        pytimetk.load_dataset("wrong_name")

def test_load_dataset():
    """Test if load_dataset is working"""
    
    # Options: ['bike_sales_sample', 'bike_sharing_daily', 'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_weekly', 'm4_yearly', 'taylor_30_min', 'walmart_sales_weekly', 'wikipedia_traffic_daily']
    
    # m4_daily
    data = pytimetk.load_dataset("m4_daily")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
    
    assert data.shape == (9743, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_hourly
    
    data = pytimetk.load_dataset("m4_hourly")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
        
    assert data.shape == (3060, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_weekly
    
    data = pytimetk.load_dataset("m4_weekly")
    
    assert isinstance(data, pd.DataFrame), \
        'The dataset is not a pandas DataFrame!'
        
    assert data.shape == (2295, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
        
    # m4_monthly
    
    data = pytimetk.load_dataset("m4_monthly")
    
    assert data.shape == (1574, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'

    # m4_quarterly
    
    data = pytimetk.load_dataset("m4_quarterly")
    
    assert data.shape == (196, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'

    # m4_yearly
    
    data = pytimetk.load_dataset("m4_yearly")
    
    assert data.shape == (135, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'date', 'value'], \
        'The dataset has the wrong columns!'
    
    # bike_sales_sample
    
    data = pytimetk.load_dataset("bike_sales_sample")
    
    assert data.shape == (2466, 13), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['order_id', 'order_line', 'order_date', 'quantity', 'price',\
                                    'total_price', 'model', 'category_1', 'category_2', 'frame_material',\
                                    'bikeshop_name', 'city', 'state'], \
        'The dataset has the wrong columns!'

    # bike_sharing_daily
    
    data = pytimetk.load_dataset("bike_sharing_daily")
    
    assert data.shape == (731, 16), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', \
                                    'weekday', 'workingday', 'weathersit', 'temp', 'atemp', \
                                    'hum', 'windspeed', 'casual', 'registered', 'cnt'], \
        'The dataset has the wrong columns!'  

    # taylor_30_min
    
    data = pytimetk.load_dataset("taylor_30_min")
    
    assert data.shape == (4032, 2), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['date', 'value'], \
        'The dataset has the wrong columns!'  

    # walmart_sales_weekly
    
    data = pytimetk.load_dataset("walmart_sales_weekly")
    
    assert data.shape == (1001, 17), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['id', 'Store', 'Dept', 'Date', 'Weekly_Sales', \
                                    'IsHoliday', 'Type', 'Size', 'Temperature', \
                                    'Fuel_Price', 'MarkDown1', 'MarkDown2','MarkDown3', \
                                    'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment'], \
        'The dataset has the wrong columns!'  

    # wikipedia_traffic_daily
    
    data = pytimetk.load_dataset("wikipedia_traffic_daily")
    
    assert data.shape == (5500, 3), \
        'The dataset has the wrong shape!'
        
    assert data.columns.tolist() == ['Page', 'date', 'value'], \
        'The dataset has the wrong columns!'   

        
    
        
    
    


import pandas as pd
import pytimetk as tk
import numpy as np

from pandas.testing import assert_frame_equal

# IMPORTANT TEST: ORDER THE STOCKS BY DATE

stocks_df = tk.load_dataset("stocks_daily")
stocks_df['date'] = pd.to_datetime(stocks_df['date'])

stocks_df = stocks_df.sort_values("date").reset_index(drop=True)


# ROLLING ----

def test_sort_rolling_pandas():

    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'pandas',
            show_progress = False
        )
    
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])
    

def test_sort_rolling_pandas_lambda():
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = ('mean', lambda x: x.mean()),
            engine = 'pandas',
            show_progress = False
        )
    
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])

def test_sort_rolling_pandas_parallel():
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'pandas',
            show_progress = False,
            threads = 2
        )
    
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])

def test_sort_rolling_polars():
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'polars',
            show_progress = False
        )
        
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])
    

def test_sort_atr_pandas():
    
    result = stocks_df \
        .groupby('symbol') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "pandas")
        
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])
        
def test_sort_atr_polars():
    
    result = stocks_df \
        .groupby('symbol') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "polars")
        
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']])   
    
def test_sort_qsmom_pandas(): 
    
    result = stocks_df \
        .groupby('symbol') \
        .augment_qsmomentum(
            date_column = 'date', 
            close_column = 'close', 
            roc_fast_period = [21], 
            roc_slow_period = 252, 
            returns_period = 126, 
            engine = "pandas"
        ) 
    
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']]) 
    
def test_sort_qsmom_polars(): 
    
    result = stocks_df \
        .groupby('symbol') \
        .augment_qsmomentum(
            date_column = 'date', 
            close_column = 'close', 
            roc_fast_period = [21], 
            roc_slow_period = 252, 
            returns_period = 126, 
            engine = "polars"
        ) 
    
    assert_frame_equal(stocks_df[['symbol','date']], result[['symbol','date']]) 
    
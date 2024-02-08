
import pytest

import pandas as pd
import pytimetk as tk

df = tk.load_dataset("stocks_daily", parse_dates = ['date'])


# NATR
def test_atr():
    
    test_1 = df \
        .groupby('symbol') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "pandas")
        
    assert test_1.shape == (16194, 10)
        
    test_2 = df \
        .query('symbol == "GOOG"') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "pandas")
    
    assert test_2.shape == (2699, 10)

    test_3 = df \
        .groupby('symbol') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "polars")
        
    assert test_3.shape == (16194, 10)

    test_4 = df \
        .query('symbol == "GOOG"') \
        .augment_atr(date_column = 'date', high_column = 'high', low_column = 'low', close_column='close', periods = [14,28], normalize = True, engine = "polars")
    
    assert test_4.shape == (2699, 10)
        
    

# RSI
def test_rsi():
    
    test_1 = df \
        .groupby('symbol') \
        .augment_rsi(date_column = 'date', close_column='close', periods = [14,28], engine = "pandas")
        
    assert test_1.shape == (16194, 10)
        
    test_2 = df \
        .query('symbol == "GOOG"') \
        .augment_rsi(date_column = 'date', close_column='close', periods = [14,28], engine = "pandas")
        
    assert test_2.shape == (2699, 10)
        
    test_3 = df \
        .groupby('symbol') \
        .augment_rsi(date_column = 'date', close_column='close', periods = [14,28], engine = "polars")
        
    assert test_3.shape == (16194, 10)
        
    test_4 = df \
        .query('symbol == "GOOG"') \
        .augment_rsi(date_column = 'date', close_column='close', periods = [14,28], engine = "polars")
    
    assert test_4.shape == (2699, 10)

# BBANDS TEST
def test_bbands(): 
    
    test_1 = df \
        .groupby('symbol') \
        .augment_bbands(date_column = 'date', close_column='close', periods=20, std_dev=2, engine = "pandas")
    
    assert test_1.shape == (16194, 11)
        
    test_2 = df \
        .query('symbol == "AAPL"') \
        .augment_bbands(date_column = 'date', close_column='close', periods=[20, 40], std_dev=[1.5, 2], engine = "pandas")
    
    assert test_2.shape == (2699, 20)

    test_3 = df \
        .groupby('symbol') \
        .augment_bbands(date_column = 'date', close_column='close', periods=20, std_dev=2, engine = "polars")
    
    assert test_3.shape == (16194, 11)
        
    test_4 = df \
        .query('symbol == "AAPL"') \
        .augment_bbands(date_column = 'date', close_column='close', periods=[20, 40], std_dev=[1.5, 2], engine = "polars")
    
    assert test_4.shape == (2699, 20)


# PPO
def test_ppo():
    
    test_1 = df \
        .groupby('symbol') \
        .augment_ppo(date_column = 'date', close_column='close',
                    fast_period=12, slow_period=26, engine = "pandas")
    
    assert test_1.shape == (16194, 9)
        
    test_2 = df \
        .query('symbol == "GOOG"') \
        .augment_ppo(date_column = 'date', close_column='close', fast_period=12, slow_period=26, engine = "pandas")
    
    assert test_2.shape == (2699, 9)
        
    test_3 = df \
        .groupby('symbol') \
        .augment_ppo(date_column = 'date', close_column='close', fast_period=12, slow_period=26, engine = "polars")
    
    assert test_3.shape == (16194, 9)
        
    test_4 = df \
        .query('symbol == "GOOG"') \
        .augment_ppo(date_column = 'date', close_column='close', fast_period=12, slow_period=26, engine = "polars")
    
    assert test_4.shape == (2699, 9)
    

# MACD
def test_macd():
    
    test_1 = df \
        .groupby('symbol') \
        .augment_macd(date_column = 'date', close_column='close', fast_period=12, slow_period=26, signal_period=9, engine = "pandas")
    
    assert test_1.shape == (16194, 11)
        
    test_2 = df \
        .query('symbol == "GOOG"') \
        .augment_macd(date_column = 'date', close_column='close', fast_period=12, slow_period=26, signal_period=9, engine = "pandas")
    
    assert test_2.shape == (2699, 11)
        
    test_3 = df \
        .groupby('symbol') \
        .augment_macd(date_column = 'date', close_column='close', fast_period=12, slow_period=26, signal_period=9, engine = "polars")
     
    assert test_3.shape == (16194, 11)
        
    test_4 = df \
        .query('symbol == "GOOG"') \
        .augment_macd(date_column = 'date', close_column='close', fast_period=12, slow_period=26, signal_period=9, engine = "polars")
        
    assert test_4.shape == (2699, 11)
        

# CMO
def test_cmo():
    
    test_1 = df \
        .groupby('symbol') \
        .augment_cmo(date_column = 'date', close_column='close', periods = [14,28], engine = "pandas")
        
    assert test_1.shape == (16194, 10)    
        
    test_2 = df \
        .query('symbol == "GOOG"') \
        .augment_cmo(date_column = 'date', close_column='close', periods = [14,28], engine = "pandas")
     
    assert test_2.shape == (2699, 10) 
        
    test_3 = df \
        .groupby('symbol') \
        .augment_cmo(date_column = 'date', close_column='close', periods = [14,28], engine = "polars")
    
    assert test_3.shape == (16194, 10)
        
    test_4 = df \
        .query('symbol == "GOOG"') \
        .augment_cmo(date_column = 'date', close_column='close', periods = [14,28], engine = "polars")
    
    assert test_4.shape == (2699, 10)
    
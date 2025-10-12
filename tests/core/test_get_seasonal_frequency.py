import pandas as pd
import polars as pl
import pytimetk as tk
import pytest

@pytest.mark.parametrize('freq, regular, engine, expected',[
    ('D', False, 'pandas', '1W'), ('D', False, 'polars', '1W'),
    ('D', True,  'pandas', '1W'), ('D', True,  'polars', '1W'),
    ('W', False, 'pandas', '1Q'), ('W', False, 'polars', '1Q'),
    ('W', True,  'pandas', '1Q'), ('W', True,  'polars', '1Q'),
    ('M', False, 'pandas', '1Y'), ('M', False, 'polars', '1Y'),
    ('M', True,  'pandas', '1Y'), ('M', True,  'polars', '1Y'),
    ('Q', False, 'pandas', '1Y'), ('Q', False, 'polars', '1Y'),
    ('Q', True,  'pandas', '1Y'), ('Q', True,  'polars', '1Y'),
    ('Y', False, 'pandas', '5Y'), ('Y', False, 'polars', '5Y'),
    ('Y', True,  'pandas', '5Y'), ('Y', True,  'polars', '5Y'),
])
def test_correct_seasonal_inference(freq, regular, engine, expected):
    """
    This function tests if the inferred seasonal frequency of 
    a datetime series is correct.    
    """

    # Create a sample datetime series using freq parameter
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq=freq)
    dates_input = pl.Series("date", dates) if engine == 'polars' else dates

    # Invoke the get_seasonal_frequency function
    result = tk.get_seasonal_frequency(dates_input, force_regular=regular, engine=engine)

    # Check inferred frequency
    assert result == expected

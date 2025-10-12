import pandas as pd
import polars as pl
import pytimetk as tk
import pytest

@pytest.mark.parametrize('freq, regular, engine, expected',[
    ('D', False, 'pandas', '1Q'), ('D', False, 'polars', '1Q'),
    ('D', True,  'pandas', '1Q'), ('D', True,  'polars', '1Q'),
    ('W', False, 'pandas', '1Y'), ('W', False, 'polars', '1Y'),
    ('W', True,  'pandas', '1Y'), ('W', True,  'polars', '1Y'),
    ('M', False, 'pandas', '5Y'), ('M', False, 'polars', '5Y'),
    ('M', True,  'pandas', '5Y'), ('M', True,  'polars', '5Y'),
    ('Q', False, 'pandas', '10Y'), ('Q', False, 'polars', '10Y'),
    ('Q', True,  'pandas', '10Y'), ('Q', True,  'polars', '10Y'),
    ('Y', False, 'pandas', '30Y'), ('Y', False, 'polars', '30Y'),
    ('Y', True,  'pandas', '30Y'), ('Y', True,  'polars', '30Y'),
])
def test_correct_trend_inference(freq, regular, engine, expected):
    """
    This function tests if the inferred trend frequency of 
    a datetime series is correct.    
    """

    # Create a sample datetime series using freq parameter
    dates = pd.date_range(start='2021-01-01', end='2024-01-01', freq=freq)
    dates_input = pl.Series("date", dates) if engine == 'polars' else dates

    # Invoke the get_trend_frequency function
    result = tk.get_trend_frequency(dates_input, force_regular=regular, engine=engine)

    # Check inferred frequency
    assert result == expected

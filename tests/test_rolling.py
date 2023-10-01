import pytest
import pandas as pd
from timetk import augment_rolling  

# Sample data for testing
df = pd.DataFrame({
    'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
    'value': [1, 2, 3]
})

def test_augment_rolling_single_window_single_func():
    result = df.augment_rolling(date_column='date', value_column='value', window=2, window_func='mean')
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [1.0, 1.5, 2.5]
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_multi_window_multi_func():
    result = df.augment_rolling(date_column='date', value_column='value', window=[2, 3], window_func=['mean', 'sum'])
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [1.0, 1.5, 2.5]
    expected['value_rolling_sum_win_2'] = [1.0, 3.0, 5.0]
    expected['value_rolling_mean_win_3'] = [1.0, 1.5, 2.0]
    expected['value_rolling_sum_win_3'] = [1.0, 3.0, 6.0]
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_custom_func():
    #custom_func = ('custom', lambda x: x.max() - x.min())
    result = df.augment_rolling(date_column='date', value_column='value', window=2, window_func=[('custom', lambda x: x.max() - x.min())])
    expected = df.copy()
    expected['value_rolling_custom_win_2'] = [0.0, 1.0, 1.0]
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_invalid_data_type():
    with pytest.raises(TypeError):
        invalid_data = [1, 2, 3]  
        augment_rolling(data=invalid_data, date_column='date', value_column='value', window=2, window_func='mean')

def test_augment_rolling_invalid_window_type():
    with pytest.raises(TypeError):
        df.augment_rolling(date_column='date', value_column='value', window="invalid", window_func='mean')

def test_augment_rolling_invalid_func_name():
    with pytest.raises(ValueError):
        df.augment_rolling(date_column='date', value_column='value', window=2, window_func='invalid_function')

def test_augment_rolling_invalid_func_type():
    with pytest.raises(TypeError):
        df.augment_rolling(date_column='date', value_column='value', window=2, window_func=123)


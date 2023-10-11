import pytest
import pandas as pd
import numpy as np
from pytimetk import augment_rolling  

# Sample data for testing
df = pd.DataFrame({
    'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
    'value': [1, 2, 3]
})

# Generate sample data for testing
def generate_data(num_groups=5, num_entries_per_group=365):
    np.random.seed(42)
    date_rng = pd.date_range(start='2020-01-01', freq='D', periods=num_entries_per_group)
    data = []
    for i in range(num_groups):
        group_data = np.random.randn(num_entries_per_group).cumsum() + 50
        group_df = pd.DataFrame({
            'date': date_rng,
            'value': group_data,
            'group': f'group_{i}'
        })
        data.append(group_df)
    return pd.concat(data)



@pytest.fixture
def sample_data():
    return generate_data()

def test_dataframe_size(sample_data):
    df_rolled = augment_rolling(sample_data.groupby('group'), 'date', 'value', window_func='sum', threads=2)
    assert df_rolled.shape[1] > sample_data.shape[1]

def test_custom_functions(sample_data):
    custom_func = [('range', lambda x: x.max() - x.min())]
    df_rolled = augment_rolling(sample_data.groupby('group'), 'date', 'value', window_func=custom_func)
    expected_column = "value_rolling_range_win_2"
    assert expected_column in df_rolled.columns

def test_augment_rolling_single_window_single_func():
    result = df.augment_rolling(date_column='date', value_column='value', window=2, window_func='mean')
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [np.nan, 1.5, 2.5]
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_multi_window_multi_func():
    result = df.augment_rolling(date_column='date', value_column='value', window=[2, 3], window_func=['mean', 'sum'])
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [np.nan, 1.5, 2.5]
    expected['value_rolling_sum_win_2'] = [np.nan, 3.0, 5.0]
    expected['value_rolling_mean_win_3'] = [np.nan, 1.5, 2.0]
    expected['value_rolling_sum_win_3'] = [np.nan, 3.0, 6.0]
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_custom_func():
    #custom_func = ('custom', lambda x: x.max() - x.min())
    result = df.augment_rolling(date_column='date', value_column='value', window=2, window_func=[('custom', lambda x: x.max() - x.min())])
    expected = df.copy()
    expected['value_rolling_custom_win_2'] = [np.nan, 1.0, 1.0]
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


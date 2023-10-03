import pandas as pd
import pytest

# Ensure the function and dependencies are imported
from pytimetk import augment_lags

# Sample data for testing
df_sample = pd.DataFrame({
    'date': pd.date_range('2021-01-01', periods=5),
    'value': [1, 2, 3, 4, 5],
    'id': ['A', 'A', 'A', 'B', 'B']
})

# Basic Functionality Test
def test_basic_functionality():
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=1)
    assert 'value_lag_1' in df_result.columns
    assert df_result['value_lag_1'].iloc[1] == 1

# GroupBy Functionality Test
def test_groupby_functionality():
    df_result = df_sample.groupby('id').augment_lags(date_column='date', value_column='value', lags=1)
    assert 'value_lag_1' in df_result.columns
    assert pd.isna(df_result['value_lag_1'].iloc[3])

# Multiple Lags Test
def test_multiple_lags():
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=(1, 2))
    assert 'value_lag_1' in df_result.columns
    assert 'value_lag_2' in df_result.columns
    assert df_result['value_lag_1'].iloc[2] == 2
    assert df_result['value_lag_2'].iloc[2] == 1

# Invalid Lags Type Test
def test_invalid_lags_type():
    with pytest.raises(TypeError):
        df_sample.augment_lags(date_column='date', value_column='value', lags='string')

def test_invalid_dataframe_or_groupby_input():
    invalid_data = {"key": "value"}
    with pytest.raises(TypeError, match="`data` is not a Pandas DataFrame or GroupBy object."):
        augment_lags(data=invalid_data, date_column="date", value_column="value", lags=1)

# Value Column as List Test
def test_value_column_list():
    df_result = df_sample.augment_lags(date_column='date', value_column=['value'], lags=1)
    assert 'value_lag_1' in df_result.columns

# Lags as List Test
def test_lags_list():
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=[1, 3])
    assert 'value_lag_1' in df_result.columns
    assert 'value_lag_3' in df_result.columns
    assert df_result['value_lag_1'].iloc[3] == 3
    assert df_result['value_lag_3'].iloc[3] == 1

if __name__ == "__main__":
    pytest.main()
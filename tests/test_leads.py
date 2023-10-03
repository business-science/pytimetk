import pandas as pd
import pytest
import numpy as np
from pytimetk import augment_leads  

# Sample Data for testing
data = {
    'date': pd.date_range(start='2021-01-01', periods=5),
    'value': [1, 2, 3, 4, 5],
    'id': ['A', 'A', 'B', 'B', 'B']
}
df = pd.DataFrame(data)

df.dtypes
# Tests
def test_dataframe_extension_single_lead():
    result = augment_leads(df, date_column='date', value_column='value', leads=1)
    assert 'value_lead_1' in result.columns
    assert np.array_equal(result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, 4.0, 5.0, np.nan]), equal_nan=True)

def test_dataframe_extension_multiple_leads():
    result = augment_leads(df, date_column='date', value_column='value', leads=[1, 2])
    assert 'value_lead_1' in result.columns
    assert 'value_lead_2' in result.columns
    assert np.array_equal(result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, 4.0, 5.0, np.nan]), equal_nan=True)
    assert np.array_equal(result['value_lead_2'].to_numpy() , np.array([3.0, 4.0, 5.0, np.nan, np.nan]), equal_nan=True)

def test_groupby_extension_single_lead():
    grouped = df.groupby('id')
    result = augment_leads(grouped, date_column='date', value_column='value', leads=1)
    assert 'value_lead_1' in result.columns
    assert np.array_equal(result['value_lead_1'].to_numpy() , np.array([2.0, np.nan, 4.0, 5.0, np.nan]), equal_nan=True)

def test_invalid_data_type_error():
    with pytest.raises(TypeError):
        augment_leads("invalid_data", date_column='date', value_column='value', leads=1)

def test_invalid_leads_value_error():
    with pytest.raises(ValueError):
        augment_leads(df, date_column='date', value_column='value', leads="invalid_lead")


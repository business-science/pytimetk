import pytest
import pandas as pd
from timetk.utils import get_pandas_frequency
from timetk import ts_summary  # Adjust the module name accordingly

# Sample test data
dates = pd.to_datetime(["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10"])
df_sample = pd.DataFrame(dates, columns=["date"])

# Test ts_summary on a regular dataframe
def test_ts_summary_regular_df():
    result = df_sample.ts_summary(date_column='date')
    
    # Basic checks
    assert 'date_n' in result.columns
    assert result['date_n'].values[0] == len(df_sample)
    assert result['date_start'].values[0] == df_sample['date'].min()
    assert result['date_end'].values[0] == df_sample['date'].max()

# Test ts_summary on a grouped dataframe
def test_ts_summary_grouped_df():
    # Grouped DataFrame sample
    df_grouped = df_sample.copy()
    df_grouped['group'] = ['A', 'B', 'A', 'B', 'A', 'B', 'A']
    
    result = df_grouped.groupby('group').ts_summary(date_column='date')
    
    # Basic checks
    assert 'group' in result.columns
    assert result['date_n'].sum() == len(df_grouped)

# Test ts_summary type check for invalid data
def test_ts_summary_invalid_data_type():
    with pytest.raises(TypeError):
        ts_summary(data=[1, 2, 3], date_column='date')

# Test get_diff_summary for numeric flag
def test_get_diff_summary_numeric_flag():
    from timetk import get_diff_summary  # Adjust the module name accordingly
    result = get_diff_summary(df_sample['date'], numeric=True)
    
    # Checking columns for numeric flag
    assert 'diff_min_seconds' in result.columns
    assert 'diff_median_seconds' in result.columns

# Test get_date_summary for basic output
def test_get_date_summary():
    from timetk import get_date_summary  # Adjust the module name accordingly
    result = get_date_summary(df_sample['date'])
    
    # Basic checks
    assert 'date_n' in result.columns
    assert result['date_n'].values[0] == len(df_sample)

# Test get_frequency_summary for basic output
def test_get_frequency_summary():
    from timetk import get_frequency_summary  # Adjust the module name accordingly
    result = get_frequency_summary(df_sample['date'])
    
    # Basic checks
    assert 'freq_inferred_unit' in result.columns
    assert 'freq_median_timedelta' in result.columns

# More tests can be added as needed. 

if __name__ == "__main__":
    pytest.main([__file__])
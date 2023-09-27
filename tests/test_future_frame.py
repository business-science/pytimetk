import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
import timetk

# Creates a sample DataFrame for testing
data = {
    'date': pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06']),
    'value': [1, 2, 3, 4]
}
df = pd.DataFrame(data)

# Creates a sample irregular DataFrame for testing using business days
data_irr = {
    'date': pd.to_datetime(['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06','2022-01-07',\
                            '2022-01-10','2022-01-11','2022-01-12','2022-01-13','2022-01-14']),
    'value': [1, 2, 3, 4,5,6,7,8,9,10]
}
df_irr = pd.DataFrame(data_irr)


# Test make_future_timeseries function
def test_make_future_timeseries():
    # Test with regular frequency
    future_dates = timetk.make_future_timeseries(df['date'], 5)
    expected_dates = pd.Series(pd.date_range('2022-01-07', periods=5, freq='D'))

    assert_series_equal(future_dates, expected_dates, check_freq=False)

    # Test with irregular frequency (business days)
    future_dates_irregular = timetk.make_future_timeseries(df_irr['date'], 6, force_regular=False)
    expected_dates_irregular = pd.Series(pd.to_datetime(['2022-01-17', '2022-01-18', '2022-01-19','2022-01-20',\
                                                        '2022-01-21','2022-01-24']))
    assert_series_equal(future_dates_irregular, expected_dates_irregular, check_freq=False)


# Test future_frame function
def test_future_frame():
    # Test extending a DataFrame with regular frequency
    extended_df = df.future_frame(date_column='date', length_out=2)
    expected_data = {
        'date': ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08'],
        'value': [1, 2, 3, 4, np.nan, np.nan]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Convert the 'date' column to date-only format for comparison
    expected_df['date'] = pd.to_datetime(expected_df['date'])
    
    assert_frame_equal(extended_df, expected_df, check_dtype=False)

    # Test extending a DataFrame with irregular frequency (business days)
    extended_df_irregular = df_irr.future_frame(date_column='date', length_out=2, force_regular=False)
    expected_data_irregular = {
        'date': ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06','2022-01-07',\
                '2022-01-10','2022-01-11','2022-01-12','2022-01-13','2022-01-14','2022-01-17',\
                '2022-01-18'],
        'value': [1, 2, 3, 4,5,6,7,8,9,10, np.nan, np.nan]
    }
    expected_df_irregular = pd.DataFrame(expected_data_irregular)
    expected_df_irregular['date'] = pd.to_datetime(expected_df_irregular['date'])

    assert_frame_equal(extended_df_irregular, expected_df_irregular, check_dtype=False)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])

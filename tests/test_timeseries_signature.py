import pytest
import pandas as pd
from timetk import get_timeseries_signature, augment_timeseries_signature  

def test_get_timeseries_signature():

    # Create a sample date range
    dates = pd.date_range(start='2019-01', end='2019-03', freq='D')
    
    # Test if function produces the correct number of features
    signature = get_timeseries_signature(dates)
    assert signature.shape[1] == 29, "Number of features generated is not correct."
    
    # Test if the function produces the correct feature names
    expected_columns = [
        'index_num', 'year', 'year_iso', 'yearstart', 'yearend', 'leapyear', 
        'half', 'quarter', 'quarteryear', 'quarterstart', 'quarterend', 
        'month', 'month_lbl', 'monthstart', 'monthend', 'yweek', 'mweek', 
        'wday', 'wday_lbl', 'mday', 'qday', 'yday', 'weekend', 'hour', 
        'minute', 'second', 'msecond', 'nsecond', 'am_pm'
    ]
    assert all(signature.columns == expected_columns), "Feature names are not as expected."
    
    # Test if function raises TypeError for incorrect input types
    with pytest.raises(TypeError):
        get_timeseries_signature([1, 2, 3])

def test_augment_timeseries_signature():

    # Sample DataFrame
    df = pd.DataFrame({
        'order_date': pd.date_range(start='2019-01', end='2019-03', freq='D'),
        'value': range(60)
    })
    
    # Test if function adds the correct number of features
    augmented = df.augment_timeseries_signature(date_column='order_date')
    assert augmented.shape[1] == df.shape[1] + 29, "Number of features in the augmented dataframe is not correct."
    
    # Test if original dataframe data remains the same
    assert all(df['value'] == augmented['value']), "Original data in the augmented dataframe has changed."
    
    # Test if the function raises KeyError for non-existent column
    with pytest.raises(KeyError):
        df.augment_timeseries_signature(date_column='nonexistent_column')

if __name__ == "__main__":
    pytest.main()

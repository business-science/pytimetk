import pytest
import pandas as pd
from datetime import datetime

# Assuming the provided code is in a module named 'holidays_module'
import pytimetk


def test_augment_holiday_signature_us():
    df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", end="2023-01-10")
    })
    augmented_df = df.augment_holiday_signature('date', 'UnitedStates')

    assert 'is_holiday' in augmented_df.columns
    assert 'before_holiday' in augmented_df.columns
    assert 'after_holiday' in augmented_df.columns
    assert 'holiday_name' in augmented_df.columns

    # 2023-01-01 is New Year's Day in the US
    assert augmented_df.loc[augmented_df['date'] == datetime(2023, 1, 1), 'is_holiday'].item() == 1
    assert augmented_df.loc[augmented_df['date'] == datetime(2023, 1, 1), 'holiday_name'].item() == "New Year's Day"


def test_augment_holiday_signature_invalid_country():
    df = pd.DataFrame({
        'date': pd.date_range(start="2023-01-01", end="2023-01-10")
    })

    with pytest.raises(ValueError):
        df.augment_holiday_signature('date', 'InvalidCountry')


def test_get_holiday_signature():
    dates = pd.date_range(start="2023-01-01", end="2023-01-10")
    signature_df = pytimetk.get_holiday_signature(dates, 'UnitedStates')

    assert 'is_holiday' in signature_df.columns
    assert 'before_holiday' in signature_df.columns
    assert 'after_holiday' in signature_df.columns
    assert 'holiday_name' in signature_df.columns

    # 2023-01-01 is New Year's Day in the US
    assert signature_df.loc[signature_df['idx'] == datetime(2023, 1, 1), 'is_holiday'].item() == 1
    assert signature_df.loc[signature_df['idx'] == datetime(2023, 1, 1), 'holiday_name'].item() == "New Year's Day"


if __name__ == "__main__":
    pytest.main()

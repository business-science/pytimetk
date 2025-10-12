import pandas as pd
import polars as pl
import pytimetk as tk
import pytest
from pandas.testing import assert_frame_equal

@pytest.mark.parametrize('freq, regular, inferred_unit, median_timedelta, median_scale, median_unit',[
    ('D', False, 'D'     , '1 days'  , 1.0, 'D'),
    ('D', True,  'D'     , '1 days'  , 1.0, 'D'),
    ('W', False, 'W-SUN' , '7 days'  , 1.0, 'W'),
    ('W', True,  'W'     , '7 days'  , 1.0, 'W'),
    ('M', False, 'ME'     , '31 days' , 1.0, 'M'),
    ('M', True,  'ME'     , '31 days' , 1.0, 'M'),
    ('Q', False, 'QE-DEC' , '92 days' , 1.0, 'Q'),
    ('Q', True,  'QE-DEC', '92 days' , 1.0, 'Q'),
    ('Y', False, 'YE-DEC' , '365 days', 1.0, 'Y'),
    ('Y', True,  'YE-DEC', '365 days', 1.0, 'Y')
])

def test_correct_frequency_inference(
    freq, regular, inferred_unit, median_timedelta, median_scale, median_unit
):
    """
    The function tests if the inferred frequency of a datetime series is correct.    
    """

    # Create a sample datetime series with inferred frequency using freq parameter
    dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq=freq)

    for engine in ("pandas", "polars"):
        dates_input = pl.Series("date", dates) if engine == "polars" else dates

        result = tk.get_frequency_summary(
            dates_input,
            force_regular=regular,
            engine=engine,
        )

        expected_result = pd.DataFrame([
            {
                'freq_inferred_unit': inferred_unit,
                'freq_median_timedelta': median_timedelta,
                'freq_median_scale': median_scale,
                'freq_median_unit': median_unit,
            }
        ])
        expected_result['freq_median_timedelta'] = pd.to_timedelta(
            expected_result['freq_median_timedelta']
        )

        assert_frame_equal(result, expected_result)

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])

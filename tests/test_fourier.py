import pandas as pd
import pytimetk as tk


def test_engine_equivalence():
    
    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    by = ['date', 'id']

    # test pandas + DataFrame 
    output1 = df.augment_fourier(
        date_column='date',
        periods=[1, 2],
        max_order=1,
        reduce_memory=False,
        engine='pandas'
    ).sort_values(by=by).reset_index(drop=True)

    # test pandas + groupby
    output2 = df.groupby('id').augment_fourier(
        date_column='date',
        periods=[1, 2],
        max_order=1,
        reduce_memory=False,
        engine='pandas'
    ).sort_values(by=by).reset_index(drop=True)

    # test polars + DataFrame 
    output3 = df.augment_fourier(
        date_column='date',
        periods=[1, 2],
        max_order=1,
        reduce_memory=False,
        engine='polars'
    ).sort_values(by=by).reset_index(drop=True)

    # test polars + groupby
    output4 = df.groupby('id').augment_fourier(
        date_column='date',
        periods=[1, 2],
        max_order=1,
        reduce_memory=False,
        engine='polars'
    ).sort_values(by=by).reset_index(drop=True)

    pd.testing.assert_frame_equal(output1, output2)
    pd.testing.assert_frame_equal(output1, output3)
    pd.testing.assert_frame_equal(output1, output4)
    

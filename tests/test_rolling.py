import pytest
import pandas as pd
import numpy as np
from pytimetk import augment_rolling  
import pytimetk as tk

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
   
    result = df.augment_rolling(
        date_column='date', 
        value_column='value', 
        window=2, 
        window_func='mean'
    )
    
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [np.nan, 1.5, 2.5]
    
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_multi_window_multi_func():
    
    result = df.augment_rolling(
        date_column='date', 
        value_column='value', 
        window=[2, 3], 
        window_func=['mean', 'sum']
    )
    
    expected = df.copy()
    expected['value_rolling_mean_win_2'] = [np.nan, 1.5, 2.5]
    expected['value_rolling_sum_win_2'] = [np.nan, 3.0, 5.0]
    expected['value_rolling_mean_win_3'] = [np.nan, 1.5, 2.0]
    expected['value_rolling_sum_win_3'] = [np.nan, 3.0, 6.0]
    
    pd.testing.assert_frame_equal(result, expected)

def test_augment_rolling_custom_func():
    #custom_func = ('custom', lambda x: x.max() - x.min())
    
    result = df.augment_rolling(
        date_column='date', 
        value_column='value', 
        window=2, 
        window_func=[('custom', lambda x: x.max() - x.min())]
    )
    
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

def test_example_1():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = [2,7],  # Specifying multiple window sizes
                window_func = [
                    'mean',  # Built-in mean function
                    ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                ],
                threads = 1,  # Disabling parallel processing
                engine = 'pandas'  # Using pandas engine
            )
    )
    
    assert rolled_df.shape[1] == 7
    assert rolled_df.shape[0] == df.shape[0]
    
def test_example_1_parallel():
    # Set threads to 2
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = [2,7],  # Specifying multiple window sizes
                window_func = [
                    'mean',  # Built-in mean function
                    ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                ],
                threads = 2,  # Threads = 2
                engine = 'pandas'  # Using pandas engine
            )
    )
    
    assert rolled_df.shape[1] == 7
    assert rolled_df.shape[0] == df.shape[0]
    
def test_example_1_polars():
    # Set threads to 2
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date', 
                value_column = 'value', 
                window = [2,7],  # Specifying multiple window sizes
                window_func = [
                    'mean',  # Built-in mean function
                    # 'std'
                    ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                ],
                engine = 'polars'  # Using pandas engine
            )
    )
    
    assert rolled_df.shape[1] == 7
    assert rolled_df.shape[0] == df.shape[0]


def test_sort_pandas():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    rolled_df_pandas_fast = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'pandas',
            show_progress = False
        )
    
    result = rolled_df_pandas_fast.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_rolling_mean_win_20']) / result['adjusted']
    
    assert result['test'].mean() < 0.10
    

def test_sort_pandas_lambda():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    # stocks_df = stocks_df.sort_values('date').reset_index()
    
    rolled_df_pandas_fast = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = ('mean', lambda x: x.mean()),
            engine = 'pandas',
            show_progress = False
        )
    
    result = rolled_df_pandas_fast.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_rolling_mean_win_20']) / result['adjusted']
    
    assert result['test'].mean() < 0.10

def test_sort_pandas_parallel():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    rolled_df_pandas_fast = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'pandas',
            show_progress = False,
            threads = 2
        )
    
    result = rolled_df_pandas_fast.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_rolling_mean_win_20']) / result['adjusted']
    
    assert result['test'].mean() < 0.10

def test_sort_polars():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    rolled_df_pandas_fast = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_rolling(
            date_column = 'date',
            value_column = 'adjusted',
            window = [20],
            window_func = 'mean',
            engine = 'polars',
            show_progress = False
        )
    
    result = rolled_df_pandas_fast.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_rolling_mean_win_20']) / result['adjusted']
    
    assert result['test'].mean() < 0.10
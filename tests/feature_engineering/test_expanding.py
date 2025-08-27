import pytest
import pandas as pd
import numpy as np
import pytimetk as tk

from pytimetk.utils.polars_helpers import pl_quantile
from pytimetk.utils.pandas_helpers import pd_quantile

def test_example_1():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function,
                    ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function
                        
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
                threads = 1,  # Disable parallel processing
                show_progress = True,  # Display a progress bar
                )
    )
    
    assert expanded_df.shape[1] == 6
    assert expanded_df.shape[0] == df.shape[0]
    
def test_example_1_parallel():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function,
                    ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function
                        
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
                threads = 2,  
                show_progress = True,  # Display a progress bar
                )
    )
    
    assert expanded_df.shape[1] == 6
    assert expanded_df.shape[0] == df.shape[0]


def test_example_1_polars():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    
    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function,
                    ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function
                        
                ],
                min_periods = 1,
                engine = 'polars',  # Utilize pandas for the underlying computations
                threads = 1,  # Disable parallel processing
                show_progress = True,  # Display a progress bar
                )
    )
    
    assert expanded_df.shape[1] == 6
    assert expanded_df.shape[0] == df.shape[0]
    
def test_example_2():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in std function
                    ('quantile_75', pl_quantile(quantile=0.75)),  # Configurable with all parameters found in polars.Expr.rolling_quantile
                ],
                min_periods = 1,
                engine = 'polars',  # Utilize Polars for the underlying computations
            )
    )
    assert expanded_df.shape[1] == 6
    assert expanded_df.shape[0] == df.shape[0]
    
def test_example_3():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    
                    ('range', lambda x: x.max() - x.min()),  # Identity lambda function: can be slower, especially in Polars
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
            )
    )
    
    assert expanded_df.shape[1] == 4
    assert expanded_df.shape[0] == df.shape[0]
    
def test_example_3_parallel():
    
    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date', 
                value_column = 'value', 
                window_func = [
                    
                    ('range', lambda x: x.max() - x.min()),  # Identity lambda function: can be slower, especially in Polars
                ],
                min_periods = 2,
                engine = 'pandas',  # Utilize pandas for the underlying computations
            )
    )
    
    assert expanded_df.shape[1] == 4
    assert expanded_df.shape[0] == df.shape[0]
    

def test_sort_pandas():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_expanding(
            date_column = 'date',
            value_column = 'adjusted',
            window_func = 'max',
            engine = 'pandas',
            show_progress = False
        )
    
    result = result.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_expanding_max']) / result['adjusted']
    
    assert result['test'].mean() < 0.35
    

def test_sort_pandas_lambda():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_expanding(
            date_column = 'date',
            value_column = 'adjusted',
            window_func = ('max', lambda x: x.max()),
            engine = 'pandas',
            show_progress = False
        )
    
    result = result.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_expanding_max']) / result['adjusted']
    
    assert result['test'].mean() < 0.35

def test_sort_pandas_parallel():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_expanding(
            date_column = 'date',
            value_column = 'adjusted',
            window_func = 'max',
            engine = 'pandas',
            show_progress = False,
            threads = 2,
        )
    
    result = result.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_expanding_max']) / result['adjusted']
    
    assert result['test'].mean() < 0.35

def test_sort_polars():
    
    import pandas as pd
    import pytimetk as tk
    import numpy as np

    stocks_df = tk.load_dataset("stocks_daily")
    stocks_df['date'] = pd.to_datetime(stocks_df['date'])
    
    result = stocks_df[['symbol','date','adjusted']] \
        .groupby('symbol') \
        .augment_expanding(
            date_column = 'date',
            value_column = 'adjusted',
            window_func = 'max',
            engine = 'polars',
            show_progress = False
        )
    
    result = result.groupby('symbol').apply(lambda x: x.tail(1))
    
    result['test'] = np.abs(result['adjusted'] - result['adjusted_expanding_max']) / result['adjusted']
    
    assert result['test'].mean() < 0.35
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytimetk as tk

def test_01_parallel_apply_returns_series():
    # Test 1 - Single argument returns Series
    
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 3, 4, 5, 6]
    })

    grouped = df.groupby('A')
    
    result_1 = grouped.apply(lambda df: df['B'].sum())
    
    result_2 = tk.parallel_apply(grouped, lambda df: df['B'].sum(), show_progress=True, threads=2)
    
    assert_series_equal(result_1, result_2)
    

def test_02_parallel_apply_returns_dataframe():
    # Test 2 - Multiple arguments returns MultiIndex DataFrame
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'two', 'two', 'one', 'two'],
        'C': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        return pd.DataFrame({
            'sum': [group['C'].sum()],
            'mean': [group['C'].mean()]
        })

    grouped = df.groupby(['A', 'B'])
    
    result_1 = grouped.apply(calculate)
    
    result_2 = tk.parallel_apply(grouped, calculate, show_progress=True, threads=2)
    
    assert_frame_equal(result_1, result_2)

def test_03_parallel_slicing_dataframes():
    # Test 3 - Multiple arguments returns MultiIndex DataFrame
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'two', 'two', 'one', 'two'],
        'C': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        return group.head(2)

    grouped = df.groupby(['A', 'B'])
    
    result_1 = grouped.apply(calculate)
    
    result_2 = tk.parallel_apply(grouped, calculate, show_progress=True, threads=2)
    
    assert_frame_equal(result_1, result_2)

def test_04_parallel_slicing_dataframes():
    # Test 4 - Single Group returns MultiIndex DataFrame
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        return pd.DataFrame({
            'sum': [group['B'].sum()],
            'mean': [group['B'].mean()]
        })

    grouped = df.groupby(['A'])
    
    result_1 = grouped.apply(calculate)
    
    result_2 = tk.parallel_apply(grouped, calculate, show_progress=True, threads=2)
    
    assert_frame_equal(result_1, result_2)
    

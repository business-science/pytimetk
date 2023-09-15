import numpy as np
import pandas as pd
import pytest
import timetk

@pytest.fixture
def sumarize_by_time_data_test():
    """ A dataframe to test summarize_by_time function"""
    
    data = pd.DataFrame({
        'date': pd.date_range(start='1/1/2020', periods=60),
        'value': np.arange(1, 61, dtype=np.int64),
        'groups': ['Group_1', 'Group_2'] * 30
    })
    
    return data

def test_summarize_by_time_agg_functions(sumarize_by_time_data_test):
    """ Test if the aggreagation functions is working"""
    
    data = sumarize_by_time_data_test
    
    # test with one function    
    result = data.summarize_by_time(
        'date', 'value',
        agg_func = 'sum',
        rule = 'M'
    )
    
    expected = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31', '2020-02-29']),
        'value': [496, 1334]
    }).set_index('date')
    
    assert result.equals(expected), \
        'Aggregate with one function is not working!'

    
    # test with the functions as a list
    result = data.summarize_by_time(
        'date', 'value',
        agg_func = ['sum', 'mean'],
        rule = 'M'
    )
    
    expected = pd.DataFrame({
        'date': pd.to_datetime(['2020-01-31', '2020-02-29']),
        'sum': [496, 1334],
        'mean': [16.0, 46.0]
    }) \
        .set_index('date')
    multilevel_column = [('value', 'sum'), ('value', 'mean')]
    expected.columns = pd.MultiIndex.from_tuples(multilevel_column)    
    
    assert result.equals(expected), \
        'Aggregate with two functions as a list is not working!'
        

import pytest
import pandas as pd
import numpy as np
import pytimetk as tk 
from sklearn.linear_model import LinearRegression

import numpy.testing as npt

# Generate sample data and functions for testing
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

def generate_sample_data_1():
    return pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [2, 16, 20, 40, 41, 50],
    })

def generate_sample_data_2():
    return pd.DataFrame({
        'id': [1, 1, 1, 2, 2, 2],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
        'value1': [10, 20, 29, 42, 53, 59],
        'value2': [5, 16, 24, 35, 45, 58],
        'value3': [2, 3, 6, 9, 10, 13]
    })

def regression(df):
    model = LinearRegression()
    X = df[['value2', 'value3']]
    y = df['value1']
    model.fit(X, y)
    return pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])

# TESTS

def test_augment_rolling_apply():
    # Generate a sample dataset
    df = generate_data(num_groups=10, num_entries_per_group=365)

    # Apply the rolling function
    result = df.set_index(np.arange(50,3650+50)).groupby('group').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('mean', lambda df: df['value'].mean())]
    )

    # Test shape and values
    assert result.shape[0] == df.shape[0]
    assert 'rolling_mean_win_3' in result.columns
    npt.assert_array_equal(result.index.values, np.arange(50,3650+50))



def test_augment_rolling_apply_parallel():
    df = generate_data(num_groups=10, num_entries_per_group=365)

    result = df.groupby('group').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('mean', lambda x: x['value'].mean())],
        threads = 2,
    )
    
    assert result.shape[0] == df.shape[0]


def test_example_1():
    df = generate_sample_data_1()
    result = df.groupby('id').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],
        center=False,
        threads=1
    )
    assert 'rolling_corr_win_3' in result.columns

def test_example_1_parallel():
    df = generate_sample_data_1()
    result = df.groupby('id').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],
        center=False,
        threads=2
    )
    assert 'rolling_corr_win_3' in result.columns

def test_example_2():
    df = generate_sample_data_2()
    result = df.groupby('id').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('regression', regression)]
    ).dropna()

    wide_result = pd.concat(result['rolling_regression_win_3'].to_list(), axis=1).T
    combined_result = pd.concat([result.reset_index(drop=True), wide_result], axis=1)

    assert 'Intercept' in combined_result.columns
    assert 'Slope' in combined_result.columns


def test_example_2_parallel():
    df = generate_sample_data_2()
    result = df.groupby('id').augment_rolling_apply(
        date_column='date',
        window=3,
        window_func=[('regression', regression)],
        threads=2
    ).dropna()

    wide_result = pd.concat(result['rolling_regression_win_3'].to_list(), axis=1).T
    combined_result = pd.concat([result.reset_index(drop=True), wide_result], axis=1)

    assert 'Intercept' in combined_result.columns
    assert 'Slope' in combined_result.columns
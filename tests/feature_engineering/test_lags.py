import pandas as pd
import polars as pl
import pytest

# Ensure the function and dependencies are imported
import pytimetk as tk

# Sample data for testing
@pytest.fixture
def df_sample():
    df_sample = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=5),
        'value': [1, 2, 3, 4, 5],
        'id': ['A', 'A', 'A', 'B', 'B']
    })

    return df_sample


@pytest.fixture
def pl_df_sample(df_sample):
    return pl.from_pandas(df_sample)

# Basic Functionality Test
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_basic_functionality(df_sample, engine):
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=1, engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lag_1'])
    assert df_result['value_lag_1'].iloc[1] == 1

# GroupBy Functionality Test
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_functionality(df_sample, engine):
    df_result = df_sample.groupby('id').augment_lags(date_column='date', value_column='value', lags=1, engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lag_1'])
    assert pd.isna(df_result['value_lag_1'].iloc[3])

# Multiple Lags Test
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_multiple_lags(df_sample, engine):
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=(1, 2), engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lag_1', 'value_lag_2'])
    assert df_result['value_lag_1'].iloc[2] == 2
    assert df_result['value_lag_2'].iloc[2] == 1

# Grouped Dataframe with Multiple Lags Test
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_multiple_lags(df_sample, engine):
    df_result = df_sample.groupby('id').augment_lags(date_column='date', value_column='value', lags=(1, 2), engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lag_1', 'value_lag_2'])
    assert df_result['value_lag_1'].iloc[2] == 2
    assert df_result['value_lag_2'].iloc[2] == 1

# Invalid Lags Type Test
def test_invalid_lags_type(df_sample):
    with pytest.raises(TypeError):
        df_sample.augment_lags(date_column='date', value_column='value', lags='string')

def test_invalid_dataframe_or_groupby_input():
    invalid_data = {"key": "value"}
    with pytest.raises(TypeError, match="`data` is not a Pandas DataFrame or GroupBy object."):
        tk.augment_lags(data=invalid_data, date_column="date", value_column="value", lags=1)

# Value Column as List Test
def test_value_column_list(df_sample):
    df_result = df_sample.augment_lags(date_column='date', value_column=['value'], lags=1)
    assert 'value_lag_1' in df_result.columns

# Lags as List Test
def test_lags_list(df_sample):
    df_result = df_sample.augment_lags(date_column='date', value_column='value', lags=[1, 3])
    assert 'value_lag_1' in df_result.columns
    assert 'value_lag_3' in df_result.columns
    assert df_result['value_lag_1'].iloc[3] == 3
    assert df_result['value_lag_3'].iloc[3] == 1
    
def test_lags_string():
    
    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    
    df_result = df.augment_lags("date", "id", lags = (1,7))
    
    assert "id_lag_1" in df_result.columns


def test_polars_dataframe_roundtrip(df_sample, pl_df_sample):
    pandas_result = df_sample.augment_lags(
        date_column="date", value_column="value", lags=(1, 2)
    )
    polars_result = tk.augment_lags(
        data=pl_df_sample, date_column="date", value_column="value", lags=(1, 2)
    )

    assert isinstance(polars_result, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_result, polars_result.to_pandas())


def test_polars_groupby_roundtrip(df_sample, pl_df_sample):
    pandas_group = df_sample.groupby("id").augment_lags(
        date_column="date", value_column="value", lags=(1, 2)
    )
    polars_group = tk.augment_lags(
        data=pl_df_sample.group_by("id"),
        date_column="date",
        value_column="value",
        lags=(1, 2),
    )

    assert isinstance(polars_group, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_group, polars_group.to_pandas())


if __name__ == "__main__":
    pytest.main()

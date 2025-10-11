import pandas as pd
import polars as pl
import pytest
import numpy as np
from pytimetk import augment_leads  
import pytimetk as tk

# Sample Data for testing
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

@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_single_lead(df_sample, engine):
    df_result = augment_leads(df_sample, date_column='date', value_column='value', leads=1, engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lead_1'])
    assert np.array_equal(df_result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, 4.0, 5.0, np.nan]), equal_nan=True)

@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_multiple_leads(df_sample, engine):
    df_result = augment_leads(df_sample, date_column='date', value_column='value', leads=[1, 2], engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lead_1', 'value_lead_2'])
    assert np.array_equal(df_result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, 4.0, 5.0, np.nan]), equal_nan=True)
    assert np.array_equal(df_result['value_lead_2'].to_numpy() , np.array([3.0, 4.0, 5.0, np.nan, np.nan]), equal_nan=True)

@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_single_lead(df_sample, engine):
    grouped = df_sample.groupby('id')
    df_result = augment_leads(grouped, date_column='date', value_column='value', leads=1, engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lead_1'])
    assert np.array_equal(df_result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, np.nan, 5.0, np.nan]), equal_nan=True)

@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_groupby_multiple_leads(df_sample, engine):
    df_result = df_sample.groupby('id').augment_leads(date_column='date', value_column='value', leads=(1, 2), engine = engine)
    assert all(df_result.columns == ['date', 'value', 'id', 'value_lead_1', 'value_lead_2'])
    assert np.array_equal(df_result['value_lead_1'].to_numpy() , np.array([2.0, 3.0, np.nan, 5.0, np.nan]), equal_nan=True)
    assert np.array_equal(df_result['value_lead_2'].to_numpy() , np.array([3.0, np.nan, np.nan, np.nan, np.nan]), equal_nan=True)

def test_invalid_dataframe_or_groupby_input():
    invalid_data = {"key": "value"}
    with pytest.raises(TypeError, match="`data` is not a Pandas DataFrame or GroupBy object."):
        augment_leads(data=invalid_data, date_column="date", value_column="value", leads=1)

def test_invalid_leads_type(df_sample):
    with pytest.raises(TypeError):
        df_sample.augment_lags(date_column='date', value_column='value', leads='string')

def test_lags_string():
    
    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    
    df_result = df.augment_leads("date", "id", leads = (1,7))
    
    assert "id_lead_1" in df_result.columns


def test_polars_dataframe_roundtrip(df_sample, pl_df_sample):
    pandas_result = augment_leads(
        df_sample, date_column="date", value_column="value", leads=[1, 2]
    )
    polars_result = augment_leads(
        pl_df_sample, date_column="date", value_column="value", leads=[1, 2]
    )

    assert isinstance(polars_result, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_result, polars_result.to_pandas())


def test_polars_groupby_roundtrip(df_sample, pl_df_sample):
    pandas_group = augment_leads(
        df_sample.groupby("id"),
        date_column="date",
        value_column="value",
        leads=[1, 2],
    )
    polars_group = augment_leads(
        pl_df_sample.group_by("id"),
        date_column="date",
        value_column="value",
        leads=[1, 2],
    )

    assert isinstance(polars_group, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_group, polars_group.to_pandas())

if __name__ == "__main__":
    pytest.main()

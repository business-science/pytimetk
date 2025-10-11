import numpy as np
import pandas as pd
import polars as pl
import pytest

import pytimetk as tk


@pytest.fixture
def df_sample():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=6, freq="D"),
            "value": [10, 12, 15, 18, 21, 25],
            "id": ["A", "A", "A", "B", "B", "B"],
        }
    )
    return df


@pytest.fixture
def pl_df_sample(df_sample):
    return pl.from_pandas(df_sample)


def test_augment_diffs_dataframe_parity(df_sample, pl_df_sample):
    pandas_result = df_sample.augment_diffs(
        date_column="date", value_column="value", periods=[1, 2]
    )
    polars_result = tk.augment_diffs(
        data=pl_df_sample, date_column="date", value_column="value", periods=[1, 2]
    )

    assert isinstance(polars_result, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_result, polars_result.to_pandas())


def test_augment_diffs_groupby_parity(df_sample, pl_df_sample):
    pandas_group = df_sample.groupby("id").augment_diffs(
        date_column="date", value_column="value", periods=[1, 2]
    )
    polars_group = tk.augment_diffs(
        data=pl_df_sample.group_by("id"),
        date_column="date",
        value_column="value",
        periods=[1, 2],
    )

    assert isinstance(polars_group, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_group, polars_group.to_pandas())


def test_augment_diffs_normalize(df_sample, pl_df_sample):
    pandas_result = tk.augment_diffs(
        data=df_sample,
        date_column="date",
        value_column="value",
        periods=[1],
        normalize=True,
    )
    polars_result = tk.augment_diffs(
        data=pl_df_sample,
        date_column="date",
        value_column="value",
        periods=[1],
        normalize=True,
    )

    pd.testing.assert_frame_equal(pandas_result, polars_result.to_pandas())
    expected = np.array([np.nan, 0.2, 0.25, 0.2, 0.16666667, 0.19047619])
    np.testing.assert_allclose(
        pandas_result["value_pctdiff_1"].to_numpy(),
        expected,
        rtol=1e-6,
        equal_nan=True,
    )


def test_augment_pct_change_dataframe_parity(df_sample, pl_df_sample):
    pandas_pct = tk.augment_pct_change(
        data=df_sample, date_column="date", value_column="value", periods=[1, 2]
    )
    polars_pct = tk.augment_pct_change(
        data=pl_df_sample, date_column="date", value_column="value", periods=[1, 2]
    )

    assert isinstance(polars_pct, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_pct, polars_pct.to_pandas())


def test_augment_pct_change_groupby_parity(df_sample, pl_df_sample):
    pandas_group = tk.augment_pct_change(
        data=df_sample.groupby("id"),
        date_column="date",
        value_column="value",
        periods=[1],
    )
    polars_group = tk.augment_pct_change(
        data=pl_df_sample.group_by("id"),
        date_column="date",
        value_column="value",
        periods=[1],
    )

    assert isinstance(polars_group, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_group, polars_group.to_pandas())

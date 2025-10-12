import pytest
import pandas as pd
import numpy as np
import pytimetk as tk
import polars as pl

# noqa: F401
from sklearn.linear_model import LinearRegression

import numpy.testing as npt


def generate_sample_data_1():
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [2, 16, 20, 40, 41, 50],
        }
    )


def generate_sample_data_2():
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [5, 16, 24, 35, 45, 58],
            "value3": [2, 3, 6, 9, 10, 13],
        }
    )


def regression(df):
    model = LinearRegression()
    X = df[["value2", "value3"]]
    y = df["value1"]
    model.fit(X, y)
    return pd.Series([model.intercept_, model.coef_[0]], index=["Intercept", "Slope"])


def test_example_1():
    df = generate_sample_data_1()
    result = df.groupby("id").augment_expanding_apply(
        date_column="date",
        window_func=[("corr", lambda x: x["value1"].corr(x["value2"]))],
        threads=1,
    )
    assert "expanding_corr" in result.columns


def test_example_parallel():
    df = generate_sample_data_1()
    result = df.groupby("id").augment_expanding_apply(
        date_column="date",
        window_func=[("corr", lambda x: x["value1"].corr(x["value2"]))],
        threads=2,
    )
    assert "expanding_corr" in result.columns


def test_example_2():
    df = generate_sample_data_2()
    result_df = (
        df.groupby("id")
        .augment_expanding_apply(
            date_column="date", window_func=[("regression", regression)], threads=1
        )
        .dropna()
    )

    regression_wide_df = pd.concat(
        result_df["expanding_regression"].to_list(), axis=1
    ).T
    regression_wide_df = pd.concat(
        [result_df.reset_index(drop=True), regression_wide_df], axis=1
    )

    assert "Intercept" in regression_wide_df.columns
    assert "Slope" in regression_wide_df.columns


def test_example_2_parallel():
    df = generate_sample_data_2()
    result_df = (
        df.groupby("id")
        .augment_expanding_apply(
            date_column="date", window_func=[("regression", regression)], threads=2
        )
        .dropna()
    )

    regression_wide_df = pd.concat(
        result_df["expanding_regression"].to_list(), axis=1
    ).T
    regression_wide_df = pd.concat(
        [result_df.reset_index(drop=True), regression_wide_df], axis=1
    )

    assert "Intercept" in regression_wide_df.columns
    assert "Slope" in regression_wide_df.columns


def test_polars_accessor():
    df = generate_sample_data_1()
    result = (
        pl.from_pandas(df)
        .group_by("id")
        .tk.augment_expanding_apply(
            date_column="date",
            window_func=[("corr", lambda x: x["value1"].corr(x["value2"]))],
        )
    )
    assert "expanding_corr" in result.columns

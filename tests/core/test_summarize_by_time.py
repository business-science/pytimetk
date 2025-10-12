import numpy as np
import pandas as pd
import polars as pl
import pytest
import pytimetk
# noqa: F401


@pytest.fixture
def summarize_by_time_data_test():
    """The function `summarize_by_time_data_test` creates a pandas DataFrame with date, value, and groups columns.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame object named "data".

    """

    data = pd.DataFrame(
        {
            "date": pd.date_range(start="1/1/2020", periods=60),
            "value": np.arange(1, 61, dtype=np.int64),
            "groups": ["Group_1", "Group_2"] * 30,
        }
    )

    return data


def test_summarize_by_time_dataframe_functions(summarize_by_time_data_test):
    """Tests the `summarize_by_time` function using DataFrames.

    Parameters
    ----------
    summarize_by_time_data_test
        The `summarize_by_time_data_test` parameter is a DataFrame that contains the data to be summarized. See summarize_by_time_data_test()

    """

    data = summarize_by_time_data_test

    # test with one function
    result = data.summarize_by_time(
        "date",
        "value",
        agg_func="sum",
        freq="M",
    )

    expected = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "value": [496, 1334]}
    )

    assert result.equals(expected), (
        "Summarize by time with one function is not working!"
    )

    # test with flatten_column_names = True and reset_index = True
    result = data.summarize_by_time(
        "date",
        "value",
        agg_func="sum",
        freq="M",
    )

    expected = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "value": [496, 1334]}
    )

    assert result.equals(expected), (
        "Summarize by time with flatten_column_names is not working!"
    )

    # test with the functions as a list
    result = data.summarize_by_time(
        "date",
        "value",
        agg_func=["sum", "mean"],
        freq="M",
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "value_sum": [496, 1334],
            "value_mean": [16.0, 46.0],
        }
    )

    assert result.equals(expected), (
        "Summarize by time with two functions as a list is not working!"
    )


def test_summarize_by_time_grouped_functions(summarize_by_time_data_test):
    """Tests the `summarize_by_time` function with GroupBy objects.

    Parameters
    ----------
    summarize_by_time_data_test
        The parameter `summarize_by_time_data_test` is a DataFrame that contains the data to be summarized. See summarize_by_time_data_test()
    """

    data = summarize_by_time_data_test

    # Test groupby objects
    result = data.groupby("groups").summarize_by_time(
        "date",
        "value",
        freq="MS",
        agg_func="sum",
        wide_format=True,
    )

    idx = pd.DatetimeIndex(
        ["2020-01-01", "2020-02-01"], dtype="datetime64[ns]", name="date", freq="MS"
    )

    expected = pd.DataFrame(
        [[256, 240], [644, 690]], index=idx, columns=["value_Group_1", "value_Group_2"]
    ).reset_index()

    assert result.equals(expected), (
        "Summarize by time with grouped objects is not working!"
    )


def test_summarize_by_time_lambda_functions(summarize_by_time_data_test):
    data = summarize_by_time_data_test

    result = data.groupby("groups").summarize_by_time(
        date_column="date",
        value_column="value",
        freq="MS",
        agg_func=["sum", ("q25", lambda x: x.quantile(0.25))],
        wide_format=True,
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "value_sum_Group_1": [256, 644],
            "value_sum_Group_2": [240, 690],
            "value_q25_Group_1": [8.5, 39.5],
            "value_q25_Group_2": [9.0, 39.0],
        }
    )

    assert result.equals(expected), (
        "Summarize by time with lambda functions is not working!"
    )


def test_summarize_by_time_polars_dataframe(summarize_by_time_data_test):
    pl_df = pl.from_pandas(summarize_by_time_data_test)

    result = pytimetk.summarize_by_time(
        data=pl_df,
        date_column="date",
        value_column="value",
        agg_func="sum",
        freq="M",
        engine="polars",
    )

    expected = pd.DataFrame(
        {"date": pd.to_datetime(["2020-01-31", "2020-02-29"]), "value": [496, 1334]}
    )

    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_summarize_by_time_polars_grouped_wide(summarize_by_time_data_test):
    pl_df = pl.from_pandas(summarize_by_time_data_test)

    result = pytimetk.summarize_by_time(
        data=pl_df.group_by("groups"),
        date_column="date",
        value_column="value",
        freq="MS",
        agg_func="sum",
        wide_format=True,
        engine="polars",
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "value_Group_1": [256, 644],
            "value_Group_2": [240, 690],
        }
    )

    pd.testing.assert_frame_equal(result.to_pandas(), expected)


def test_summarize_by_time_polars_disallows_callables(summarize_by_time_data_test):
    pl_df = pl.from_pandas(summarize_by_time_data_test)

    with pytest.raises(ValueError):
        pytimetk.summarize_by_time(
            data=pl_df,
            date_column="date",
            value_column="value",
            freq="MS",
            agg_func=["sum", ("q25", lambda x: x.quantile(0.25))],
            engine="polars",
        )


def test_summarize_by_time_polars_accessor(summarize_by_time_data_test):
    pl_df = pl.from_pandas(summarize_by_time_data_test)

    result = pl_df.tk.summarize_by_time(
        date_column="date",
        value_column="value",
        freq="MS",
        agg_func="sum",
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
            "value": [496, 1334],
        }
    )

    pd.testing.assert_frame_equal(result.to_pandas(), expected)

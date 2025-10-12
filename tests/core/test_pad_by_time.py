import pandas as pd
import pytest
import numpy as np
import polars as pl
from pandas.testing import assert_frame_equal
# noqa: F401

# Import the pad_by_time function from your module
from pytimetk import pad_by_time

data = {
    "date": pd.date_range("2022-01-01", periods=5, freq="D"),
    "value": [1, 2, 3, 4, 5],
}
df = pd.DataFrame(data)


# Define a test DataFrame for testing
@pytest.fixture
def test_dataframe():
    data = {
        "date": pd.date_range("2022-01-01", periods=5, freq="D"),
        "value": [1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(data)
    return df


# Test the pad_by_time function
def test_pad_by_time_single_series(test_dataframe):
    # Apply pad_by_time to a single series
    padded_df = test_dataframe.pad_by_time(date_column="date", freq="D")

    # Define the expected result
    expected_data = {
        "date": pd.date_range("2022-01-01", "2022-01-05", freq="D"),
        "value": [1, 2, 3, 4, 5],
    }
    expected_df = pd.DataFrame(expected_data)

    # Check if the result matches the expected DataFrame
    assert_frame_equal(padded_df, expected_df, check_dtype=False)


data2 = {
    "date": pd.date_range("2022-01-01", periods=6, freq="D"),
    "value": [1, 2, 3, 4, 5, 6],
}
df2 = pd.DataFrame(data2)
grouped_df = df2.copy()
grouped_df["group"] = ["A", "B", "A", "B", "B", "A"]


# Apply pad_by_time to the grouped DataFrame
def test_pad_by_time_grouped(test_dataframe):
    # Create a grouped DataFrame for testing
    # grouped_df = test_dataframe.copy()
    grouped_df["group"] = ["A", "B", "A", "B", "B", "A"]
    # grouped_df = grouped_df.groupby("group")

    # Apply pad_by_time to the grouped DataFrame
    padded_df = grouped_df.groupby("group").pad_by_time(date_column="date", freq="D")

    # Define the expected result for each group
    expected_data = {
        "date": [
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
        ],
        "value": [1, np.nan, 3, np.nan, np.nan, 6, np.nan, 2, np.nan, 4, 5, np.nan],
        "group": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    # Check if the result matches the expected DataFrame
    assert_frame_equal(padded_df, expected_df, check_dtype=True)


def test_pad_by_time_grouped_end(test_dataframe):
    # Create a grouped DataFrame for testing
    # grouped_df = test_dataframe.copy()
    grouped_df["group"] = ["A", "B", "A", "B", "B", "A"]
    # grouped_df = grouped_df.groupby("group")

    # Apply pad_by_time to the grouped DataFrame
    padded_df = grouped_df.groupby("group").pad_by_time(
        date_column="date", freq="D", end_date="2022-01-06"
    )

    # Define the expected result for each group
    expected_data = {
        "date": [
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
            "2022-01-01",
            "2022-01-02",
            "2022-01-03",
            "2022-01-04",
            "2022-01-05",
            "2022-01-06",
        ],
        "value": [1, np.nan, 3, np.nan, np.nan, 6, np.nan, 2, np.nan, 4, 5, np.nan],
        "group": ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "B"],
    }
    expected_df = pd.DataFrame(expected_data)
    expected_df["date"] = pd.to_datetime(expected_df["date"])
    # Check if the result matches the expected DataFrame
    assert_frame_equal(padded_df, expected_df, check_dtype=False)


# def test_pad_by_time_auto_freq(test_dataframe):
#     # Apply pad_by_time with auto frequency detection
#     padded_df = (
#         test_dataframe
#         .pad_by_time(date_column="date", freq="auto")
#     )

#     # Define the expected result
#     expected_data = {
#         "date": pd.date_range("2022-01-01", "2022-01-05", freq="D"),
#         "value": [1, 2, 3, 4, 5],
#     }
#     expected_df = pd.DataFrame(expected_data)

#     # Check if the result matches the expected DataFrame
#     assert_frame_equal(padded_df, expected_df, check_dtype=False)

# Add more test cases as needed


def test_pad_by_time_polars_accessor():
    sample = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-01", "2022-01-03"]),
            "value": [1, 3],
        }
    )

    pl_df = pl.from_pandas(sample)

    result = pl_df.tk.pad_by_time(date_column="date", freq="D")

    assert isinstance(result, pl.DataFrame)

    expected = sample.pad_by_time(date_column="date", freq="D")
    result_pd = result.to_pandas()

    assert_frame_equal(result_pd, expected, check_dtype=False)


if __name__ == "__main__":
    pytest.main()

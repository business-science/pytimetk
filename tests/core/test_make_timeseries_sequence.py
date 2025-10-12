import pandas as pd
import polars as pl
import pytest

import pytimetk as tk


def test_make_weekday_sequence_pandas():
    result = tk.make_weekday_sequence("2023-01-01", "2023-01-07", engine="pandas")
    assert isinstance(result, pd.Series)
    expected = pd.Series(
        pd.to_datetime(
            ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
        ),
        name="Weekday Dates",
    )
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_make_weekday_sequence_polars():
    result = tk.make_weekday_sequence("2023-01-01", "2023-01-07", engine="polars")
    assert isinstance(result, pl.Series)
    expected = [
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-03"),
        pd.Timestamp("2023-01-04"),
        pd.Timestamp("2023-01-05"),
        pd.Timestamp("2023-01-06"),
    ]
    assert result.to_list() == expected


def test_make_weekday_sequence_remove_holiday():
    result = tk.make_weekday_sequence(
        "2023-07-03",
        "2023-07-05",
        remove_holidays=True,
        country="UnitedStates",
        engine="pandas",
    )
    expected = pd.Series(
        pd.to_datetime(["2023-07-03", "2023-07-05"]), name="Weekday Dates"
    )
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_make_weekend_sequence_default():
    result = tk.make_weekend_sequence("2023-01-01", "2023-01-08", engine="pandas")
    assert isinstance(result, pd.Series)
    expected = pd.Series(
        pd.to_datetime(["2023-01-01", "2023-01-07", "2023-01-08"]), name="Weekend Dates"
    )
    pd.testing.assert_series_equal(result, expected, check_dtype=False)


def test_make_weekend_sequence_friday_saturday_polars():
    result = tk.make_weekend_sequence(
        "2023-01-01",
        "2023-01-08",
        friday_saturday=True,
        engine="polars",
    )
    assert isinstance(result, pl.Series)
    expected = [
        pd.Timestamp("2023-01-06"),
        pd.Timestamp("2023-01-07"),
    ]
    assert result.to_list() == expected

import pytest
import pandas as pd
import polars as pl

from pytimetk import get_timeseries_signature, augment_timeseries_signature


def test_get_timeseries_signature():
    # Create a sample date range
    dates = pd.date_range(start="2019-01", end="2019-03", freq="D")

    # Test if function produces the correct number of features
    signature = get_timeseries_signature(dates)
    assert signature.shape[1] == 30, "Number of features generated is not correct."

    # Test if the function produces the correct feature names
    expected_columns = [
        "idx",
        "idx_index_num",
        "idx_year",
        "idx_year_iso",
        "idx_yearstart",
        "idx_yearend",
        "idx_leapyear",
        "idx_half",
        "idx_quarter",
        "idx_quarteryear",
        "idx_quarterstart",
        "idx_quarterend",
        "idx_month",
        "idx_month_lbl",
        "idx_monthstart",
        "idx_monthend",
        "idx_yweek",
        "idx_mweek",
        "idx_wday",
        "idx_wday_lbl",
        "idx_mday",
        "idx_qday",
        "idx_yday",
        "idx_weekend",
        "idx_hour",
        "idx_minute",
        "idx_second",
        "idx_msecond",
        "idx_nsecond",
        "idx_am_pm",
    ]
    assert all(signature.columns == expected_columns), (
        "Feature names are not as expected."
    )

    # Test if function raises TypeError for incorrect input types
    with pytest.raises(TypeError):
        get_timeseries_signature([1, 2, 3])


def test_augment_timeseries_signature():
    # Sample DataFrame
    df = pd.DataFrame(
        {
            "order_date": pd.date_range(start="2019-01", end="2019-03", freq="D"),
            "value": range(60),
        }
    )

    # Test if function adds the correct number of features
    augmented = df.augment_timeseries_signature(date_column="order_date")
    assert augmented.shape[1] == df.shape[1] + 29, (
        "Number of features in the augmented dataframe is not correct."
    )

    # Test if original dataframe data remains the same
    assert all(df["value"] == augmented["value"]), (
        "Original data in the augmented dataframe has changed."
    )

    expected_columns = [
        "order_date",
        "value",
        "order_date_index_num",
        "order_date_year",
        "order_date_year_iso",
        "order_date_yearstart",
        "order_date_yearend",
        "order_date_leapyear",
        "order_date_half",
        "order_date_quarter",
        "order_date_quarteryear",
        "order_date_quarterstart",
        "order_date_quarterend",
        "order_date_month",
        "order_date_month_lbl",
        "order_date_monthstart",
        "order_date_monthend",
        "order_date_yweek",
        "order_date_mweek",
        "order_date_wday",
        "order_date_wday_lbl",
        "order_date_mday",
        "order_date_qday",
        "order_date_yday",
        "order_date_weekend",
        "order_date_hour",
        "order_date_minute",
        "order_date_second",
        "order_date_msecond",
        "order_date_nsecond",
        "order_date_am_pm",
    ]
    assert all(augmented.columns == expected_columns), (
        "Feature names are not as expected."
    )

    # Test if the function raises KeyError for non-existent column
    with pytest.raises(ValueError):
        df.augment_timeseries_signature(date_column="nonexistent_column")


def test_augment_timeseries_signature_polars():
    # Sample DataFrame
    df = pd.DataFrame(
        {
            "order_date": pd.date_range(start="2019-01", end="2019-03", freq="D"),
            "value": range(60),
        }
    )

    # Test if function adds the correct number of features
    augmented = df.augment_timeseries_signature(
        date_column="order_date", engine="polars"
    )

    assert augmented.shape[1] == df.shape[1] + 29, (
        "Number of features in the augmented dataframe is not correct."
    )

    # Test if original dataframe data remains the same
    assert all(df["value"] == augmented["value"]), (
        "Original data in the augmented dataframe has changed."
    )

    expected_columns = [
        "order_date",
        "value",
        "order_date_index_num",
        "order_date_year",
        "order_date_year_iso",
        "order_date_yearstart",
        "order_date_yearend",
        "order_date_leapyear",
        "order_date_half",
        "order_date_quarter",
        "order_date_quarteryear",
        "order_date_quarterstart",
        "order_date_quarterend",
        "order_date_month",
        "order_date_month_lbl",
        "order_date_monthstart",
        "order_date_monthend",
        "order_date_yweek",
        "order_date_mweek",
        "order_date_wday",
        "order_date_wday_lbl",
        "order_date_mday",
        "order_date_qday",
        "order_date_yday",
        "order_date_weekend",
        "order_date_hour",
        "order_date_minute",
        "order_date_second",
        "order_date_msecond",
        "order_date_nsecond",
        "order_date_am_pm",
    ]
    assert all(augmented.columns == expected_columns)

    # Test if the function raises KeyError for non-existent column
    with pytest.raises(ValueError):
        df.augment_timeseries_signature(date_column="nonexistent_column")


def test_augment_timeseries_signature_polars_accessor():
    df = pd.DataFrame(
        {
            "order_date": pd.date_range(start="2019-01", end="2019-01-10", freq="D"),
            "value": range(10),
        }
    )

    pl_result = pl.from_pandas(df).tk.augment_timeseries_signature(
        date_column="order_date"
    )

    assert "order_date_index_num" in pl_result.columns
    assert pl_result.height == len(df)


if __name__ == "__main__":
    pytest.main()

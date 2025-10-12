import pytest
import pandas as pd
import polars as pl
import pytimetk as tk
# noqa: F401

from tsfeatures import acf_features, series_length, hurst


@pytest.fixture
def data_frame_to_test() -> pd.DataFrame:
    """The function  creates a pandas DataFrame with date and value
    for tests.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame.
    """
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="1/1/2020", periods=10),
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )

    return data


@pytest.fixture
def grouped_data_frame_to_test() -> pd.DataFrame:
    """The function loads m4_daily pandas DataFrame to make the
    tests with grouped dataframe.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame.
    """
    data = tk.load_dataset("m4_daily", parse_dates=["date"])

    return data


def test_ts_features_dataframe_with_all_features(data_frame_to_test):
    # Load data frame
    df = data_frame_to_test

    # Call the ts_features function
    result = tk.ts_features(
        df, "date", "value", features=[acf_features, series_length, hurst]
    )

    # Assert the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the result has the correct columns
    expected_columns = [
        "hurst",
        "series_length",
        "x_acf1",
        "x_acf10",
        "diff1_acf1",
        "diff1_acf10",
        "diff2_acf1",
        "diff2_acf10",
    ]

    assert result.columns.tolist() == expected_columns

    # Assert if was generated one row
    assert result.shape[0] == 1


def test_ts_features_dataframe_specifying_features(data_frame_to_test):
    # Load data frame
    df = data_frame_to_test

    # Call the ts_features function
    result = tk.ts_features(df, "date", "value", features=[hurst, series_length])

    # Assert the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the result has the correct columns
    expected_columns = ["series_length", "hurst"]

    assert result.columns.tolist() == expected_columns

    # Assert if was generated one row
    assert result.shape[0] == 1


def test_ts_features_grouped_dataframe_with_all_features(grouped_data_frame_to_test):
    # Load data frame with groups
    df = grouped_data_frame_to_test

    # Call the ts_features function
    result = df.groupby("id").ts_features(date_column="date", value_column="value")

    # Assert the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the result has the correct columns
    expected_columns = [
        "id",
        "hurst",
        "series_length",
        "unitroot_pp",
        "unitroot_kpss",
        "hw_alpha",
        "hw_beta",
        "hw_gamma",
        "stability",
        "nperiods",
        "seasonal_period",
        "trend",
        "spike",
        "linearity",
        "curvature",
        "e_acf1",
        "e_acf10",
        "x_pacf5",
        "diff1x_pacf5",
        "diff2x_pacf5",
        "nonlinearity",
        "lumpiness",
        "alpha",
        "beta",
        "arch_acf",
        "garch_acf",
        "arch_r2",
        "garch_r2",
        "flat_spots",
        "entropy",
        "crossing_points",
        "arch_lm",
        "x_acf1",
        "x_acf10",
        "diff1_acf1",
        "diff1_acf10",
        "diff2_acf1",
        "diff2_acf10",
    ]

    assert result.columns.tolist() == expected_columns

    # Assert if was generated four rows
    assert result.shape[0] == 4


def test_ts_features_grouped_dataframe_specifying_features(grouped_data_frame_to_test):
    # Load data frame with groups
    df = grouped_data_frame_to_test

    # Call the ts_features function
    result = df.groupby("id").ts_features(
        date_column="date",
        value_column="value",
        features=[acf_features, hurst, series_length],
    )

    # Assert the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the result has the correct columns
    expected_columns = [
        "id",
        "series_length",
        "hurst",
        "x_acf1",
        "x_acf10",
        "diff1_acf1",
        "diff1_acf10",
        "diff2_acf1",
        "diff2_acf10",
    ]

    assert result.columns.tolist() == expected_columns

    # Assert if was generated four rows
    assert result.shape[0] == 4


def test_ts_features_grouped_dataframe_parallel_processing(grouped_data_frame_to_test):
    # Load data frame with groups
    df = grouped_data_frame_to_test

    # Call the ts_features function
    result = df.groupby("id").ts_features(
        date_column="date",
        value_column="value",
        features=[acf_features, hurst, series_length],
        threads=2,
    )

    # Assert the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert the result has the correct columns
    expected_columns = [
        "id",
        "series_length",
        "hurst",
        "x_acf1",
        "x_acf10",
        "diff1_acf1",
        "diff1_acf10",
        "diff2_acf1",
        "diff2_acf10",
    ]

    assert result.columns.tolist() == expected_columns

    # Assert if was generated four rows
    assert result.shape[0] == 4


def test_ts_features_polars_accessor(data_frame_to_test):
    pl_df = pl.from_pandas(data_frame_to_test)

    result = pl_df.tk.ts_features(
        date_column="date",
        value_column="value",
        features=[hurst, series_length],
        show_progress=False,
    )

    assert isinstance(result, pl.DataFrame)

    result_pd = result.to_pandas()

    assert result_pd.loc[0, "series_length"] == len(data_frame_to_test)
    assert "hurst" in result_pd.columns


if __name__ == "__main__":
    pytest.main([__file__])

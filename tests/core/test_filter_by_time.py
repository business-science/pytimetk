import pandas as pd
import polars as pl
import pytimetk as tk
# noqa: F401


def _sample_df():
    return pd.DataFrame(
        {
            "id": ["A", "A", "B", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-07",
                    "2023-01-09",
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )


def test_filter_by_time_pandas_dataframe():
    df = _sample_df()
    result = tk.filter_by_time(
        data=df,
        date_column="date",
        start_date="2023-01-02",
        end_date="2023-01-07",
        engine="pandas",
    )
    expected = df[(df["date"] >= "2023-01-02") & (df["date"] <= "2023-01-07")]
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_filter_by_time_pandas_groupby():
    df = _sample_df()
    grouped = df.groupby("id")
    result = tk.filter_by_time(
        data=grouped,
        date_column="date",
        start_date="2023-01-02",
        end_date="2023-01-07",
        engine="pandas",
    )
    expected = grouped.obj[
        (grouped.obj["date"] >= "2023-01-02") & (grouped.obj["date"] <= "2023-01-07")
    ]
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_filter_by_time_polars_dataframe():
    df = _sample_df()
    pl_df = pl.from_pandas(df)
    result = tk.filter_by_time(
        data=pl_df,
        date_column="date",
        start_date="2023-01-02",
        end_date="2023-01-07",
        engine="polars",
    )
    expected = df[(df["date"] >= "2023-01-02") & (df["date"] <= "2023-01-07")]
    pd.testing.assert_frame_equal(
        result.to_pandas().reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_filter_by_time_polars_accessor_grouped():
    df = _sample_df()
    pl_df = pl.from_pandas(df)

    result = pl_df.group_by("id").tk.filter_by_time(
        date_column="date",
        start_date="2023-01-02",
        end_date="2023-01-07",
    )

    expected = df[(df["date"] >= "2023-01-02") & (df["date"] <= "2023-01-07")]
    pd.testing.assert_frame_equal(
        result.to_pandas().reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_filter_by_time_defaults():
    df = _sample_df()
    result = tk.filter_by_time(data=df, date_column="date", engine="pandas")
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), df.reset_index(drop=True)
    )

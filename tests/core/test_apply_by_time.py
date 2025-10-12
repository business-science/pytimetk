import pandas as pd
import polars as pl
import pytimetk as tk
# noqa: F401


def _sample_df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-01",
                    "2023-01-03",
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )


def test_apply_by_time_pandas_dataframe():
    df = _sample_df()
    result = tk.apply_by_time(
        data=df,
        date_column="date",
        freq="D",
        total=lambda frame: frame["value"].sum(),
    )
    expected = (
        df.set_index("date")
        .resample("D")
        .apply(lambda frame: pd.Series({"total": frame["value"].sum()}))
        .fillna(0)
        .reset_index()
    )
    pd.testing.assert_frame_equal(result, expected)


def test_apply_by_time_pandas_groupby():
    df = _sample_df()
    result = tk.apply_by_time(
        data=df.groupby("group"),
        date_column="date",
        freq="D",
        total=lambda frame: frame["value"].sum(),
    )
    assert {"group", "total"}.issubset(result.columns)


def test_apply_by_time_polars_dataframe():
    df = _sample_df()
    pl_df = pl.from_pandas(df)
    result = tk.apply_by_time(
        data=pl_df,
        date_column="date",
        freq="D",
        total=lambda frame: frame["value"].sum(),
        engine="polars",
    )
    assert isinstance(result, pl.DataFrame)
    assert "total" in result.columns


def test_apply_by_time_polars_accessor():
    df = _sample_df()
    pl_df = pl.from_pandas(df)
    result = pl_df.group_by("group").tk.apply_by_time(
        date_column="date",
        freq="D",
        total=lambda frame: frame["value"].sum(),
    )
    assert isinstance(result, pl.DataFrame)
    assert {"group", "total"}.issubset(result.columns)

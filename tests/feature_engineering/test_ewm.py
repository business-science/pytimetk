import pandas as pd
import polars as pl
import pytimetk as tk


def _sample_data():
    df = tk.load_dataset("m4_daily", parse_dates=["date"]).query("id == 'D10'").head(10)
    return df.copy()


def test_augment_ewm_reduce_memory_pandas():
    df = _sample_data()

    result = df.augment_ewm(
        date_column="date",
        value_column="value",
        window_func="mean",
        alpha=0.3,
        reduce_memory=True,
    )

    expected_cols = {"value", "value_ewm_mean_alpha_0.3"}
    assert expected_cols.issubset(result.columns)


def test_augment_ewm_polars_accessor():
    df = _sample_data()
    pl_df = pl.from_pandas(df)

    result = pl_df.tk.augment_ewm(
        date_column="date",
        value_column="value",
        window_func="mean",
        alpha=0.3,
        reduce_memory=True,
    )

    assert isinstance(result, pl.DataFrame)
    assert "value_ewm_mean_alpha_0.3" in result.columns


def test_augment_ewm_polars_groupby_accessor():
    df = tk.load_dataset("m4_daily", parse_dates=["date"]).head(200)

    result = (
        pl.from_pandas(df)
        .group_by("id")
        .tk.augment_ewm(
            date_column="date",
            value_column="value",
            window_func="mean",
            alpha=0.2,
        )
    )

    assert "value_ewm_mean_alpha_0.2" in result.columns

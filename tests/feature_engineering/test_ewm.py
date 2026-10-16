import pandas as pd
import polars as pl
import pytimetk as tk
import pytest


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


def test_augment_ewm_multiple_alpha_pandas():
    df = _sample_data()

    result = df.augment_ewm(
        date_column="date",
        value_column="value",
        window_func=["mean", "std"],
        alpha=[0.1, 0.3],
    )

    expected_cols = {
        "value_ewm_mean_alpha_0.1",
        "value_ewm_std_alpha_0.1",
        "value_ewm_mean_alpha_0.3",
        "value_ewm_std_alpha_0.3",
    }
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


@pytest.mark.parametrize("window_func", ["mean", "std", "var"])
@pytest.mark.parametrize("grouped", [False, True])
def test_augment_ewm_polars_min_periods(window_func, grouped):
    frame = pd.DataFrame({
        "id": ["A"] * 6 + ["B"] * 6,
        "date": pd.date_range("2024-01-01", periods=12),
        "value": [2.0, 4.0, 8.0, 3.0, 9.0, 5.0] * 2,
    })
    polars_frame = pl.from_pandas(frame)
    pandas_input = frame.groupby("id") if grouped else frame
    polars_input = polars_frame.group_by("id") if grouped else polars_frame
    kwargs = {
        "date_column": "date",
        "value_column": "value",
        "window_func": window_func,
        "alpha": 0.5,
        "min_periods": 3,
        "adjust": False,
    }

    expected = tk.augment_ewm(pandas_input, engine="pandas", **kwargs)
    result = tk.augment_ewm(polars_input, engine="polars", **kwargs)

    pd.testing.assert_frame_equal(result.to_pandas(), expected)

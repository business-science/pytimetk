import pandas as pd
import pytest
import pytimetk as tk
import polars as pl


def test_engine_equivalence():
    df = tk.load_dataset("m4_daily", parse_dates=["date"])
    by = ["date", "id"]

    # test pandas + DataFrame
    output1 = (
        df.augment_fourier(
            date_column="date",
            periods=[1, 2],
            max_order=1,
            reduce_memory=False,
            engine="pandas",
        )
        .sort_values(by=by)
        .reset_index(drop=True)
    )

    # test pandas + groupby
    output2 = (
        df.groupby("id")
        .augment_fourier(
            date_column="date",
            periods=[1, 2],
            max_order=1,
            reduce_memory=False,
            engine="pandas",
        )
        .sort_values(by=by)
        .reset_index(drop=True)
    )

    # test polars + DataFrame
    output3 = (
        df.augment_fourier(
            date_column="date",
            periods=[1, 2],
            max_order=1,
            reduce_memory=False,
            engine="polars",
        )
        .sort_values(by=by)
        .reset_index(drop=True)
    )

    # test polars + groupby
    output4 = (
        df.groupby("id")
        .augment_fourier(
            date_column="date",
            periods=[1, 2],
            max_order=1,
            reduce_memory=False,
            engine="polars",
        )
        .sort_values(by=by)
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(output1, output2)
    pd.testing.assert_frame_equal(output1, output3)
    pd.testing.assert_frame_equal(output1, output4)


def test_polars_accessor_groupby():
    df = tk.load_dataset("m4_daily", parse_dates=["date"]).head(200)

    pl_result = (
        pl.from_pandas(df)
        .group_by("id")
        .tk.augment_fourier(
            date_column="date",
            periods=[1],
            max_order=1,
        )
    )

    assert "date_sin_1_1" in pl_result.columns
    assert "date_cos_1_1" in pl_result.columns


def test_docstring_single_group_example():
    df = tk.load_dataset("m4_daily", parse_dates=["date"])

    result = df.query("id == 'D10'").augment_fourier(
        date_column="date", periods=[1, 7], max_order=1
    )

    expected_cols = {
        "id",
        "date",
        "value",
        "date_sin_1_1",
        "date_cos_1_1",
        "date_sin_1_7",
        "date_cos_1_7",
    }
    assert expected_cols.issubset(set(result.columns))

    first_row = result.sort_values("date").iloc[0]
    assert first_row["date_sin_1_1"] == pytest.approx(0.0)
    assert first_row["date_cos_1_1"] == pytest.approx(1.0)
    assert first_row["date_sin_1_7"] == pytest.approx(0.0)
    assert first_row["date_cos_1_7"] == pytest.approx(1.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_docstring_groupby_examples(engine):
    df = tk.load_dataset("m4_daily", parse_dates=["date"])

    direct_result = (
        df.augment_fourier(
            date_column="date",
            periods=[1, 7],
            max_order=1,
            engine=engine,
        )
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )

    group_result = (
        df.groupby("id")
        .augment_fourier(date_column="date", periods=[1, 7], max_order=1, engine=engine)
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )

    expected_cols = [
        "date_sin_1_1",
        "date_cos_1_1",
        "date_sin_1_7",
        "date_cos_1_7",
    ]

    for col in expected_cols:
        assert col in direct_result.columns
        assert col in group_result.columns

    pd.testing.assert_frame_equal(
        direct_result[expected_cols],
        group_result[expected_cols],
    )

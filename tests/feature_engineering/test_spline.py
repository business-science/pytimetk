import pandas as pd
import polars as pl
import pytest
import pytimetk as tk


def _prepare_dataframe():
    df = tk.load_dataset("m4_daily", parse_dates=["date"])
    return df


def test_dataframe_function_equivalence():
    df = _prepare_dataframe()

    result_df = df.augment_spline(
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    result_fn = tk.augment_spline(
        data=df,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    assert list(result_df.columns[-4:]) == [
        "value_bs_1",
        "value_bs_2",
        "value_bs_3",
        "value_bs_4",
    ]
    assert result_df.equals(result_fn)


def test_groupby_function_equivalence():
    df = _prepare_dataframe()
    grouped = df.groupby("id")

    result_group = grouped.augment_spline(
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    result_fn = tk.augment_spline(
        data=grouped,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    assert result_group.equals(result_fn)


def test_polars_engine_dataframe_equivalence():
    df = _prepare_dataframe()

    result_pandas = tk.augment_spline(
        data=df,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    result_polars_mode = tk.augment_spline(
        data=df,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
        engine="polars",
    )

    pd.testing.assert_frame_equal(result_pandas, result_polars_mode)


def test_polars_engine_groupby_equivalence():
    df = _prepare_dataframe()
    grouped = df.groupby("id")

    result_pandas = tk.augment_spline(
        data=grouped,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    result_polars_mode = tk.augment_spline(
        data=grouped,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
        engine="polars",
    )

    pd.testing.assert_frame_equal(result_pandas, result_polars_mode)


def test_polars_dataframe_roundtrip_matches_pandas():
    df = _prepare_dataframe()
    pl_df = pl.from_pandas(df)

    pandas_result = tk.augment_spline(
        data=df,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    polars_result = tk.augment_spline(
        data=pl_df,
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    assert isinstance(polars_result, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_result, polars_result.to_pandas())


def test_polars_groupby_roundtrip_matches_pandas():
    df = _prepare_dataframe()
    pl_df = pl.from_pandas(df)

    pandas_group = tk.augment_spline(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    polars_group = tk.augment_spline(
        data=pl_df.group_by("id"),
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=4,
        degree=3,
        include_intercept=False,
        prefix="value_bs",
    )

    assert isinstance(polars_group, pl.DataFrame)
    pd.testing.assert_frame_equal(pandas_group, polars_group.to_pandas())


def test_spline_type_aliases_and_prefix():
    df = _prepare_dataframe()

    natural = df.augment_spline(
        date_column="date", value_column="value", spline_type="natural", df=5, prefix="ns"
    )
    cyclic = df.augment_spline(
        date_column="date",
        value_column="value",
        spline_type="cyclic",
        df=5,
        prefix="cs",
    )

    natural_cols = [c for c in natural.columns if c.startswith("ns_")]
    cyclic_cols = [c for c in cyclic.columns if c.startswith("cs_")]

    assert len(natural_cols) == 5
    assert len(cyclic_cols) == 5


def test_invalid_spline_type():
    df = _prepare_dataframe()

    with pytest.raises(ValueError):
        df.augment_spline(date_column="date", value_column="value", spline_type="unknown", df=4)

import pandas as pd
import polars as pl
import pytest

import pytimetk as tk
# noqa: F401

from pytimetk.utils.polars_helpers import pl_quantile


@pytest.fixture(scope="module")
def m4_daily_df():
    df = tk.load_dataset("m4_daily", parse_dates=["date"])
    return df


def test_docstring_augment_diffs_examples(m4_daily_df):
    single = m4_daily_df.query("id == 'D10'").augment_diffs(
        date_column="date",
        value_column="value",
        periods=(1, 7),
    )
    assert {"value_diff_1", "value_diff_7"}.issubset(single.columns)

    polars_result = (
        pl.from_pandas(m4_daily_df)
        .group_by("id")
        .tk.augment_diffs(
            date_column="date",
            value_column="value",
            periods=2,
        )
    )
    assert isinstance(polars_result, pl.DataFrame)
    assert "value_diff_2" in polars_result.columns

    custom = m4_daily_df.query("id == 'D10'").augment_diffs(
        date_column="date",
        value_column="value",
        periods=[2, 4],
    )
    assert {"value_diff_2", "value_diff_4"}.issubset(custom.columns)


def test_docstring_augment_pct_change_examples(m4_daily_df):
    single = m4_daily_df.query("id == 'D10'").augment_pct_change(
        date_column="date",
        value_column="value",
        periods=(1, 7),
    )
    assert {"value_pctdiff_1", "value_pctdiff_7"}.issubset(single.columns)

    polars_result = (
        pl.from_pandas(m4_daily_df)
        .group_by("id")
        .tk.augment_pct_change(
            date_column="date",
            value_column="value",
            periods=2,
        )
    )
    assert isinstance(polars_result, pl.DataFrame)
    assert "value_pctdiff_2" in polars_result.columns

    custom = m4_daily_df.query("id == 'D10'").augment_pct_change(
        date_column="date",
        value_column="value",
        periods=[2, 4],
    )
    assert {"value_pctdiff_2", "value_pctdiff_4"}.issubset(custom.columns)


def test_docstring_augment_lags_examples(m4_daily_df):
    single = (
        m4_daily_df.query("id == 'D10'")
        .augment_lags(
            date_column="date",
            value_column="value",
            lags=(1, 7),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    assert {"value_lag_1", "value_lag_7"}.issubset(single.columns)
    assert single.loc[1, "value_lag_1"] == pytest.approx(single.loc[0, "value"])

    polars_result = (
        pl.from_pandas(m4_daily_df)
        .group_by("id")
        .tk.augment_lags(
            date_column="date",
            value_column="value",
            lags=(1, 3),
        )
    )
    assert isinstance(polars_result, pl.DataFrame)
    assert {"value_lag_1", "value_lag_3"}.issubset(polars_result.columns)

    custom = m4_daily_df.query("id == 'D10'").augment_lags(
        date_column="date",
        value_column="value",
        lags=[2, 4],
    )
    assert {"value_lag_2", "value_lag_4"}.issubset(custom.columns)


def test_docstring_augment_ewm_examples(m4_daily_df):
    pandas_result = m4_daily_df.groupby("id").augment_ewm(
        date_column="date",
        value_column="value",
        window_func=["mean", "std"],
        alpha=0.1,
        engine="pandas",
    )
    expected_cols = {"value_ewm_mean_alpha_0.1", "value_ewm_std_alpha_0.1"}
    assert expected_cols.issubset(pandas_result.columns)

    polars_result = (
        pl.from_pandas(m4_daily_df)
        .group_by("id")
        .tk.augment_ewm(
            date_column="date",
            value_column="value",
            window_func="mean",
            alpha=0.1,
        )
    )
    assert "value_ewm_mean_alpha_0.1" in polars_result.columns


def test_docstring_augment_rolling_examples(m4_daily_df):
    df_small = m4_daily_df.head(400)

    example_one = df_small.groupby("id").augment_rolling(
        date_column="date",
        value_column="value",
        window=[2, 7],
        window_func=[
            "mean",
            ("std", lambda x: x.std()),
        ],
        threads=1,
        engine="pandas",
        show_progress=False,
    )
    assert {
        "value_rolling_mean_win_2",
        "value_rolling_std_win_2",
        "value_rolling_mean_win_7",
        "value_rolling_std_win_7",
    }.issubset(example_one.columns)

    example_two = df_small.groupby("id").augment_rolling(
        date_column="date",
        value_column="value",
        window=(1, 3),
        window_func=[
            "mean",
            ("std", lambda x: x.std()),
        ],
        threads=1,
        engine="pandas",
        show_progress=False,
    )
    assert {
        "value_rolling_mean_win_3",
        "value_rolling_std_win_3",
    }.issubset(example_two.columns)

    polars_result = (
        pl.from_pandas(df_small)
        .group_by("id")
        .tk.augment_rolling(
            date_column="date",
            value_column="value",
            window=(1, 3),
            window_func=["mean", "std"],
        )
    )
    assert {
        "value_rolling_mean_win_3",
        "value_rolling_std_win_3",
    }.issubset(polars_result.columns)


def test_docstring_augment_rolling_apply_examples():
    df_corr = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [2, 16, 20, 40, 41, 50],
        }
    )

    corr_result = df_corr.groupby("id").augment_rolling_apply(
        date_column="date",
        window=3,
        window_func=[("corr", lambda x: x["value1"].corr(x["value2"]))],
        center=False,
        threads=1,
    )

    assert "rolling_corr_win_3" in corr_result.columns
    expected_corr = (
        df_corr[df_corr["id"] == 1]
        .sort_values("date")[["value1", "value2"]]
        .rolling(3)
        .corr()
        .groupby(level=0, sort=False)
        .apply(lambda g: g.xs("value1", level=1)["value2"].iloc[-1])
    )
    computed_corr = (
        corr_result[corr_result["id"] == 1]["rolling_corr_win_3"].dropna().iloc[-1]
    )
    assert computed_corr == pytest.approx(expected_corr.iloc[-1], rel=1e-6)

    pytest.importorskip(
        "sklearn.linear_model",
        reason="Docstring regression example requires scikit-learn",
    )
    from sklearn.linear_model import LinearRegression  # type: ignore[attr-defined]

    df_reg = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [5, 16, 24, 35, 45, 58],
            "value3": [2, 3, 6, 9, 10, 13],
        }
    )

    def regression(window: pd.DataFrame) -> pd.Series:
        model = LinearRegression()
        model.fit(window[["value2", "value3"]], window["value1"])
        return pd.Series(
            [model.intercept_, model.coef_[0]],
            index=["Intercept", "Slope"],
        )

    regression_result = (
        df_reg.groupby("id")
        .augment_rolling_apply(
            date_column="date",
            window=3,
            window_func=[("regression", regression)],
            threads=1,
        )
        .dropna()
    )
    assert "rolling_regression_win_3" in regression_result.columns
    first_row = regression_result["rolling_regression_win_3"].iloc[0]
    assert {"Intercept", "Slope"}.issubset(first_row.index)


def test_docstring_augment_expanding_examples(m4_daily_df):
    df_small = m4_daily_df.head(400)

    pandas_result = df_small.groupby("id").augment_expanding(
        date_column="date",
        value_column="value",
        window_func=[
            "mean",
            "std",
            ("quantile_75", lambda x: pd.Series(x).quantile(0.75)),
        ],
        min_periods=1,
        engine="pandas",
        threads=1,
        show_progress=False,
    )
    assert {
        "value_expanding_mean",
        "value_expanding_std",
        "value_expanding_quantile_75",
    }.issubset(pandas_result.columns)

    polars_result = (
        pl.from_pandas(df_small)
        .group_by("id")
        .tk.augment_expanding(
            date_column="date",
            value_column="value",
            window_func=[
                "mean",
                "std",
                ("quantile_75", pl_quantile(quantile=0.75)),
            ],
            min_periods=1,
        )
    )
    assert {
        "value_expanding_mean",
        "value_expanding_std",
        "value_expanding_quantile_75",
    }.issubset(polars_result.columns)

    range_result = df_small.groupby("id").augment_expanding(
        date_column="date",
        value_column="value",
        window_func=[("range", lambda x: x.max() - x.min())],
        min_periods=1,
        engine="pandas",
        threads=1,
        show_progress=False,
    )
    assert "value_expanding_range" in range_result.columns


def test_docstring_augment_expanding_apply_examples():
    df_corr = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [2, 16, 20, 40, 41, 50],
        }
    )

    corr_result = df_corr.groupby("id").augment_expanding_apply(
        date_column="date",
        window_func=[("corr", lambda x: x["value1"].corr(x["value2"]))],
        threads=1,
    )

    assert "expanding_corr" in corr_result.columns
    expected_last = (
        df_corr[df_corr["id"] == 1][["value1", "value2"]]
        .expanding()
        .corr()
        .groupby(level=0, sort=False)
        .apply(lambda g: g.xs("value1", level=1)["value2"].iloc[-1])
        .iloc[-1]
    )
    computed_last = (
        corr_result[corr_result["id"] == 1]["expanding_corr"].dropna().iloc[-1]
    )
    assert computed_last == pytest.approx(expected_last, rel=1e-6)

    pytest.importorskip(
        "sklearn.linear_model",
        reason="Docstring regression example requires scikit-learn",
    )
    from sklearn.linear_model import LinearRegression  # type: ignore[attr-defined]

    df_reg = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2],
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-05",
                    "2023-01-06",
                ]
            ),
            "value1": [10, 20, 29, 42, 53, 59],
            "value2": [5, 16, 24, 35, 45, 58],
            "value3": [2, 3, 6, 9, 10, 13],
        }
    )

    def regression(window: pd.DataFrame) -> pd.Series:
        model = LinearRegression()
        model.fit(window[["value2", "value3"]], window["value1"])
        return pd.Series(
            [model.intercept_, model.coef_[0]],
            index=["Intercept", "Slope"],
        )

    regression_result = (
        df_reg.groupby("id")
        .augment_expanding_apply(
            date_column="date",
            window_func=[("regression", regression)],
            threads=1,
        )
        .dropna()
    )
    assert "expanding_regression" in regression_result.columns
    first_row = regression_result["expanding_regression"].iloc[0]
    assert {"Intercept", "Slope"}.issubset(first_row.index)


def test_docstring_augment_wavelet_examples():
    walmart = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"]).head(60)
    pandas_result = walmart.groupby("id").augment_wavelet(
        date_column="Date",
        value_column="Weekly_Sales",
        scales=[15],
        sample_rate=1,
        method="bump",
    )
    assert {"bump_scale_15_real", "bump_scale_15_imag"}.issubset(pandas_result.columns)

    taylor = tk.load_dataset("taylor_30_min", parse_dates=["date"]).head(60)
    polars_result = pl.from_pandas(taylor).tk.augment_wavelet(
        date_column="date",
        value_column="value",
        scales=[15],
        sample_rate=1000,
        method="morlet",
    )
    assert {"morlet_scale_15_real", "morlet_scale_15_imag"}.issubset(
        polars_result.columns
    )


def test_docstring_augment_hilbert_examples():
    walmart = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"]).head(80)
    pandas_result = walmart.groupby("id").augment_hilbert(
        date_column="Date",
        value_column=["Weekly_Sales"],
        engine="pandas",
    )
    assert {"Weekly_Sales_hilbert_real", "Weekly_Sales_hilbert_imag"}.issubset(
        pandas_result.columns
    )

    taylor = tk.load_dataset("taylor_30_min", parse_dates=["date"]).head(120)
    polars_result = pl.from_pandas(taylor).tk.augment_hilbert(
        date_column="date",
        value_column=["value"],
    )
    assert {"value_hilbert_real", "value_hilbert_imag"}.issubset(polars_result.columns)


def test_docstring_augment_holiday_signature_examples():
    pytest.importorskip(
        "holidays", reason="Holiday signature example requires the 'holidays' package"
    )

    df = pd.DataFrame(
        pd.date_range(start="2023-01-01", end="2023-01-10"),
        columns=["date"],
    )

    us_result = tk.augment_holiday_signature(df.copy(), "date", "UnitedStates")
    assert {"is_holiday", "before_holiday", "after_holiday", "holiday_name"}.issubset(
        us_result.columns
    )

    france_result = tk.augment_holiday_signature(df.copy(), "date", "France")
    assert france_result["holiday_name"].notna().any()

    polars_result = tk.augment_holiday_signature(
        df.copy(), "date", "France", engine="polars"
    )
    assert polars_result["holiday_name"].notna().any()


def test_docstring_timeseries_signature_examples():
    df = tk.load_dataset("bike_sales_sample", parse_dates=["order_date"])

    pandas_result = df.augment_timeseries_signature(
        date_column="order_date",
        engine="pandas",
    )
    assert "order_date_index_num" in pandas_result.columns

    polars_result = df.augment_timeseries_signature(
        date_column="order_date",
        engine="polars",
    )
    assert "order_date_index_num" in polars_result.columns


def test_docstring_get_timeseries_signature_examples():
    dates = pd.date_range(start="2019-01-01", end="2019-01-10", freq="D")

    pandas_signature = tk.get_timeseries_signature(dates, engine="pandas")
    assert isinstance(pandas_signature, pd.DataFrame)
    assert "idx_index_num" in pandas_signature.columns

    polars_signature = tk.get_timeseries_signature(dates, engine="polars")
    assert isinstance(polars_signature, pl.DataFrame)
    assert "idx_index_num" in polars_signature.columns

    series_signature = pd.Series(dates, name="date").get_timeseries_signature(
        engine="pandas"
    )
    assert isinstance(series_signature, pd.DataFrame)
    assert "date_index_num" in series_signature.columns

    series_polars = pd.Series(dates, name="date").get_timeseries_signature(
        engine="polars"
    )
    assert isinstance(series_polars, pl.DataFrame)
    assert "date_index_num" in series_polars.columns


def test_docstring_augment_spline_examples():
    df = tk.load_dataset("m4_daily", parse_dates=["date"])

    pandas_result = df.query("id == 'D10'").augment_spline(
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=5,
        degree=3,
        prefix="value_bs",
    )
    assert {f"value_bs_{i}" for i in range(1, 6)}.issubset(pandas_result.columns)

    polars_result = pl.from_pandas(df.query("id == 'D10'")).tk.augment_spline(
        date_column="date",
        value_column="value",
        spline_type="bs",
        df=5,
        degree=3,
        prefix="value_bs",
    )
    assert {f"value_bs_{i}" for i in range(1, 6)}.issubset(polars_result.columns)

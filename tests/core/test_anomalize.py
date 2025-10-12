import pytimetk as tk
import pandas as pd
import polars as pl
import pytest
import multiprocess as mp  # Use multiprocess instead of multiprocessing
from itertools import product
# noqa: F401

# Set spawn start method before tests
mp.set_start_method("spawn", force=True)

METHODS = ["stl", "twitter"]

threads = [1, 2]
combinations = list(product(threads, METHODS))


@pytest.mark.parametrize("threads, method", combinations)
def test_01_grouped_anomalize(threads, method):
    df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[
        ["id", "Date", "Weekly_Sales"]
    ]

    anomalize_df = df.groupby("id").anomalize(
        "Date",
        "Weekly_Sales",
        period=52,
        trend=52,
        method=method,
        threads=threads,
        show_progress=False,
    )

    expected_colnames = [
        "id",
        "Date",
        "observed",
        "seasonal",
        "seasadj",
        "trend",
        "remainder",
        "anomaly",
        "anomaly_score",
        "anomaly_direction",
        "recomposed_l1",
        "recomposed_l2",
        "observed_clean",
    ]

    assert anomalize_df.shape[0] == df.shape[0]
    assert expected_colnames == list(anomalize_df.columns)


# Rest of the test file remains unchanged


def test_anomalize_polars_accessor():
    sample = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=12, freq="MS"),
            "value": [10, 12, 13, 14, 50, 18, 19, 20, 21, 22, 23, 24],
        }
    )

    pl_df = pl.from_pandas(sample)

    result = pl_df.tk.anomalize(
        date_column="date",
        value_column="value",
        period=12,
        method="stl",
        show_progress=False,
    )

    assert isinstance(result, pl.DataFrame)

    result_pd = result.to_pandas()

    assert {"observed", "anomaly", "observed_clean"}.issubset(result_pd.columns)
    assert len(result_pd) == len(sample)

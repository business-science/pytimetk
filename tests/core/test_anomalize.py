import pytimetk as tk
import pandas as pd
import pytest
import multiprocess as mp  # Use multiprocess instead of multiprocessing
from itertools import product

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

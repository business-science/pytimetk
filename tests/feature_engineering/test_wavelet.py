import pandas as pd
import polars as pl

import pytimetk as tk


def _sample_wavelet_data():
    df = tk.load_dataset("taylor_30_min", parse_dates=["date"]).head(60)
    return df.copy()


def test_augment_wavelet_pandas():
    df = _sample_wavelet_data()

    result = df.augment_wavelet(
        date_column="date",
        value_column="value",
        method="morlet",
        sample_rate=1000,
        scales=[15],
    )

    assert "morlet_scale_15_real" in result.columns
    assert "morlet_scale_15_imag" in result.columns


def test_augment_wavelet_polars_accessor():
    df = _sample_wavelet_data()

    pl_result = pl.from_pandas(df).tk.augment_wavelet(
        date_column="date",
        value_column="value",
        method="morlet",
        sample_rate=1000,
        scales=[15],
    )

    assert "morlet_scale_15_real" in pl_result.columns
    assert "morlet_scale_15_imag" in pl_result.columns
    assert pl_result.height == len(df)

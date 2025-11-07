import numpy as np
import pandas as pd
import pytimetk as tk
from pytimetk.utils.selection import contains


def _hourly_frame():
    rng = pd.date_range("2020-01-01", periods=48, freq="H")
    df = pd.DataFrame(
        {
            "id": ["A"] * 24 + ["B"] * 24,
            "date": list(rng[:24]) + list(rng[:24]),
            "value": np.random.RandomState(0).normal(size=48),
        }
    )
    return df


def test_seasonal_diagnostics_auto_features():
    df = _hourly_frame()
    result = tk.seasonal_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
    )

    assert {"seasonal_feature", "seasonal_value"}.issubset(result.columns)
    features = set(result["seasonal_feature"].unique())
    assert "hour" in features
    assert features.issubset({"second", "minute", "hour"})


def test_seasonal_diagnostics_explicit_features():
    df = _hourly_frame()
    result = tk.seasonal_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        feature_set=["hour", "wday.lbl"],
    )

    assert {"hour", "wday.lbl"}.issubset(set(result["seasonal_feature"].unique()))


def test_seasonal_diagnostics_grouped_union_of_features():
    df = _hourly_frame()
    # Make group B have longer horizon to trigger week feature
    extended = df.copy()
    extended.loc[extended["id"] == "B", "date"] = pd.date_range(
        "2020-01-01", periods=24, freq="D"
    )

    result = tk.seasonal_diagnostics(
        data=extended.groupby("id"),
        date_column="date",
        value_column="value",
    )

    assert "id" in result.columns
    assert {"A", "B"} == set(result["id"].unique())
    # ensure both groups present and auto union includes year fallback
    all_features = set(result["seasonal_feature"].unique())
    assert {"hour", "wday.lbl", "week"}.issubset(all_features)


def test_seasonal_diagnostics_supports_tidy_selectors():
    df = _hourly_frame()
    result = tk.seasonal_diagnostics(
        data=df,
        date_column=contains("dat"),
        value_column=contains("val"),
        feature_set=["hour"],
    )
    assert "hour" in set(result["seasonal_feature"].unique())

import numpy as np
import pandas as pd
import pytimetk as tk


def _daily_series_single():
    rng = pd.date_range("2021-01-01", periods=120, freq="D")
    values = np.sin(np.linspace(0, 6 * np.pi, len(rng))) + np.random.RandomState(
        42
    ).normal(scale=0.1, size=len(rng))
    return pd.DataFrame({"date": rng, "value": values})


def _daily_series_grouped():
    base = _daily_series_single()
    df_a = base.iloc[:90].copy()
    df_a["id"] = "A"
    df_b = base.iloc[30:120].copy()
    df_b["id"] = "B"
    return pd.concat([df_a, df_b], ignore_index=True)


def test_stl_diagnostics_columns_present():
    df = _daily_series_single()
    result = tk.stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
    )

    expected_cols = {"date", "observed", "season", "trend", "remainder", "seasadj"}
    assert expected_cols == set(result.columns)
    assert not result.isna().all().any()


def test_stl_diagnostics_allows_custom_frequency():
    df = _daily_series_single()
    result = tk.stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        frequency="7D",
        trend=21,
    )

    assert result["season"].abs().mean() > 0


def test_stl_diagnostics_grouped_preserves_group_cols():
    df = _daily_series_grouped()
    result = tk.stl_diagnostics(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        frequency=7,
        trend="30 days",
    )

    assert "id" in result.columns
    assert {"A", "B"} == set(result["id"].unique())

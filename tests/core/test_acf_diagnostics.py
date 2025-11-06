import numpy as np
import pandas as pd
import pytimetk as tk
from pytimetk.utils.selection import contains
import pytest
import plotly.graph_objects as go


def _sample_acf_frame():
    rng = pd.date_range("2020-01-01", periods=40, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, len(rng)))
    df = pd.DataFrame(
        {
            "id": ["A"] * (len(rng) // 2) + ["B"] * (len(rng) // 2),
            "date": list(rng[: len(rng) // 2]) + list(rng[: len(rng) // 2]),
            "value": np.tile(values[: len(rng) // 2], 2),
            "driver": np.tile(np.cos(np.linspace(0, 2 * np.pi, len(rng) // 2)), 2),
        }
    )
    return df


def test_acf_diagnostics_basic():
    df = _sample_acf_frame()
    result = tk.acf_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        lags=8,
        ccf_columns=["driver"],
    )

    expected_columns = {
        "metric",
        "lag",
        "value",
        "white_noise_upper",
        "white_noise_lower",
    }
    assert expected_columns.issubset(result.columns)
    assert {"ACF", "PACF"}.issubset(result["metric"].unique())
    assert result["lag"].max() <= 8
    assert any(result["metric"].str.startswith("CCF_"))


def test_acf_diagnostics_lag_phrase():
    df = _sample_acf_frame()
    result = tk.acf_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        lags="20 days",
    )
    assert result["lag"].max() <= 20


def test_acf_diagnostics_grouped_output_contains_group_columns():
    df = _sample_acf_frame()
    grouped = df.groupby("id")
    result = tk.acf_diagnostics(
        data=grouped,
        date_column="date",
        value_column="value",
        lags=5,
    )
    assert "id" in result.columns
    assert set(result["id"].unique()) == {"A", "B"}


def test_plot_acf_diagnostics_accepts_tidy_selectors():
    df = _sample_acf_frame()
    fig = tk.plot_acf_diagnostics(
        data=df,
        date_column="date",
        value_column=contains("value", case=False),
        ccf_columns=contains("driver"),
        lags=5,
    )
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_plot_acf_diagnostics_polars(engine):
    df = _sample_acf_frame()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        pl_df = pl.from_pandas(df)
        fig = tk.plot_acf_diagnostics(
            data=pl_df,
            date_column="date",
            value_column=contains("value", case=False),
            ccf_columns=contains("driver"),
            lags=5,
        )
        assert isinstance(fig, go.Figure)
    else:
        fig = tk.plot_acf_diagnostics(
            data=df,
            date_column="date",
            value_column="value",
            ccf_columns="driver",
            lags=5,
        )
        assert isinstance(fig, go.Figure)

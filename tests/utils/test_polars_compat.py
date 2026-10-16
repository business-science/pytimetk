import math

import polars as pl
import pytest

import pytimetk._polars_compat as compat


@pytest.mark.parametrize("supports_min_samples", [False, True])
@pytest.mark.parametrize("keyword", ["min_periods", "min_samples"])
def test_window_kwargs_preserve_minimum_samples(
    monkeypatch, supports_min_samples, keyword
):
    monkeypatch.setattr(
        compat, "_polars_supports_min_samples", lambda: supports_min_samples
    )
    kwargs = {"window_size": 5, keyword: 3}
    expected_keyword = "min_samples" if supports_min_samples else "min_periods"

    assert compat.ensure_polars_rolling_kwargs(kwargs) == {
        "window_size": 5,
        expected_keyword: 3,
    }
    assert kwargs == {"window_size": 5, keyword: 3}


@pytest.mark.parametrize(
    "method, kwargs, expected",
    [
        ("rolling_mean", {"window_size": 3}, 14 / 3),
        ("rolling_sum", {"window_size": 3}, 14),
        ("rolling_std", {"window_size": 3}, math.sqrt(28 / 3)),
        (
            "rolling_map",
            {"window_size": 3, "function": lambda window: window.sum()},
            14,
        ),
        ("ewm_mean", {"alpha": 0.5, "adjust": False}, 5.5),
    ],
)
def test_window_minimum_samples_with_runtime_polars(method, kwargs, expected):
    kwargs = compat.ensure_polars_rolling_kwargs({**kwargs, "min_samples": 3})
    result = pl.DataFrame({"value": [2.0, 4.0, 8.0]}).select(
        getattr(pl.col("value"), method)(**kwargs)
    )["value"]

    assert result.to_list()[:2] == [None, None]
    assert result[2] == pytest.approx(expected)

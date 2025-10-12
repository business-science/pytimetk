"""
Polars method-chaining namespace for pytimetk.

This module registers a ``.tk`` accessor on :class:`polars.DataFrame` objects
and their ``group_by`` result, mirroring the pandas-flavor integration that
already exists for pandas objects. The namespace simply forwards calls to the
public ``augment_*`` APIs while forcing ``engine="polars"`` so that users can
chain finance helpers directly on polars data structures.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable, Dict

import polars as pl

from pytimetk.finance import (
    augment_adx,
    augment_atr,
    augment_bbands,
    augment_cmo,
    augment_drawdown,
    augment_ewma_volatility,
    augment_fip_momentum,
    augment_hurst_exponent,
    augment_macd,
    augment_ppo,
    augment_qsmomentum,
    augment_regime_detection,
    augment_roc,
    augment_rolling_risk_metrics,
    augment_rsi,
    augment_stochastic_oscillator,
)
from pytimetk.feature_engineering import (
    augment_diffs,
    augment_expanding,
    augment_expanding_apply,
    augment_ewm,
    augment_fourier,
    augment_hilbert,
    augment_holiday_signature,
    augment_lags,
    augment_leads,
    augment_pct_change,
    augment_rolling,
    augment_rolling_apply,
    augment_spline,
    augment_timeseries_signature,
    augment_wavelet,
)
from pytimetk.core.summarize_by_time import summarize_by_time

FinanceFunc = Callable[..., pl.DataFrame]


_AUGMENT_FUNCTIONS: Dict[str, FinanceFunc] = {
    "augment_adx": augment_adx,
    "augment_atr": augment_atr,
    "augment_bbands": augment_bbands,
    "augment_cmo": augment_cmo,
    "augment_drawdown": augment_drawdown,
    "augment_ewma_volatility": augment_ewma_volatility,
    "augment_fip_momentum": augment_fip_momentum,
    "augment_hurst_exponent": augment_hurst_exponent,
    "augment_macd": augment_macd,
    "augment_ppo": augment_ppo,
    "augment_qsmomentum": augment_qsmomentum,
    "augment_regime_detection": augment_regime_detection,
    "augment_roc": augment_roc,
    "augment_rolling_risk_metrics": augment_rolling_risk_metrics,
    "augment_rsi": augment_rsi,
    "augment_stochastic_oscillator": augment_stochastic_oscillator,
    "augment_diffs": augment_diffs,
    "augment_expanding": augment_expanding,
    "augment_expanding_apply": augment_expanding_apply,
    "augment_ewm": augment_ewm,
    "augment_fourier": augment_fourier,
    "augment_hilbert": augment_hilbert,
    "augment_holiday_signature": augment_holiday_signature,
    "augment_lags": augment_lags,
    "augment_leads": augment_leads,
    "augment_pct_change": augment_pct_change,
    "augment_rolling": augment_rolling,
    "augment_rolling_apply": augment_rolling_apply,
    "augment_spline": augment_spline,
    "augment_timeseries_signature": augment_timeseries_signature,
    "augment_wavelet": augment_wavelet,
    "summarize_by_time": summarize_by_time,
}


def _make_df_method(func: FinanceFunc) -> Callable[..., pl.DataFrame]:
    @wraps(func)
    def method(self, *args, **kwargs):
        kwargs["engine"] = "polars"
        return func(data=self._df, *args, **kwargs)

    return method


def _make_groupby_method(func: FinanceFunc) -> Callable[..., pl.DataFrame]:
    @wraps(func)
    def method(self, *args, **kwargs):
        kwargs["engine"] = "polars"
        return func(data=self._gb, *args, **kwargs)

    return method


class TkDataFrameNamespace:
    """
    Namespace attached to :class:`polars.DataFrame` that exposes pytimetk
    finance augmenters for fluent method chaining.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df


class _TkGroupByNamespace:
    """
    Lightweight wrapper used for ``df.group_by(...).tk`` chains.
    """

    def __init__(self, gb: pl.dataframe.group_by.GroupBy):
        self._gb = gb


def _register_groupby_namespace():
    groupby_cls = pl.dataframe.group_by.GroupBy

    def _tk(self):
        return _TkGroupByNamespace(self)

    # Attach as a cached property so that each call returns a fresh wrapper.
    setattr(groupby_cls, "tk", property(_tk))


def register_polars_namespace() -> None:
    """
    Register ``.tk`` namespace methods on polars DataFrame and GroupBy objects.
    """
    for name, func in _AUGMENT_FUNCTIONS.items():
        setattr(TkDataFrameNamespace, name, _make_df_method(func))
        setattr(_TkGroupByNamespace, name, _make_groupby_method(func))

    pl.api.register_dataframe_namespace("tk")(TkDataFrameNamespace)

    def _df_tk(self):
        return TkDataFrameNamespace(self)

    setattr(pl.DataFrame, "tk", property(_df_tk))
    _register_groupby_namespace()


# Execute registration on import.
register_polars_namespace()

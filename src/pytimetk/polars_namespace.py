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
from typing import Callable, Dict, Any, TYPE_CHECKING

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
from pytimetk.core.ts_summary import ts_summary
from pytimetk.core.filter_by_time import filter_by_time
from pytimetk.core.apply_by_time import apply_by_time
from pytimetk.plot.plot_anomalies import plot_anomalies
from pytimetk.plot.plot_anomalies_decomp import plot_anomalies_decomp
from pytimetk.plot.plot_anomalies_cleaned import plot_anomalies_cleaned
from pytimetk.plot.plot_correlation_funnel import plot_correlation_funnel
from pytimetk.plot.plot_timeseries import plot_timeseries
from pytimetk.utils.dataframe_ops import convert_to_engine
from pytimetk.core.pad import pad_by_time
from pytimetk.core.ts_features import ts_features
from pytimetk.core.correlationfunnel import binarize, correlate
from pytimetk.core.anomalize import anomalize
from pytimetk.core.future import future_frame
from pytimetk.feature_store import FeatureStoreAccessor

FinanceFunc = Callable[..., pl.DataFrame]
PandasReturnFunc = Callable[..., Any]


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
    "ts_summary": ts_summary,
    "filter_by_time": filter_by_time,
    "apply_by_time": apply_by_time,
    "pad_by_time": pad_by_time,
    "ts_features": ts_features,
    "binarize": binarize,
    "correlate": correlate,
    "anomalize": anomalize,
    "future_frame": future_frame,
}

_PANDAS_RESULT_FUNCTIONS: Dict[str, PandasReturnFunc] = {
    "plot_anomalies": plot_anomalies,
    "plot_anomalies_decomp": plot_anomalies_decomp,
    "plot_anomalies_cleaned": plot_anomalies_cleaned,
    "plot_timeseries": plot_timeseries,
}

_PANDAS_DF_ONLY_FUNCTIONS: Dict[str, PandasReturnFunc] = {
    "plot_correlation_funnel": plot_correlation_funnel,
}


def _make_df_method(func: FinanceFunc) -> Callable[..., pl.DataFrame]:
    @wraps(func)
    def method(self, *args, **kwargs):
        kwargs["engine"] = "polars"
        return func(data=self._df, *args, **kwargs)

    return method


def _make_lazy_method(func: FinanceFunc) -> Callable[..., pl.LazyFrame]:
    @wraps(func)
    def method(self, *args, **kwargs):
        kwargs["engine"] = "polars"
        return func(data=self._lf, *args, **kwargs)

    return method


def _make_groupby_method(func: FinanceFunc) -> Callable[..., pl.DataFrame]:
    @wraps(func)
    def method(self, *args, **kwargs):
        kwargs["engine"] = "polars"
        return func(data=self._gb, *args, **kwargs)

    return method


def _make_df_method_pandas(func: PandasReturnFunc) -> Callable[..., Any]:
    @wraps(func)
    def method(self, *args, **kwargs):
        conversion = convert_to_engine(self._df, "pandas")
        pandas_data = conversion.data
        if isinstance(pandas_data, pl.DataFrame):
            pandas_data = pandas_data.to_pandas()
        return func(pandas_data, *args, **kwargs)

    return method


def _make_groupby_method_pandas(func: PandasReturnFunc) -> Callable[..., Any]:
    @wraps(func)
    def method(self, *args, **kwargs):
        conversion = convert_to_engine(self._gb, "pandas")
        pandas_groupby = conversion.data
        return func(pandas_groupby, *args, **kwargs)

    return method


class TkDataFrameNamespace:
    """
    Namespace attached to :class:`polars.DataFrame` that exposes pytimetk
    finance augmenters for fluent method chaining.
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def feature_store(self, store=None, **store_kwargs):
        """
        Return a feature store accessor bound to this DataFrame.
        """
        return FeatureStoreAccessor(frame=self._df, store=store, store_kwargs=store_kwargs)


class TkLazyFrameNamespace:
    """
    Namespace attached to :class:`polars.LazyFrame` mirroring DataFrame helpers.
    """

    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf


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
        setattr(TkLazyFrameNamespace, name, _make_lazy_method(func))
        setattr(_TkGroupByNamespace, name, _make_groupby_method(func))

    for name, func in _PANDAS_RESULT_FUNCTIONS.items():
        setattr(TkDataFrameNamespace, name, _make_df_method_pandas(func))
        setattr(TkLazyFrameNamespace, name, _make_lazy_method(func))
        setattr(_TkGroupByNamespace, name, _make_groupby_method_pandas(func))

    for name, func in _PANDAS_DF_ONLY_FUNCTIONS.items():
        setattr(TkDataFrameNamespace, name, _make_df_method_pandas(func))
        setattr(TkLazyFrameNamespace, name, _make_lazy_method(func))

    pl.api.register_dataframe_namespace("tk")(TkDataFrameNamespace)
    pl.api.register_lazyframe_namespace("tk")(TkLazyFrameNamespace)

    def _df_tk(self):
        return TkDataFrameNamespace(self)

    def _lf_tk(self):
        return TkLazyFrameNamespace(self)

    setattr(pl.DataFrame, "tk", property(_df_tk))
    setattr(pl.LazyFrame, "tk", property(_lf_tk))
    _register_groupby_namespace()


# Execute registration on import.
register_polars_namespace()


if TYPE_CHECKING:
    class _DataFrameWithTk(pl.DataFrame):
        @property
        def tk(self) -> TkDataFrameNamespace:
            ...

    class _LazyFrameWithTk(pl.LazyFrame):
        @property
        def tk(self) -> TkLazyFrameNamespace:
            ...

    pl.DataFrame = _DataFrameWithTk  # type: ignore[assignment]
    pl.LazyFrame = _LazyFrameWithTk  # type: ignore[assignment]

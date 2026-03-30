from functools import wraps

import pandas as pd
import pandas_flavor as pf


def patch_pandas_flavor() -> None:
    """Shim pandas_flavor rename of register_groupby_method in 0.8.0+."""

    # Pull definitions from both the top-level and the register module so we
    # can bridge exports that may have moved or been renamed.
    try:
        import pandas_flavor.register as pf_register  # type: ignore
    except Exception:  # pragma: no cover - defensive import
        pf_register = None

    register_groupby = getattr(pf, "register_groupby_method", None)
    register_df_groupby = getattr(pf, "register_dataframe_groupby_method", None)

    if pf_register:
        register_groupby = register_groupby or getattr(
            pf_register, "register_groupby_method", None
        )
        register_df_groupby = register_df_groupby or getattr(
            pf_register, "register_dataframe_groupby_method", None
        )

    # If neither name exists (e.g., future rename), create a minimal fallback
    # implementation so our decorators continue to work.
    if not register_groupby and not register_df_groupby:
        try:
            from pandas.core.groupby.generic import DataFrameGroupBy
        except Exception:  # pragma: no cover - pandas import failure
            DataFrameGroupBy = None  # type: ignore

        if DataFrameGroupBy is not None:

            def register_groupby_method(method):
                @wraps(method)
                def wrapper(self, *args, **kwargs):
                    return method(self, *args, **kwargs)

                setattr(DataFrameGroupBy, method.__name__, wrapper)
                return method

            register_groupby = register_groupby_method
            register_df_groupby = register_groupby_method

    # Fill in any missing alias using whichever version we could find.
    if not register_groupby and register_df_groupby:
        register_groupby = register_df_groupby

    if not register_df_groupby and register_groupby:
        register_df_groupby = register_groupby

    if register_groupby and not hasattr(pf, "register_groupby_method"):
        pf.register_groupby_method = register_groupby

    if register_df_groupby and not hasattr(pf, "register_dataframe_groupby_method"):
        pf.register_dataframe_groupby_method = register_df_groupby

    _patch_pandas_future_options()
    _patch_pandas_frequency_aliases()
    _patch_polars_to_pandas()
    _patch_groupby_grouper()


def _patch_groupby_grouper() -> None:
    """Restore the removed pandas GroupBy.grouper attribute on pandas 3.x."""

    try:
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
    except Exception:  # pragma: no cover - pandas import failure
        return

    def _install_grouper_property(groupby_cls):
        if hasattr(groupby_cls, "grouper"):
            return
        if getattr(groupby_cls, "_pytimetk_grouper_patched", False):
            return

        def _getter(self):
            try:
                return object.__getattribute__(self, "_grouper")
            except AttributeError as exc:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute 'grouper'"
                ) from exc

        groupby_cls.grouper = property(_getter)  # type: ignore[attr-defined]
        groupby_cls._pytimetk_grouper_patched = True

    for groupby_cls in (DataFrameGroupBy, SeriesGroupBy):
        _install_grouper_property(groupby_cls)


def _patch_pandas_future_options() -> None:
    """Disable pandas 3 string inference so pandas/polars parity stays stable."""

    try:
        if hasattr(pd.options, "future") and hasattr(pd.options.future, "infer_string"):
            pd.options.future.infer_string = False
    except Exception:  # pragma: no cover - option may not exist on older pandas
        return


def _patch_pandas_frequency_aliases() -> None:
    """Restore support for deprecated pandas frequency aliases on pandas 3.x."""

    try:
        from pandas._libs.tslibs import offsets as tslib_offsets
        from pandas.core.indexes import datetimes as datetimes_module
        from pytimetk.utils.datetime_helpers import normalize_frequency_alias
    except Exception:  # pragma: no cover - defensive import
        return

    if getattr(pd, "_pytimetk_freq_alias_patched", False):
        return

    original_to_offset = tslib_offsets.to_offset
    original_date_range = pd.date_range
    original_to_datetime = pd.to_datetime

    def _normalize(freq):
        return normalize_frequency_alias(freq) if isinstance(freq, str) else freq

    def _normalize_datetime_precision(obj):
        if isinstance(obj, pd.DatetimeIndex):
            return obj.as_unit("ns")
        if isinstance(obj, pd.Series) and pd.api.types.is_datetime64_any_dtype(obj.dtype):
            return obj.dt.as_unit("ns")
        if isinstance(obj, pd.Timestamp):
            return obj.as_unit("ns")
        return obj

    def _compat_to_offset(freq, *args, **kwargs):
        return original_to_offset(_normalize(freq), *args, **kwargs)

    def _compat_date_range(*args, **kwargs):
        if "freq" in kwargs:
            kwargs = dict(kwargs)
            kwargs["freq"] = _normalize(kwargs["freq"])
        return _normalize_datetime_precision(original_date_range(*args, **kwargs))

    def _compat_to_datetime(*args, **kwargs):
        return _normalize_datetime_precision(original_to_datetime(*args, **kwargs))

    tslib_offsets.to_offset = _compat_to_offset
    pd.tseries.frequencies.to_offset = _compat_to_offset
    datetimes_module.to_offset = _compat_to_offset
    pd.date_range = _compat_date_range
    pd.to_datetime = _compat_to_datetime
    pd._pytimetk_freq_alias_patched = True


def _patch_polars_to_pandas() -> None:
    """Normalize polars->pandas round-trips for pandas 3 compatibility."""

    try:
        import polars as pl
    except Exception:  # pragma: no cover - optional dependency
        return

    if getattr(pl, "_pytimetk_to_pandas_patched", False):
        return

    original_df_to_pandas = pl.DataFrame.to_pandas
    original_series_to_pandas = pl.Series.to_pandas

    def _normalize_pandas_obj(obj):
        if isinstance(obj, pd.DataFrame):
            for col in obj.columns:
                dtype = obj[col].dtype
                if isinstance(dtype, pd.StringDtype):
                    obj[col] = obj[col].astype(object)
                if obj[col].dtype == object:
                    obj[col] = obj[col].where(obj[col].notna(), None)
                elif pd.api.types.is_datetime64_any_dtype(obj[col].dtype) and str(
                    obj[col].dtype
                ).endswith("[us]"):
                    obj[col] = obj[col].dt.as_unit("ns")
        elif isinstance(obj, pd.Series):
            dtype = obj.dtype
            if isinstance(dtype, pd.StringDtype):
                obj = obj.astype(object)
            if obj.dtype == object:
                obj = obj.where(obj.notna(), None)
            elif pd.api.types.is_datetime64_any_dtype(obj.dtype) and str(obj.dtype).endswith(
                "[us]"
            ):
                obj = obj.dt.as_unit("ns")
        return obj

    def _compat_df_to_pandas(self, *args, **kwargs):
        return _normalize_pandas_obj(original_df_to_pandas(self, *args, **kwargs))

    def _compat_series_to_pandas(self, *args, **kwargs):
        return _normalize_pandas_obj(original_series_to_pandas(self, *args, **kwargs))

    pl.DataFrame.to_pandas = _compat_df_to_pandas
    pl.Series.to_pandas = _compat_series_to_pandas
    pl._pytimetk_to_pandas_patched = True

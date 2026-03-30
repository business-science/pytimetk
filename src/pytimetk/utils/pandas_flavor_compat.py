from functools import wraps

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

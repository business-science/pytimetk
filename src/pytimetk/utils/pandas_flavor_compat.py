from functools import wraps
from typing import Callable

try:
    import pandas_flavor as pf

    _has_pandas_flavor = True
except ImportError:
    pf = None
    _has_pandas_flavor = False


def patch_pandas_flavor() -> None:
    """Shim pandas_flavor rename of register_groupby_method in 0.8.0+."""
    if not _has_pandas_flavor:
        return

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


# Expose decorators
if _has_pandas_flavor:

    def register_dataframe_method(func: Callable) -> Callable:
        return pf.register_dataframe_method(func)

    def register_groupby_method(func: Callable) -> Callable:
        if hasattr(pf, "register_groupby_method"):
            return pf.register_groupby_method(func)
        # Fallback if patch hasn't run or failed?
        return func

    def register_series_method(func: Callable) -> Callable:
        if hasattr(pf, "register_series_method"):
            return pf.register_series_method(func)
        return func

else:

    def register_dataframe_method(func: Callable) -> Callable:
        return func

    def register_groupby_method(func: Callable) -> Callable:
        return func

    def register_series_method(func: Callable) -> Callable:
        return func

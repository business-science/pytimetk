import importlib.util


def _check_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def has_pandas() -> bool:
    return _check_import("pandas")


def has_polars() -> bool:
    return _check_import("polars")


def has_cudf() -> bool:
    return _check_import("cudf")


def has_plotly() -> bool:
    return _check_import("plotly")


def has_plotnine() -> bool:
    return _check_import("plotnine")


def has_matplotlib() -> bool:
    return _check_import("matplotlib")


def require_pandas():
    if not has_pandas():
        raise ImportError(
            "This feature requires 'pandas'. Install it with `pip install pytimetk[pandas]`."
        )


def require_polars():
    if not has_polars():
        raise ImportError(
            "This feature requires 'polars'. Install it with `pip install pytimetk[polars]`."
        )


def require_cudf():
    if not has_cudf():
        raise ImportError(
            "This feature requires 'cudf'. Install it with `pip install pytimetk[cudf]`."
        )


def require_plotly():
    if not has_plotly():
        raise ImportError(
            "This feature requires 'plotly'. Install it with `pip install pytimetk[plotly]`."
        )


def require_plotnine():
    if not has_plotnine():
        raise ImportError(
            "This feature requires 'plotnine'. Install it with `pip install pytimetk[plotnine]`."
        )


def require_matplotlib():
    if not has_matplotlib():
        raise ImportError(
            "This feature requires 'matplotlib'. Install it with `pip install pytimetk[matplotlib]`."
        )

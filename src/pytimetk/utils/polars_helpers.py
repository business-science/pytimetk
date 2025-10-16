import os
import warnings
from typing import Optional

import polars as pl

from pytimetk.utils.gpu_support import is_polars_gpu_available
from pytimetk.utils.string_helpers import parse_freq_str

_GPU_WARNED = False


def pandas_to_polars_frequency(pandas_freq_str, default=(1, "d")):
    quantity, unit = parse_freq_str(pandas_freq_str)

    unit = unit.upper()

    dict_mapping = {
        "S": (1, "s"),
        "min": (1, "m"),
        "T": (1, "m"),
        "H": (1, "h"),
        "D": (1, "d"),
        "W": (1, "w"),
        "M": (1, "mo"),
        "MS": (1, "mo"),
        "Q": (3, "mo"),
        "QS": (3, "mo"),
        "Y": (1, "y"),
        "YS": (1, "y"),
    }

    polars_tup = dict_mapping.get(unit, default)

    polars_freq_str = f"{quantity * polars_tup[0]}{polars_tup[1]}"

    return polars_freq_str


def pandas_to_polars_aggregation_mapping(column_name):
    return {
        "sum": pl.col(column_name).sum().alias(f"{column_name}_sum"),
        "mean": pl.col(column_name).mean().alias(f"{column_name}_mean"),
        "median": pl.col(column_name).median().alias(f"{column_name}_median"),
        "min": pl.col(column_name).min().alias(f"{column_name}_min"),
        "max": pl.col(column_name).max().alias(f"{column_name}_max"),
        "std": pl.col(column_name).std().alias(f"{column_name}_std"),
        "var": pl.col(column_name).var().alias(f"{column_name}_var"),
        "first": pl.col(column_name).first().alias(f"{column_name}_first"),
        "last": pl.col(column_name).last().alias(f"{column_name}_last"),
        "count": pl.col(column_name).count().alias(f"{column_name}_count"),
        "nunique": pl.col(column_name).n_unique().alias(f"{column_name}_nunique"),
    }


def pl_quantile(**kwargs):
    """Generates configuration for the rolling quantile function in Polars."""
    # Designate this function as a 'configurable' type - this helps 'augment_expanding' recognize and process it appropriately
    func_type = "configurable"
    # Specify the Polars rolling function to be called, `rolling_<func_name>`
    func_name = "quantile"
    # Initial parameters for Polars' rolling quantile function
    # Many will be updated by **kwargs or inferred externally based on the dataframe
    default_kwargs = {
        "quantile": None,
        "interpolation": "midpoint",
        "window_size": None,
        "weights": None,
        "min_periods": None,
        "center": False,
        # 'by': None,
        # 'closed': 'left'
    }
    return func_type, func_name, default_kwargs, kwargs


def update_dict(d1, d2):
    """
    Update values in dictionary `d1` based on matching keys from dictionary `d2`.

    This function will only update the values of existing keys in `d1`.
    New keys present in `d2` but not in `d1` will be ignored.
    """
    for key in d1.keys():
        if key in d2:
            d1[key] = d2[key]
    return d1


def collect_lazyframe(
    lazy_frame: pl.LazyFrame,
    *,
    force_gpu: Optional[bool] = None,
) -> pl.DataFrame:
    """Collect a Polars ``LazyFrame`` while optionally attempting GPU execution.

    Parameters
    ----------
    lazy_frame : pl.LazyFrame
        The Polars lazy plan to collect.
    force_gpu : Optional[bool], optional
        When ``True`` the function attempts GPU execution even if the environment
        variable ``PYTIMETK_POLARS_GPU`` disables it. When ``False`` the GPU path
        is skipped. ``None`` (default) respects the environment variable and
        defaults to attempting the GPU when available.
    """

    try_gpu: bool
    if force_gpu is not None:
        try_gpu = force_gpu
    else:
        env_setting = os.getenv("PYTIMETK_POLARS_GPU", "auto").strip().lower()
        if env_setting in {"0", "false", "off", "disable"}:
            try_gpu = False
        elif env_setting in {"1", "true", "on", "enable"}:
            try_gpu = True
        else:  # "auto" or unspecified
            try_gpu = True

    global _GPU_WARNED

    if try_gpu and is_polars_gpu_available():
        try:
            gpu_engine = pl.GPUEngine(raise_on_fail=False)
            collected = lazy_frame.collect(engine=gpu_engine)
            return collected
        except Exception:
            if not _GPU_WARNED:
                warnings.warn(
                    "Polars GPU execution failed for this plan. Falling back to CPU collection.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _GPU_WARNED = True

    return lazy_frame.collect()

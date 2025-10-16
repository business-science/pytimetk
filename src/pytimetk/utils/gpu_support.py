"""
Helpers for detecting optional GPU backends.

The functions in this module are intentionally lightweight so they can be
imported without incurring heavy GPU initialisation costs. They are used by
higher-level APIs to determine whether cudf or the Polars GPU engine is
available before attempting to leverage those runtimes. All checks are
best-effort and fall back to ``False`` when optional dependencies are missing
or misconfigured to guarantee backwards compatibility for CPU-only users.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Optional

import polars as pl

_CUDA_AVAILABLE_CACHE: Optional[bool] = None
_CUDF_AVAILABLE_CACHE: Optional[bool] = None


def is_cuda_available() -> bool:
    """
    Return True when the CUDA runtime appears usable.

    This performs a best-effort import of ``numba.cuda`` which is bundled with
    RAPIDS. Failure to import the module or to query the driver is interpreted
    as CUDA being unavailable.
    """
    global _CUDA_AVAILABLE_CACHE
    if _CUDA_AVAILABLE_CACHE is not None:
        return _CUDA_AVAILABLE_CACHE

    try:
        numba_cuda = importlib.import_module("numba.cuda")
        _CUDA_AVAILABLE_CACHE = bool(numba_cuda.is_available())
    except Exception:  # pragma: no cover - environment dependent
        _CUDA_AVAILABLE_CACHE = False
    return _CUDA_AVAILABLE_CACHE


def is_cudf_available() -> bool:
    """
    Return True if cudf is importable.

    We avoid importing cudf repeatedly by caching the result. Any exception
    raised during import is treated as cudf being unavailable to prevent crashes
    in CPU-only deployments.
    """
    global _CUDF_AVAILABLE_CACHE
    if _CUDF_AVAILABLE_CACHE is not None:
        return _CUDF_AVAILABLE_CACHE

    spec = importlib.util.find_spec("cudf")
    if spec is None:
        _CUDF_AVAILABLE_CACHE = False
        return _CUDF_AVAILABLE_CACHE

    try:
        import cudf  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        _CUDF_AVAILABLE_CACHE = False
    else:
        _CUDF_AVAILABLE_CACHE = True
    return _CUDF_AVAILABLE_CACHE


def cudf_version() -> Optional[str]:
    """
    Return the cudf version if available, otherwise ``None``.
    """
    if not is_cudf_available():
        return None
    try:
        import cudf  # type: ignore

        return getattr(cudf, "__version__", None)
    except Exception:  # pragma: no cover - environment dependent
        return None


def is_polars_gpu_available() -> bool:
    """
    Check whether the Polars GPU engine API is present.

    This does not validate that an NVIDIA device is present; it merely asserts
    that the runtime was installed with GPU support (``polars[gpu]``).
    """
    return hasattr(pl, "GPUEngine")


__all__ = [
    "cudf_version",
    "is_cuda_available",
    "is_cudf_available",
    "is_polars_gpu_available",
]

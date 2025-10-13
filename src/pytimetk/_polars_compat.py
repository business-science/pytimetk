"""Compatibility utilities for bridging Polars API differences."""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict

import polars as pl


@lru_cache(maxsize=None)
def _polars_supports_min_samples() -> bool:
    """Return True if the current Polars version accepts `min_samples`."""
    try:
        signature = inspect.signature(pl.Expr.rolling_mean)
    except (TypeError, ValueError):
        signature = None

    if signature and "min_samples" in signature.parameters:
        return True

    # Fallback: attempt a tiny rolling call at runtime.
    try:
        pl.DataFrame({"_": [1]}).select(
            pl.col("_").rolling_mean(window_size=1, min_samples=1)
        )
        return True
    except TypeError:
        return False
    except Exception:
        # If something unexpected occurs, assume the modern API so we fail loudly elsewhere.
        return True


def ensure_polars_rolling_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate rolling keyword arguments to match the runtime Polars version.

    Polars < 1.3 expects `min_periods` whereas newer versions use `min_samples`.
    This helper rewrites whichever key is missing so callers can always supply
    `min_samples` in their source code.
    """
    normalized = dict(kwargs)
    if _polars_supports_min_samples():
        if "min_periods" in normalized and "min_samples" not in normalized:
            normalized["min_samples"] = normalized.pop("min_periods")
        return normalized

    if "min_samples" in normalized:
        value = normalized.pop("min_samples")
        normalized.setdefault("min_periods", value)
    return normalized

from __future__ import annotations

from typing import Any, Callable, Sequence

class Expr:
    def rolling_mean(
        self,
        window_size: int,
        weights: Sequence[float] | None = ...,
        *,
        min_periods: int | None = ...,
        center: bool = ...,
    ) -> Expr: ...

    def rolling_std(
        self,
        window_size: int,
        weights: Sequence[float] | None = ...,
        *,
        min_periods: int | None = ...,
        center: bool = ...,
        ddof: int = ...,
    ) -> Expr: ...

    def rolling_sum(
        self,
        window_size: int,
        weights: Sequence[float] | None = ...,
        *,
        min_periods: int | None = ...,
        center: bool = ...,
    ) -> Expr: ...

    def rolling_map(
        self,
        function: Callable[[Any], Any],
        window_size: int,
        weights: Sequence[float] | None = ...,
        *,
        min_periods: int | None = ...,
        center: bool = ...,
    ) -> Expr: ...

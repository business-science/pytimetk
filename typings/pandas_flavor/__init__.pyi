from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def register_dataframe_method(func: F) -> F: ...

def register_groupby_method(func: F) -> F: ...

def register_groupby_accessor(name: str) -> Callable[[type], type]: ...

def register_series_method(func: F) -> F: ...

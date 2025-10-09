from __future__ import annotations

from typing import Any, Iterable, Iterator, Sequence

from .expr import Expr

class DataFrame:
    ...

class Series:
    ...

class LazyFrame:
    ...


def from_pandas(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_arrow(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_dict(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_dicts(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_numpy(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_records(*args: Any, **kwargs: Any) -> DataFrame: ...

def from_dataframe(*args: Any, **kwargs: Any) -> DataFrame: ...

def read_csv(*args: Any, **kwargs: Any) -> DataFrame: ...

def read_parquet(*args: Any, **kwargs: Any) -> DataFrame: ...

__all__ = [
    "DataFrame",
    "LazyFrame",
    "Series",
    "Expr",
    "from_pandas",
    "from_arrow",
    "from_dict",
    "from_dicts",
    "from_numpy",
    "from_records",
    "from_dataframe",
    "read_csv",
    "read_parquet",
]

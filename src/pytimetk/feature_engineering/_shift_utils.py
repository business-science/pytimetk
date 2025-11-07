from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import pandas as pd

from pytimetk.utils.datetime_helpers import resolve_lag_sequence
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame
from pytimetk.utils.selection import ColumnSelector, resolve_column_selection
from pytimetk.utils.checks import check_value_column, check_date_column

try:  # Optional dependency
    import polars as pl
except ImportError:  # pragma: no cover - optional import
    pl = None

try:  # Optional cudf dependency
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfGroupBy  # type: ignore
except ImportError:  # pragma: no cover - optional import
    cudf = None  # type: ignore
    CudfGroupBy = None  # type: ignore


def _extract_date_series(
    data,
    date_column: str,
) -> pd.Series:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data)
        return pd.Series(frame[date_column])
    if isinstance(data, pd.DataFrame):
        return pd.Series(data[date_column])
    if pl is not None:
        if isinstance(data, pl.dataframe.group_by.GroupBy):
            base = getattr(data, "df", None)
            if base is None:
                raise TypeError(
                    "Unable to access underlying DataFrame for the supplied polars GroupBy."
                )
            return pd.Series(base.to_pandas()[date_column])
        if isinstance(data, pl.DataFrame):
            return pd.Series(data.to_pandas()[date_column])
    if cudf is not None:
        if isinstance(data, CudfGroupBy):
            frame = resolve_pandas_groupby_frame(data)
            return pd.Series(frame[date_column])
        if isinstance(data, cudf.DataFrame):
            return pd.Series(data.to_pandas()[date_column])
    if hasattr(data, "to_pandas"):
        frame = data.to_pandas()
        return pd.Series(frame[date_column])
    raise TypeError(
        "Date-based lag specifications currently require pandas, polars, or cudf inputs."
    )


def _resolve_selector_frame(
    data,
) -> pd.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        return resolve_pandas_groupby_frame(data).copy()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if pl is not None:
        if isinstance(data, pl.dataframe.group_by.GroupBy):
            base = getattr(data, "df", None)
            if base is None:
                raise TypeError(
                    "Unable to resolve columns from this polars GroupBy for selector resolution."
                )
            return base.to_pandas()
        if isinstance(data, pl.DataFrame):
            return data.to_pandas()
    if cudf is not None:
        if isinstance(data, CudfGroupBy):
            return resolve_pandas_groupby_frame(data).copy()
        if isinstance(data, cudf.DataFrame):
            return data.to_pandas()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    raise TypeError(
        "Column selectors currently require pandas, polars, or cudf inputs for shift helpers."
    )


def _resolve_single_column(frame: pd.DataFrame, selector, label: str) -> str:
    if isinstance(selector, str):
        return selector
    resolved = resolve_column_selection(
        frame, selector, allow_none=False, require_match=True
    )
    if len(resolved) != 1:
        raise ValueError(
            f"`{label}` selector must resolve to exactly one column (resolved={resolved})."
        )
    return resolved[0]


def _resolve_multi_columns(frame: pd.DataFrame, selector, label: str) -> List[str]:
    if isinstance(selector, (str, bytes)):
        return [selector]
    if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes)):
        collected: List[str] = []
        for entry in selector:
            if isinstance(entry, str):
                collected.append(entry)
            else:
                resolved = resolve_column_selection(
                    frame,
                    entry,
                    allow_none=False,
                    require_match=True,
                    unique=False,
                )
                collected.extend(resolved)
        if not collected:
            raise ValueError(f"`{label}` selector list did not match any columns.")
        ordered: List[str] = []
        seen = set()
        for name in collected:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered
    resolved = resolve_column_selection(
        frame,
        selector,
        allow_none=False,
        require_match=True,
        unique=False,
    )
    if not resolved:
        raise ValueError(f"`{label}` selector did not resolve to any columns.")
    ordered = []
    seen = set()
    for name in resolved:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def resolve_shift_columns(
    data,
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]]],
) -> Tuple[str, List[str]]:
    frame = _resolve_selector_frame(data)
    date_resolved = _resolve_single_column(frame, date_column, "date_column")
    value_resolved = _resolve_multi_columns(frame, value_column, "value_column")
    check_date_column(frame, date_resolved)
    check_value_column(frame, value_resolved, require_numeric_dtype=False)
    return date_resolved, value_resolved


def resolve_shift_values(
    spec: Union[int, Tuple[int, int], List[int], Sequence[Union[int, str]], str],
    *,
    label: str,
    data,
    date_column: str,
) -> List[int]:
    date_series = _extract_date_series(data, date_column)
    idx = pd.Series(pd.to_datetime(date_series), copy=False)

    def _coerce_single(value) -> List[int]:
        if isinstance(value, (int,)):
            if value <= 0:
                raise ValueError(f"`{label}` values must be positive integers (got {value}).")
            return [int(value)]
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"`{label}` tuple specifications must be length 2.")
            start, end = value
            if not all(isinstance(v, int) for v in value):
                raise TypeError(
                    f"`{label}` tuple specifications must contain integers; received {value!r}."
                )
            if start <= 0 or end <= 0:
                raise ValueError(f"`{label}` tuple values must be positive integers (got {value}).")
            if end < start:
                raise ValueError(f"`{label}` tuple end must be >= start (got {value}).")
            return list(range(start, end + 1))
        if isinstance(value, list):
            resolved: List[int] = []
            for entry in value:
                resolved.extend(_coerce_single(entry))
            return resolved
        if isinstance(value, str):
            try:
                resolved = resolve_lag_sequence(value, idx)
            except ValueError as exc:
                raise TypeError(
                    f"Invalid {label} specification '{value}'. Provide durations like '3 days'."
                ) from exc
            resolved = [int(val) for val in resolved if int(val) > 0]
            if not resolved:
                raise ValueError(
                    f"`{label}` duration '{value}' is shorter than the sampling interval."
                )
            return [resolved[-1]]
        raise TypeError(
            f"Unsupported `{label}` specification: {value!r}. Use integers, tuples, lists, or duration strings."
        )

    normalized = _coerce_single(spec)
    # Deduplicate while preserving order
    seen = set()
    ordered: List[int] = []
    for lag in normalized:
        if lag not in seen:
            seen.add(lag)
            ordered.append(lag)
    return ordered

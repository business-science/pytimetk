import re
from typing import Callable, Iterable, List, Sequence, Union

import pandas as pd

from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame

ColumnSelector = Union[
    str,
    Sequence[str],
    re.Pattern,
    Callable[[pd.Index], Iterable[str]],
    None,
]


def contains(pattern: str, *, case: bool = True, regex: bool = False) -> Callable[[pd.Index], List[str]]:
    """
    Build a selector that returns columns containing ``pattern``.

    Parameters
    ----------
    pattern : str
        Substring or regular expression to match.
    case : bool, optional
        Perform case-sensitive matching. Defaults to True.
    regex : bool, optional
        Treat ``pattern`` as a regular expression. Defaults to False.
    """

    def _selector(columns: pd.Index) -> List[str]:
        if regex:
            compiled = re.compile(pattern, 0 if case else re.IGNORECASE)
            return [col for col in columns if compiled.search(col)]
        if not case:
            pattern_lower = pattern.lower()
            return [col for col in columns if pattern_lower in col.lower()]
        return [col for col in columns if pattern in col]

    return _selector


def starts_with(prefix: str, *, case: bool = True) -> Callable[[pd.Index], List[str]]:
    """
    Build a selector that returns columns starting with ``prefix``.
    """

    def _selector(columns: pd.Index) -> List[str]:
        if case:
            return [col for col in columns if col.startswith(prefix)]
        prefix_lower = prefix.lower()
        return [col for col in columns if col.lower().startswith(prefix_lower)]

    return _selector


def ends_with(suffix: str, *, case: bool = True) -> Callable[[pd.Index], List[str]]:
    """
    Build a selector that returns columns ending with ``suffix``.
    """

    def _selector(columns: pd.Index) -> List[str]:
        if case:
            return [col for col in columns if col.endswith(suffix)]
        suffix_lower = suffix.lower()
        return [col for col in columns if col.lower().endswith(suffix_lower)]

    return _selector


def matches(pattern: str, *, flags: int = 0) -> Callable[[pd.Index], List[str]]:
    """
    Build a selector that returns columns matching the supplied regular expression.
    """
    compiled = re.compile(pattern, flags=flags)

    def _selector(columns: pd.Index) -> List[str]:
        return [col for col in columns if compiled.search(col)]

    return _selector


def resolve_column_selection(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    selectors: ColumnSelector,
    *,
    allow_none: bool = True,
    require_match: bool = True,
    unique: bool = True,
) -> List[str]:
    """
    Resolve flexible column selectors into a concrete list of column names.

    Parameters
    ----------
    data : DataFrame or GroupBy
        Data source used to validate column existence.
    selectors : various
        - ``str``: treated as a literal column name.
        - ``Sequence[str]``: a collection of literal column names.
        - ``re.Pattern``: matches columns via ``pattern.search``.
        - ``Callable``: receives the column ``Index`` and must return an
          iterable of column names.
        - ``None``: permitted when ``allow_none`` is True (default), yielding ``[]``.
    allow_none : bool, optional
        Allow ``None`` selectors without raising an error. Defaults to True.
    require_match : bool, optional
        Raise ``ValueError`` when a selector does not match any columns. Defaults to True.
    unique : bool, optional
        Return deduplicated column names while preserving order. Defaults to True.

    Returns
    -------
    list[str]
        Ordered list of columns satisfying the selector(s).
    """
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        columns = resolve_pandas_groupby_frame(data).columns
    elif isinstance(data, pd.DataFrame):
        columns = data.columns
    else:
        raise TypeError("`data` must be a pandas DataFrame or DataFrameGroupBy.")

    if selectors is None:
        if allow_none:
            return []
        raise ValueError("Column selector cannot be None when `allow_none=False`.")

    resolved: List[str] = []

    def _resolve_single(selector) -> List[str]:
        if callable(selector):
            result = list(selector(columns))
            if require_match and len(result) == 0:
                raise ValueError("Column selector callable returned no matches.")
            return result

        if isinstance(selector, re.Pattern):
            result = [col for col in columns if selector.search(col)]
            if require_match and len(result) == 0:
                raise ValueError(
                    f"Regular expression selector '{selector.pattern}' matched no columns."
                )
            return result

        if isinstance(selector, str):
            if selector not in columns:
                if require_match:
                    raise ValueError(f"Column '{selector}' not found in dataframe.")
                return []
            return [selector]

        if isinstance(selector, Sequence):
            collected: List[str] = []
            for item in selector:
                collected.extend(_resolve_single(item))
            return collected

        raise TypeError(f"Unsupported column selector: {selector!r}")

    resolved.extend(_resolve_single(selectors))

    if unique:
        seen = set()
        ordered: List[str] = []
        for name in resolved:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    return resolved

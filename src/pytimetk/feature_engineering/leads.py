from __future__ import annotations

import pandas as pd
import warnings

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import pytimetk.utils.pandas_flavor_compat as pf

if TYPE_CHECKING:
    import polars as pl

try:  # Optional dependency for GPU acceleration
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore
    CudfDataFrameGroupBy = None  # type: ignore

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from pytimetk.feature_engineering._shift_utils import (
    resolve_shift_values,
    resolve_shift_columns,
)
from pytimetk.utils.selection import ColumnSelector


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_leads(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]]],
    leads: Union[int, Tuple[int, int], List[int], Sequence[Union[int, str]], str] = 1,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Adds lead columns to a pandas or polars DataFrame (or grouped DataFrame).

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input tabular data to augment.
    date_column : str or ColumnSelector
        Name of the date column used to determine ordering prior to shifting.
        Accepts tidy selectors for convenience.
    value_column : str, ColumnSelector, or list
        One or more column names/tidy selectors whose lead values will be appended.
    leads : int, tuple, list, or str, optional
        Lead specification. Accepts:

        - int: single lead value
        - tuple(start, end): inclusive range of leads
        - list[int] or list[str]: explicit values/durations
        - str: duration (e.g., ``"3 days"``) converted using ``date_column``
    reduce_memory : bool, optional
        If True, attempts to reduce memory usage (pandas only).
    engine : {"auto", "pandas", "polars", "cudf"}, optional
        Execution engine. When "auto" (default) the backend is inferred from the
        input data type.

    Returns
    -------
    DataFrame
        DataFrame with lead columns appended. The return type matches the input backend.
    """

    check_dataframe_or_groupby(data)
    date_column, value_column = resolve_shift_columns(
        data,
        date_column=date_column,
        value_column=value_column,
    )

    resolved_leads = resolve_shift_values(
        leads,
        label="leads",
        data=data,
        date_column=date_column,
    )

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved in ("polars", "cudf"):
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if engine_resolved == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_leads_pandas(
            data=sorted_data,
            date_column=date_column,
            value_column=value_column,
            leads=resolved_leads,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif engine_resolved == "polars":
        result = _augment_leads_polars(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            leads=resolved_leads,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )
    elif engine_resolved == "cudf":
        result = _augment_leads_cudf(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            leads=resolved_leads,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )
    else:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Unhandled engine: {engine_resolved}")

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_leads_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    leads: List[int],
) -> pd.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for col in value_column:
            for lead in leads:
                df[f"{col}_lead_{lead}"] = df[col].shift(-lead)
        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = resolve_pandas_groupby_frame(data).copy()
        for col in value_column:
            for lead in leads:
                base_df[f"{col}_lead_{lead}"] = base_df.groupby(group_names)[col].shift(
                    -lead
                )
        return base_df

    raise TypeError("Unsupported data type passed to _augment_leads_pandas.")


def _augment_leads_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    leads: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    import polars as pl

    if isinstance(value_column, str):
        value_column = [value_column]

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    lead_columns = []
    for col in value_column:
        for lead in leads:
            expr = pl.col(col).shift(-lead)
            if resolved_groups:
                expr = expr.over(resolved_groups)
            lead_columns.append(expr.alias(f"{col}_lead_{lead}"))

    augmented = sorted_frame.with_columns(lead_columns).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _augment_leads_cudf(
    data: Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"],
    date_column: str,
    value_column: Union[str, List[str]],
    leads: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
):
    if cudf is None:
        raise ImportError(
            "cudf is required for GPU execution but is not installed. "
            "Install pytimetk with the 'gpu' extra."
        )

    if isinstance(value_column, str):
        value_column = [value_column]

    if CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
        frame = resolve_pandas_groupby_frame(data).copy(deep=True)
        resolved_groups: Sequence[str] = list(group_columns) if group_columns else []
    else:
        frame = data.copy(deep=True)  # type: ignore[assignment]
        resolved_groups = list(group_columns) if group_columns else []

    temp_row_col = row_id_column
    generated_row_id = False
    if temp_row_col is None or temp_row_col not in frame.columns:
        temp_base = "__pytimetk_row_id__"
        temp_row_col = temp_base
        suffix = 0
        while temp_row_col in frame.columns:
            suffix += 1
            temp_row_col = f"{temp_base}_{suffix}"
        frame[temp_row_col] = cudf.Series(range(len(frame)), dtype="int64")
        generated_row_id = True

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    frame = frame.sort_values(sort_keys)

    grouped = frame.groupby(resolved_groups, sort=False) if resolved_groups else None

    for col in value_column:
        for lead in leads:
            target_col = f"{col}_lead_{lead}"
            if grouped is not None:
                series = grouped[col].shift(-lead)
            else:
                series = frame[col].shift(-lead)
            frame[target_col] = series

    frame = frame.sort_values(temp_row_col)

    if generated_row_id:
        frame = frame.drop(columns=[temp_row_col])

    return frame

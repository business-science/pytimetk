import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import List, Optional, Sequence, Tuple, Union

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_leads(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    leads: Union[int, Tuple[int, int], List[int]] = 1,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Adds lead columns to a pandas or polars DataFrame (or grouped DataFrame).

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input tabular data to augment.
    date_column : str
        Name of the date column used to determine ordering prior to shifting.
    value_column : str or list
        One or more column names whose lead values will be appended.
    leads : int or tuple or list, optional
        Lead specification. Accepts:

        - int: single lead value
        - tuple(start, end): inclusive range of leads
        - list[int]: explicit list of lead values
    reduce_memory : bool, optional
        If True, attempts to reduce memory usage (pandas only).
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. When "auto" (default) the backend is inferred from the
        input data type.

    Returns
    -------
    DataFrame
        DataFrame with lead columns appended. The return type matches the input backend.
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, value_column, require_numeric_dtype=False)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)
    conversion: FrameConversion = convert_to_engine(data, engine_resolved)
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved == "polars":
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
            leads=leads,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_leads_polars(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            leads=leads,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_leads_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    leads: Union[int, Tuple[int, int], List[int]] = 1,
) -> pd.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    leads = _normalize_shift_values(leads, label="leads")

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for col in value_column:
            for lead in leads:
                df[f"{col}_lead_{lead}"] = df[col].shift(-lead)
        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = data.obj.copy()
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
    leads: Union[int, Tuple[int, int], List[int]],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    leads = _normalize_shift_values(leads, label="leads")
    resolved_groups: Sequence[str] = (
        list(group_columns) if group_columns else _resolve_group_columns(data)
    )
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


def _normalize_shift_values(
    values: Union[int, Tuple[int, int], List[int]],
    label: str,
) -> List[int]:
    if isinstance(values, int):
        return [values]
    if isinstance(values, tuple):
        if len(values) != 2:
            raise ValueError(f"Invalid {label} specification: tuple must be length 2.")
        start, end = values
        return list(range(start, end + 1))
    if isinstance(values, list):
        return [int(v) for v in values]
    raise TypeError(
        f"Invalid {label} specification: type: {type(values)}. Please use int, tuple, or list."
    )


def _resolve_group_columns(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
) -> Sequence[str]:
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        columns = []
        for entry in data.by:
            if isinstance(entry, str):
                columns.append(entry)
            elif hasattr(entry, "meta"):
                columns.append(entry.meta.output_name())
            elif isinstance(entry, list) and entry and all(
                isinstance(item, str) for item in entry
            ):
                columns.extend(entry)
            else:
                raise TypeError("Unsupported polars group key type.")
        return columns
    return []

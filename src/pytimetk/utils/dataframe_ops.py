import numpy as np
import pandas as pd
import polars as pl

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union


PandasLike = Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
PolarsLike = Union[pl.DataFrame, pl.dataframe.group_by.GroupBy]

FrameKind = Literal[
    "pandas_df",
    "pandas_groupby",
    "polars_df",
    "polars_groupby",
]

ROW_ID_BASE = "__pytimetk_row_id__"


@dataclass
class FrameConversion:
    data: Union[PandasLike, PolarsLike]
    original_kind: FrameKind
    row_id_column: Optional[str] = None
    pandas_index: Optional[pd.Index] = None
    group_columns: Optional[Sequence[str]] = None


def identify_frame_kind(data: Union[PandasLike, PolarsLike]) -> FrameKind:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        return "pandas_groupby"
    if isinstance(data, pd.DataFrame):
        return "pandas_df"
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        return "polars_groupby"
    if isinstance(data, pl.DataFrame):
        return "polars_df"
    raise TypeError(
        "`data` must be a pandas DataFrame/GroupBy or a polars DataFrame/GroupBy."
    )


def normalize_engine(
    engine: Optional[str],
    data: Union[PandasLike, PolarsLike],
) -> Literal["pandas", "polars"]:
    """
    Normalise the engine parameter. Defaults to the backend implied by the input data.
    """
    if engine is None or engine == "":
        return _engine_for_kind(identify_frame_kind(data))

    engine_normalised = engine.strip().lower()
    if engine_normalised in ("auto",):
        return _engine_for_kind(identify_frame_kind(data))

    if engine_normalised not in ("pandas", "polars"):
        raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'auto'.")

    return engine_normalised  # type: ignore[return-value]


def convert_to_engine(
    data: Union[PandasLike, PolarsLike],
    engine: Literal["pandas", "polars"],
) -> FrameConversion:
    """
    Ensure the data matches the target engine. Returns conversion metadata that
    can be used to restore the result to the original input type.
    """
    original_kind = identify_frame_kind(data)

    if engine == "pandas":
        if original_kind in ("pandas_df", "pandas_groupby"):
            group_cols = _extract_pandas_group_columns(data) if original_kind == "pandas_groupby" else None
            return FrameConversion(
                data=data,
                original_kind=original_kind,
                group_columns=group_cols,
            )
        if original_kind == "polars_df":
            return FrameConversion(data=data.to_pandas(), original_kind=original_kind)
        if original_kind == "polars_groupby":
            pandas_df = data.df.to_pandas()
            group_names = [str(col) for col in data.by]
            pandas_groupby = pandas_df.groupby(group_names, sort=False)
            return FrameConversion(
                data=pandas_groupby,
                original_kind=original_kind,
                group_columns=group_names,
            )

    if engine == "polars":
        if original_kind == "pandas_df":
            pandas_df = data.copy()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            polars_df = pl.from_pandas(pandas_df)
            return FrameConversion(
                data=polars_df,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
            )
        if original_kind == "pandas_groupby":
            pandas_df = data.obj.copy()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            group_names = [str(col) for col in data.grouper.names]
            polars_df = pl.from_pandas(pandas_df)
            group_key = group_names if len(group_names) > 1 else group_names[0]
            polars_groupby = polars_df.group_by(group_key, maintain_order=True)
            return FrameConversion(
                data=polars_groupby,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
                group_columns=group_names,
            )
        if original_kind in ("polars_df", "polars_groupby"):
            group_cols = (
                _extract_polars_group_columns(data)
                if original_kind == "polars_groupby"
                else None
            )
            return FrameConversion(
                data=data,
                original_kind=original_kind,
                group_columns=group_cols,
            )

    raise RuntimeError("Failed to convert data to requested engine.")


def restore_output_type(
    result: Union[pd.DataFrame, pl.DataFrame],
    conversion: FrameConversion,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Convert the result back to the type implied by the original input.
    """
    # Normalise ordering based on temporary row identifiers when present.
    if conversion.row_id_column:
        if isinstance(result, pl.DataFrame) and conversion.row_id_column in result.columns:
            result = result.sort(conversion.row_id_column).drop(conversion.row_id_column)
        elif isinstance(result, pd.DataFrame) and conversion.row_id_column in result.columns:
            result = (
                result.sort_values(conversion.row_id_column)
                .drop(columns=[conversion.row_id_column])
            )

    if conversion.original_kind in ("pandas_df", "pandas_groupby"):
        if isinstance(result, pl.DataFrame):
            result = result.to_pandas()
        if conversion.pandas_index is not None:
            result.index = conversion.pandas_index
        return result

    if conversion.original_kind in ("polars_df", "polars_groupby"):
        if isinstance(result, pd.DataFrame):
            return pl.from_pandas(result)
        return result

    raise RuntimeError("Unknown frame kind encountered during restoration.")


def ensure_row_id_column(
    frame: pl.DataFrame,
    existing_column: Optional[str] = None,
) -> Tuple[pl.DataFrame, str, bool]:
    """
    Ensure a row identifier column exists on the provided polars frame.
    Returns the frame (possibly modified), the column name, and a flag indicating
    whether the column was generated within this function.
    """
    if existing_column:
        return frame, existing_column, False
    column = _make_temp_column(frame.columns)
    return frame.with_row_count(column), column, True


def _make_temp_column(columns: Sequence[str], base: str = ROW_ID_BASE) -> str:
    if base not in columns:
        return base
    suffix = 1
    while f"{base}_{suffix}" in columns:
        suffix += 1
    return f"{base}_{suffix}"


def _extract_polars_group_columns(
    groupby: pl.dataframe.group_by.GroupBy,
) -> Sequence[str]:
    columns = []
    for entry in groupby.by:
        if isinstance(entry, str):
            columns.append(entry)
        elif hasattr(entry, "meta"):
            columns.append(entry.meta.output_name())
        elif isinstance(entry, list) and entry and all(
            isinstance(item, str) for item in entry
        ):
            columns.extend(entry)
        else:
            raise TypeError("Unsupported polars groupby key type.")
    return columns


def _extract_pandas_group_columns(
    groupby: pd.core.groupby.generic.DataFrameGroupBy,
) -> Sequence[str]:
    return [str(col) for col in groupby.grouper.names]


def resolve_polars_group_columns(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    group_columns: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    """
    Resolve the list of polars group columns from either stored metadata or the
    groupby object itself.
    """
    if group_columns:
        return list(group_columns)
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        return list(_extract_polars_group_columns(data))
    return []


def _engine_for_kind(kind: FrameKind) -> Literal["pandas", "polars"]:
    if kind.startswith("pandas"):
        return "pandas"
    if kind.startswith("polars"):
        return "polars"
    raise RuntimeError("Unsupported frame kind.")

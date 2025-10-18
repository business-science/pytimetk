import numpy as np
import pandas as pd
import polars as pl

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple, Union

try:  # Optional dependency; imported lazily when available
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
except ImportError:  # pragma: no cover - GPU support optional
    cudf = None  # type: ignore
    CudfDataFrameGroupBy = None  # type: ignore

from pytimetk.utils.polars_helpers import collect_lazyframe


def resolve_pandas_groupby_frame(
    groupby: pd.core.groupby.generic.DataFrameGroupBy,
) -> pd.DataFrame:
    """
    Retrieve the underlying pandas DataFrame backing a GroupBy object.

    The pandas-accelerated cudf proxy does not guarantee the public ``obj``
    attribute, so we look through a handful of internal hooks and gracefully
    degrade when only cudf frames are available by converting back to pandas.
    """
    search_order = ("_selected_obj", "obj", "_obj_with_exclusions")
    for attr in search_order:
        candidate = None
        if hasattr(groupby, "__dict__") and attr in groupby.__dict__:
            candidate = groupby.__dict__[attr]
        else:
            try:
                candidate = object.__getattribute__(groupby, attr)
            except AttributeError:
                continue
            except RecursionError:
                continue
            except Exception:
                continue

        if candidate is None:
            continue

        if not isinstance(candidate, pd.DataFrame) and hasattr(candidate, "to_pandas"):
            try:
                candidate = candidate.to_pandas()
            except Exception:
                continue

        if isinstance(candidate, pd.DataFrame):
            return candidate

    raise AttributeError(
        "Unable to access the underlying DataFrame for the supplied GroupBy object."
    )


def _patch_groupby_obj_access() -> None:
    """
    Ensure pandas-style GroupBy objects expose an ``obj`` attribute even when
    accelerated backends (e.g., cudf.pandas) replace the implementation.
    """
    groupby_cls = pd.core.groupby.generic.DataFrameGroupBy
    if getattr(groupby_cls, "_pytimetk_obj_patched", False):
        return

    try:
        sample = pd.DataFrame({"__tmp": [0]}).groupby("__tmp").obj
        if isinstance(sample, pd.DataFrame):
            groupby_cls._pytimetk_obj_patched = True
            return
    except AttributeError:
        pass

    def _getter(self):
        if hasattr(self, "__dict__") and "obj" in self.__dict__:
            candidate = self.__dict__["obj"]
            if hasattr(candidate, "to_pandas") and not isinstance(candidate, pd.DataFrame):
                try:
                    candidate = candidate.to_pandas()
                except Exception:
                    candidate = None
            if isinstance(candidate, pd.DataFrame):
                return candidate
        try:
            candidate = object.__getattribute__(self, "_obj_with_exclusions")
        except Exception:
            candidate = None
        if candidate is not None and hasattr(candidate, "to_pandas") and not isinstance(candidate, pd.DataFrame):
            try:
                candidate = candidate.to_pandas()
            except Exception:
                candidate = None
        if isinstance(candidate, pd.DataFrame):
            return candidate
        raise AttributeError("Unable to access groupby obj on this proxy object.")

    def _setter(self, value):
        if hasattr(self, "__dict__"):
            self.__dict__["obj"] = value
        for attr in ("_selected_obj", "_obj_with_exclusions"):
            try:
                object.__setattr__(self, attr, value)
            except Exception:
                continue

    groupby_cls.obj = property(_getter, _setter)  # type: ignore[attr-defined]
    groupby_cls._pytimetk_obj_patched = True


_patch_groupby_obj_access()


PandasLike = Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
PolarsLike = Union[pl.DataFrame, pl.dataframe.group_by.GroupBy]
CudfLike = Union["cudf.DataFrame", "cudf.core.groupby.groupby.DataFrameGroupBy"]
AnyFrame = Union[PandasLike, PolarsLike, CudfLike, "pl.LazyFrame"]

FrameKind = Literal[
    "pandas_df",
    "pandas_groupby",
    "polars_df",
    "polars_groupby",
    "polars_lazy",
    "cudf_df",
    "cudf_groupby",
]

ROW_ID_BASE = "__pytimetk_row_id__"


@dataclass
class FrameConversion:
    data: Union[PandasLike, PolarsLike, Any]
    original_kind: FrameKind
    row_id_column: Optional[str] = None
    pandas_index: Optional[pd.Index] = None
    group_columns: Optional[Sequence[str]] = None


def identify_frame_kind(data: AnyFrame) -> FrameKind:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        return "pandas_groupby"
    if isinstance(data, pd.DataFrame):
        return "pandas_df"
    if isinstance(data, pl.LazyFrame):
        return "polars_lazy"
    if cudf is not None:
        if CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
            return "cudf_groupby"
        if isinstance(data, cudf.DataFrame):
            return "cudf_df"
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        return "polars_groupby"
    if isinstance(data, pl.DataFrame):
        return "polars_df"
    raise TypeError(
        "`data` must be a pandas, cudf, or polars DataFrame/GroupBy."
    )


def normalize_engine(
    engine: Optional[str],
    data: AnyFrame,
) -> Literal["pandas", "polars", "cudf"]:
    """
    Normalise the engine parameter. Defaults to the backend implied by the input data.
    """
    if engine is None or engine == "":
        return _engine_for_kind(identify_frame_kind(data))

    engine_normalised = engine.strip().lower()
    if engine_normalised in ("auto",):
        return _engine_for_kind(identify_frame_kind(data))

    if engine_normalised not in ("pandas", "polars", "cudf"):
        raise ValueError("Invalid engine. Use 'pandas', 'polars', 'cudf', or 'auto'.")

    return engine_normalised  # type: ignore[return-value]


def convert_to_engine(
    data: AnyFrame,
    engine: Literal["pandas", "polars", "cudf"],
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
        if original_kind == "polars_lazy":
            collected = collect_lazyframe(data)
            pandas_df = collected.to_pandas()
            return FrameConversion(
                data=pandas_df,
                original_kind=original_kind,
            )
        if cudf is not None:
            if original_kind == "cudf_df":
                return FrameConversion(
                    data=data.to_pandas(),
                    original_kind=original_kind,
                )
            if CudfDataFrameGroupBy is not None and original_kind == "cudf_groupby":
                cudf_df = data.obj
                pandas_df = cudf_df.to_pandas()
                group_names = list(_extract_cudf_group_columns(data))
                pandas_groupby = pandas_df.groupby(group_names, sort=False)
                return FrameConversion(
                    data=pandas_groupby,
                    original_kind=original_kind,
                    group_columns=group_names,
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
        if original_kind == "polars_lazy":
            collected = collect_lazyframe(data)
            return FrameConversion(
                data=collected,
                original_kind=original_kind,
            )
        if cudf is not None and original_kind == "cudf_df":
            pandas_df = data.to_pandas()
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
            pandas_df = resolve_pandas_groupby_frame(data).copy()
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
        if cudf is not None and original_kind == "cudf_groupby":
            cudf_df = data.obj.copy()
            pandas_df = cudf_df.to_pandas()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            group_names = list(_extract_cudf_group_columns(data))
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

    if engine == "cudf":
        if cudf is None:
            raise ImportError(
                "The 'cudf' package is required to use engine='cudf'. "
                "Install pytimetk with the 'gpu' extra."
            )
        if original_kind in ("cudf_df", "cudf_groupby"):
            group_cols = (
                list(_extract_cudf_group_columns(data))
                if original_kind == "cudf_groupby"
                else None
            )
            return FrameConversion(
                data=data,
                original_kind=original_kind,
                group_columns=group_cols,
            )
        if original_kind == "pandas_df":
            pandas_df = data.copy()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            cudf_df = cudf.from_pandas(pandas_df)
            return FrameConversion(
                data=cudf_df,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
            )
        if original_kind == "pandas_groupby":
            pandas_df = resolve_pandas_groupby_frame(data).copy()
            group_names = [str(col) for col in data.grouper.names]
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            cudf_df = cudf.from_pandas(pandas_df)
            cudf_groupby = cudf_df.groupby(group_names, sort=False)
            return FrameConversion(
                data=cudf_groupby,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
                group_columns=group_names,
            )
        if original_kind == "polars_df":
            pandas_df = data.to_pandas()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            cudf_df = cudf.from_pandas(pandas_df)
            return FrameConversion(
                data=cudf_df,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
            )
        if original_kind == "polars_groupby":
            pandas_df = data.df.to_pandas()
            group_names = [str(col) for col in data.by]
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            cudf_df = cudf.from_pandas(pandas_df)
            cudf_groupby = cudf_df.groupby(group_names, sort=False)
            return FrameConversion(
                data=cudf_groupby,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
                group_columns=group_names,
            )
        if original_kind == "polars_lazy":
            collected = collect_lazyframe(data)
            pandas_df = collected.to_pandas()
            row_id_col = _make_temp_column(pandas_df.columns)
            pandas_index = pandas_df.index.copy()
            pandas_df[row_id_col] = np.arange(len(pandas_df))
            cudf_df = cudf.from_pandas(pandas_df)
            return FrameConversion(
                data=cudf_df,
                original_kind=original_kind,
                row_id_column=row_id_col,
                pandas_index=pandas_index,
            )
        if original_kind == "cudf_df":
            return FrameConversion(data=data, original_kind=original_kind)

    raise RuntimeError("Failed to convert data to requested engine.")


def restore_output_type(
    result: Union[pd.DataFrame, pl.DataFrame, Any],
    conversion: FrameConversion,
) -> Union[pd.DataFrame, pl.DataFrame, Any]:
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
        elif cudf is not None and isinstance(result, cudf.DataFrame) and conversion.row_id_column in result.columns:
            result = result.sort_values(conversion.row_id_column).drop(columns=[conversion.row_id_column])

    if conversion.original_kind in ("pandas_df", "pandas_groupby"):
        if isinstance(result, pl.DataFrame):
            result = result.to_pandas()
        if cudf is not None and isinstance(result, cudf.DataFrame):
            result = result.to_pandas()
        if conversion.pandas_index is not None and len(conversion.pandas_index) == len(result):
            result.index = conversion.pandas_index
        return result

    if conversion.original_kind in ("polars_df", "polars_groupby"):
        if isinstance(result, pd.DataFrame):
            return pl.from_pandas(result)
        if cudf is not None and isinstance(result, cudf.DataFrame):
            return pl.from_pandas(result.to_pandas())
        return result
    if conversion.original_kind == "polars_lazy":
        if isinstance(result, pd.DataFrame):
            polars_df = pl.from_pandas(result)
        elif cudf is not None and isinstance(result, cudf.DataFrame):
            polars_df = pl.from_pandas(result.to_pandas())
        elif isinstance(result, pl.DataFrame):
            polars_df = result
        else:
            raise TypeError("Unable to convert result back to polars LazyFrame.")
        return polars_df.lazy()

    if conversion.original_kind in ("cudf_df", "cudf_groupby"):
        if cudf is None:
            raise ImportError(
                "cudf is required to restore results to their original cudf type, but it "
                "is not installed in the current environment."
            )
        if isinstance(result, pl.DataFrame):
            result = result.to_pandas()
        if not isinstance(result, pd.DataFrame):
            if hasattr(result, "to_pandas"):
                result = result.to_pandas()
            else:
                raise TypeError("Unable to convert result back to cudf DataFrame.")
        cudf_result = cudf.from_pandas(result)
        if conversion.original_kind == "cudf_groupby":
            group_cols = conversion.group_columns or []
            if not group_cols:
                raise ValueError("Unable to restore cudf GroupBy without grouping columns.")
            return cudf_result.groupby(group_cols, sort=False)
        return cudf_result

    raise RuntimeError("Unknown frame kind encountered during restoration.")


def conversion_to_pandas(
    conversion: FrameConversion,
) -> Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
    """
    Convenience helper that converts the stored frame within a conversion object
    to its pandas representation. When the conversion originates from a pandas
    -> polars transformation a temporary row identifier may exist; this helper
    strips that column before returning the pandas frame.
    """
    data = conversion.data

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        return data
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, pl.DataFrame):
        pandas_df = data.to_pandas()
        row_id = conversion.row_id_column
        if row_id and row_id in pandas_df.columns:
            pandas_df = pandas_df.drop(columns=[row_id])
        return pandas_df
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        pandas_df = data.df.to_pandas()
        row_id = conversion.row_id_column
        if row_id and row_id in pandas_df.columns:
            pandas_df = pandas_df.drop(columns=[row_id])
        group_cols = resolve_polars_group_columns(
            data,
            group_columns=conversion.group_columns,
        )
        if not group_cols:
            raise ValueError("Unable to resolve group columns from polars GroupBy object.")
        return pandas_df.groupby(group_cols, sort=False)
    if cudf is not None:
        if isinstance(data, cudf.DataFrame):
            pandas_df = data.to_pandas()
            row_id = conversion.row_id_column
            if row_id and row_id in pandas_df.columns:
                pandas_df = pandas_df.drop(columns=[row_id])
            return pandas_df
        if CudfDataFrameGroupBy is not None and isinstance(data, CudfDataFrameGroupBy):
            cudf_df = data.obj
            pandas_df = cudf_df.to_pandas()
            row_id = conversion.row_id_column
            if row_id and row_id in pandas_df.columns:
                pandas_df = pandas_df.drop(columns=[row_id])
            group_cols = list(_extract_cudf_group_columns(data))
            return pandas_df.groupby(group_cols, sort=False)

    raise TypeError("Unsupported frame data type during pandas conversion.")


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


def _extract_cudf_group_columns(
    groupby: "cudf.core.groupby.groupby.DataFrameGroupBy",
) -> Sequence[str]:
    if cudf is None or CudfDataFrameGroupBy is None:
        raise RuntimeError("cudf is required to extract cudf group columns.")

    grouping = getattr(groupby, "grouping", None)
    candidates: Sequence[Any] = []
    if grouping is not None:
        for attr in ("names", "keys"):
            values = getattr(grouping, attr, None)
            if values:
                candidates = values
                break

    if not candidates:
        potential = getattr(groupby, "keys", None)
        if callable(potential):
            potential = potential()
        if potential:
            candidates = potential  # type: ignore[assignment]

    if not candidates:
        raise ValueError("Unable to resolve group columns from cudf GroupBy object.")

    return [str(col) for col in candidates]


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


def _engine_for_kind(kind: FrameKind) -> Literal["pandas", "polars", "cudf"]:
    if kind.startswith("pandas"):
        return "pandas"
    if kind.startswith("polars"):
        return "polars"
    if kind.startswith("cudf"):
        return "cudf"
    raise RuntimeError("Unsupported frame kind.")

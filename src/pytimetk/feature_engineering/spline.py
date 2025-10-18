import numpy as np
import pandas as pd
import polars as pl
import pandas_flavor as pf
from patsy import bs, cr, cc
from typing import Literal, Optional, Sequence, Union
import warnings

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    resolve_pandas_groupby_frame,
    restore_output_type,
)


SplineTypeInput = Literal[
    "bs",
    "basis",
    "b-spline",
    "bspline",
    "natural",
    "ns",
    "cr",
    "cyclic",
    "cc",
]


VALID_SPLINE_TYPES = {
    "bs": "bs",
    "basis": "bs",
    "b-spline": "bs",
    "bspline": "bs",
    "natural": "cr",
    "ns": "cr",
    "cr": "cr",
    "cyclic": "cc",
    "cc": "cc",
}

SPLINE_NAME_MAP = {
    "bs": "bspline",
    "cr": "natural_spline",
    "cc": "cyclic_spline",
}


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_spline(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: str,
    spline_type: SplineTypeInput = "bs",
    df: Optional[int] = 5,
    degree: int = 3,
    knots: Optional[Sequence[float]] = None,
    include_intercept: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    prefix: Optional[str] = None,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Add spline basis expansions for a numeric column.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input tabular data or grouped data.
    date_column : str
        Name of the datetime column used to order observations prior to building
        the spline basis.
    value_column : str
        Name of the numeric column to transform into spline basis features.
    spline_type : str, optional
        Spline family. Supported values are "bs" (B-spline), "natural"/"cr"
        (natural cubic spline) and "cyclic"/"cc" (cyclic spline). Defaults to
        "bs".
    df : int, optional
        Degrees of freedom passed to the spline constructor. Required unless
        `knots` are supplied. Defaults to 5.
    degree : int, optional
        Degree of the polynomial pieces (B-spline only). Defaults to 3.
    knots : Sequence[float], optional
        Internal knot positions to use when constructing the spline basis.
    include_intercept : bool, optional
        Whether to include the intercept column (B-spline only). Defaults to
        False.
    lower_bound : float, optional
        Lower boundary for the spline. When omitted the minimum value of
        `value_column` is used.
    upper_bound : float, optional
        Upper boundary for the spline. When omitted the maximum value of
        `value_column` is used.
    prefix : str, optional
        Custom prefix for the generated column names. When omitted a name is
        derived from `value_column` and `spline_type`.
    reduce_memory : bool, optional
        If True, attempt to downcast numeric columns to reduce memory usage.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. When set to "auto" (default) the backend is inferred
        from the input data type. Use "pandas" or "polars" to force a specific
        backend regardless of input type.

    Returns
    -------
    DataFrame
        DataFrame with spline basis columns appended. The result matches the
        input data backend (pandas or polars).

    Examples
    --------

    ```{python}
    # Pandas Example
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset('m4_daily', parse_dates=['date'])

    df_spline = (
        df
            .query("id == 'D10'")
            .augment_spline(
                date_column='date',
                value_column='value',
                spline_type='bs',
                df=5,
                degree=3,
                prefix='value_bs'
            )
    )

    df_spline.head()
    ```

    ```{python}
    pl_spline = (
        pl.from_pandas(df.query("id == 'D10'"))
        .tk.augment_spline(
            date_column='date',
            value_column='value',
            spline_type='bs',
            df=5,
            degree=3,
            prefix='value_bs'
        )
    )

    pl_spline.head()
    ```

    """

    spline_key = _normalise_spline_type(spline_type)
    engine_resolved = normalize_engine(engine, data)

    if df is None and knots is None:
        raise ValueError(
            "Either `df` or `knots` must be provided to define the spline."
        )

    if df is not None and df <= 0:
        raise ValueError("`df` must be a positive integer.")

    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column, require_numeric_dtype=True)

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
        result = _augment_spline_pandas(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            spline_key=spline_key,
            df=df,
            degree=degree,
            knots=knots,
            include_intercept=include_intercept,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prefix=prefix,
        )

        if reduce_memory:
            result = reduce_memory_usage(result)
    elif engine_resolved == "polars":
        result = _augment_spline_polars(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            spline_key=spline_key,
            df=df,
            degree=degree,
            knots=knots,
            include_intercept=include_intercept,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prefix=prefix,
            row_id_column=conversion.row_id_column,
            group_columns=conversion.group_columns,
        )
    else:
        raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'auto'.")

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_spline_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    prefix: Optional[str],
) -> pd.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        base_df = resolve_pandas_groupby_frame(data).copy()
        grouped = base_df.groupby(group_names, sort=False)
        augmented = [
            _augment_spline_frame(
                frame=group,
                date_column=date_column,
                value_column=value_column,
                spline_key=spline_key,
                df=df,
                degree=degree,
                knots=knots,
                include_intercept=include_intercept,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                prefix=prefix,
            )
            for _, group in grouped
        ]
        combined = pd.concat(augmented)
        return combined.sort_index()

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected DataFrame or GroupBy."
        )

    return _augment_spline_frame(
        frame=data.copy(),
        date_column=date_column,
        value_column=value_column,
        spline_key=spline_key,
        df=df,
        degree=degree,
        knots=knots,
        include_intercept=include_intercept,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        prefix=prefix,
    )


def _augment_spline_frame(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    prefix: Optional[str],
) -> pd.DataFrame:
    sorted_frame = frame.sort_values(date_column)

    basis = _build_spline_basis_matrix(
        values=sorted_frame[value_column].to_numpy(copy=False),
        value_column=value_column,
        spline_key=spline_key,
        df=df,
        degree=degree,
        knots=knots,
        include_intercept=include_intercept,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    prefix_value = prefix or _default_prefix(value_column, spline_key, degree)
    column_names = [f"{prefix_value}_{i + 1}" for i in range(basis.shape[1])]
    basis_df = pd.DataFrame(basis, index=sorted_frame.index, columns=column_names)

    result = frame.copy()
    basis_aligned = basis_df.reindex(result.index)
    result[basis_aligned.columns] = basis_aligned
    return result


def _augment_spline_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_column: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    prefix: Optional[str],
    row_id_column: Optional[str],
    group_columns: Optional[Sequence[str]],
) -> pl.DataFrame:
    if isinstance(data, pl.dataframe.group_by.GroupBy):
        base_df = data.df
        grouped_cols = list(group_columns or _resolve_polars_group_columns(data))
        frame_with_id, row_col, generated = ensure_row_id_column(base_df, row_id_column)
        group_key = grouped_cols if len(grouped_cols) > 1 else grouped_cols[0]

        augmented = (
            frame_with_id.group_by(group_key, maintain_order=True)
            .map_groups(
                lambda group: _augment_spline_polars_frame(
                    frame=group,
                    date_column=date_column,
                    value_column=value_column,
                    spline_key=spline_key,
                    df=df,
                    degree=degree,
                    knots=knots,
                    include_intercept=include_intercept,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    prefix=prefix,
                )
            )
            .sort(row_col)
        )

        if generated:
            augmented = augmented.drop(row_col)

        return augmented

    if isinstance(data, pl.DataFrame):
        frame_with_id, row_col, generated = ensure_row_id_column(data, row_id_column)
        augmented = _augment_spline_polars_frame(
            frame=frame_with_id,
            date_column=date_column,
            value_column=value_column,
            spline_key=spline_key,
            df=df,
            degree=degree,
            knots=knots,
            include_intercept=include_intercept,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            prefix=prefix,
        )
        augmented = augmented.sort(row_col)
        if generated:
            augmented = augmented.drop(row_col)
        return augmented

    raise TypeError(
        f"Unsupported data type: {type(data)}. Expected polars DataFrame or GroupBy."
    )


def _resolve_polars_group_columns(
    groupby: pl.dataframe.group_by.GroupBy,
) -> Sequence[str]:
    columns = []
    for entry in groupby.by:
        if isinstance(entry, str):
            columns.append(entry)
        elif hasattr(entry, "meta"):
            columns.append(entry.meta.output_name())
        else:
            raise TypeError("Unsupported polars groupby key type.")
    return columns


def _augment_spline_polars_frame(
    frame: pl.DataFrame,
    date_column: str,
    value_column: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    prefix: Optional[str],
) -> pl.DataFrame:
    sorted_frame = frame.sort(date_column)

    basis = _build_spline_basis_matrix(
        values=sorted_frame[value_column].to_numpy(),
        value_column=value_column,
        spline_key=spline_key,
        df=df,
        degree=degree,
        knots=knots,
        include_intercept=include_intercept,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    prefix_value = prefix or _default_prefix(value_column, spline_key, degree)
    new_columns = [
        pl.Series(f"{prefix_value}_{i + 1}", basis[:, i]) for i in range(basis.shape[1])
    ]

    return sorted_frame.with_columns(new_columns)


def _build_spline_basis_matrix(
    values: Union[Sequence[float], np.ndarray],
    value_column: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mask = ~np.isnan(array)

    if not mask.any():
        raise ValueError(
            f"`value_column` ({value_column}) contains only missing values. Cannot construct spline basis."
        )

    x = array[mask]
    lb = lower_bound if lower_bound is not None else float(x.min())
    ub = upper_bound if upper_bound is not None else float(x.max())

    if spline_key == "cc" and lb >= ub:
        raise ValueError(
            "For cyclic splines `lower_bound` must be less than `upper_bound`."
        )

    knot_values = _prepare_knots(knots)
    kwargs = {"df": df, "knots": knot_values, "lower_bound": lb, "upper_bound": ub}

    if spline_key == "bs":
        kwargs.update({"degree": degree, "include_intercept": include_intercept})
        transformed = bs(x, **kwargs)
    elif spline_key == "cr":
        transformed = cr(x, **kwargs)
    elif spline_key == "cc":
        transformed = cc(x, **kwargs)
    else:
        raise ValueError(f"Unsupported spline type: {spline_key}")

    transformed_array = np.asarray(transformed, dtype=float)
    basis = np.full((array.shape[0], transformed_array.shape[1]), np.nan, dtype=float)
    basis[mask] = transformed_array
    return basis


def _normalise_spline_type(spline_type: SplineTypeInput) -> str:
    key = (spline_type or "").strip().lower()
    if key not in VALID_SPLINE_TYPES:
        valid = "', '".join(sorted(set(VALID_SPLINE_TYPES.keys())))
        raise ValueError(
            f"Unsupported spline_type '{spline_type}'. "
            f"Supported values include: '{valid}'."
        )
    return VALID_SPLINE_TYPES[key]


def _default_prefix(value_column: str, spline_key: str, degree: int) -> str:
    label = SPLINE_NAME_MAP.get(spline_key, "spline")
    if spline_key == "bs":
        return f"{value_column}_{label}_degree_{degree}"
    return f"{value_column}_{label}"


def _prepare_knots(knots: Optional[Sequence[float]]) -> Optional[Sequence[float]]:
    if knots is None:
        return None

    if not isinstance(knots, Sequence):
        raise TypeError("`knots` must be a sequence of numeric values.")

    cleaned = [float(k) for k in knots]
    return sorted(cleaned)

import pandas as pd
import pandas_flavor as pf
from patsy import bs, cr, cc
from typing import Literal, Optional, Sequence, Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage


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
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    column_name: str,
    spline_type: SplineTypeInput = "bs",
    df: Optional[int] = 5,
    degree: int = 3,
    knots: Optional[Sequence[float]] = None,
    include_intercept: bool = False,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    prefix: Optional[str] = None,
    reduce_memory: bool = False,
    engine: str = "pandas",
) -> pd.DataFrame:
    """
    Add spline basis expansions for a numeric column.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Input data or grouped data.
    column_name : str
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
        `column_name` is used.
    upper_bound : float, optional
        Upper boundary for the spline. When omitted the maximum value of
        `column_name` is used.
    prefix : str, optional
        Custom prefix for the generated column names. When omitted a name is
        derived from `column_name` and `spline_type`.
    reduce_memory : bool, optional
        If True, attempt to downcast numeric columns to reduce memory usage.
    engine : str, optional
        Execution engine. Use "pandas" (default) for pandas operations or
        "polars" to mimic polars behaviour while ingesting pandas data.

    Returns
    -------
    pd.DataFrame
        DataFrame with spline basis columns appended.

    Examples
    --------

    ```{python}
    # Pandas Example
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df = df.assign(step=lambda d: d.groupby('id').cumcount())

    df_spline = (
        df
            .query("id == 'D10'")
            .augment_spline(
                column_name='step',
                spline_type='bs',
                df=5,
                degree=3,
                prefix='step_bs'
            )
    )

    df_spline.head()
    ```

    ```{python}
    # Polars Example
    import pandas as pd
    import pytimetk as tk
    import polars as pl

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df = df.assign(step=lambda d: d.groupby('id').cumcount())

    df_spline = (
        df
            .query("id == 'D10'")
            .augment_spline(
                column_name='step',
                spline_type='bs',
                df=5,
                degree=3,
                prefix='step_bs',
                engine='polars'
            )
    )

    df_spline.head()
    ```

    """

    spline_key = _normalise_spline_type(spline_type)
    engine = (engine or "").lower()

    if df is None and knots is None:
        raise ValueError(
            "Either `df` or `knots` must be provided to define the spline."
        )

    if df is not None and df <= 0:
        raise ValueError("`df` must be a positive integer.")

    if engine == "pandas":
        check_dataframe_or_groupby(data)
        check_value_column(data, column_name, require_numeric_dtype=True)

        if reduce_memory:
            data = reduce_memory_usage(data)

        df_result = _augment_spline_pandas(
            data=data,
            column_name=column_name,
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
            df_result = reduce_memory_usage(df_result)

        return df_result.sort_index()

    if engine == "polars":
        check_dataframe_or_groupby(data)
        check_value_column(data, column_name, require_numeric_dtype=True)

        if reduce_memory:
            data = reduce_memory_usage(data)

        df_result = _augment_spline_pandas(
            data=data,
            column_name=column_name,
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
            df_result = reduce_memory_usage(df_result)

        return df_result.sort_index()

    raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")


def _augment_spline_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    column_name: str,
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
        base_df = data.obj.copy()
        grouped = base_df.groupby(group_names, sort=False)
        augmented = [
            _augment_spline_frame(
                frame=group,
                column_name=column_name,
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
        result = pd.concat(augmented).sort_index()
        return result

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected DataFrame or GroupBy."
        )

    return _augment_spline_frame(
        frame=data.copy(),
        column_name=column_name,
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
    column_name: str,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
    prefix: Optional[str],
) -> pd.DataFrame:
    col_series = frame[column_name]
    basis = _build_spline_basis(
        series=col_series,
        spline_key=spline_key,
        df=df,
        degree=degree,
        knots=knots,
        include_intercept=include_intercept,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    prefix_value = prefix or _default_prefix(column_name, spline_key, degree)
    column_names = [f"{prefix_value}_{i + 1}" for i in range(basis.shape[1])]
    basis_df = pd.DataFrame(basis, index=frame.index, columns=column_names)

    return pd.concat([frame, basis_df], axis=1)


def _build_spline_basis(
    series: pd.Series,
    spline_key: str,
    df: Optional[int],
    degree: int,
    knots: Optional[Sequence[float]],
    include_intercept: bool,
    lower_bound: Optional[float],
    upper_bound: Optional[float],
) -> pd.DataFrame:
    values = series.to_numpy(dtype=float, copy=True)
    mask = ~pd.isna(values)

    if not mask.any():
        raise ValueError(
            f"`column_name` ({series.name}) contains only missing values. Cannot construct spline basis."
        )

    x = values[mask]
    lb = lower_bound if lower_bound is not None else float(x.min())
    ub = upper_bound if upper_bound is not None else float(x.max())

    # Ensure bounds make sense for cyclic splines
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

    basis = pd.DataFrame(transformed, index=series.index[mask])

    if mask.all():
        return basis.to_numpy()

    # Reinsert NaN rows to preserve original alignment.
    full_basis = pd.DataFrame(
        data=float("nan"),
        index=series.index,
        columns=basis.columns,
    )
    full_basis.loc[mask] = basis.values
    return full_basis.to_numpy()


def _normalise_spline_type(spline_type: SplineTypeInput) -> str:
    key = (spline_type or "").strip().lower()
    if key not in VALID_SPLINE_TYPES:
        valid = "', '".join(sorted(set(VALID_SPLINE_TYPES.keys())))
        raise ValueError(
            f"Unsupported spline_type '{spline_type}'. "
            f"Supported values include: '{valid}'."
        )
    return VALID_SPLINE_TYPES[key]


def _default_prefix(column_name: str, spline_key: str, degree: int) -> str:
    label = SPLINE_NAME_MAP.get(spline_key, "spline")
    if spline_key == "bs":
        return f"{column_name}_{label}_degree_{degree}"
    return f"{column_name}_{label}"


def _prepare_knots(knots: Optional[Sequence[float]]) -> Optional[Sequence[float]]:
    if knots is None:
        return None

    if not isinstance(knots, Sequence):
        raise TypeError("`knots` must be a sequence of numeric values.")

    cleaned = [float(k) for k in knots]
    return sorted(cleaned)

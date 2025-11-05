import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import Optional, Union, List, Sequence, Any, Tuple

import numpy as np

try:  # Optional dependency for GPU acceleration
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    resolve_pandas_groupby_frame,
    restore_output_type,
    conversion_to_pandas,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_ewm(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    value_column: Union[str, list],
    window_func: Union[str, list] = "mean",
    alpha: float = None,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
    **kwargs,
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Add Exponential Weighted Moving (EWM) window functions to a DataFrame or
    GroupBy object.

    The `augment_ewm` function applies Exponential Weighted Moving (EWM) window
    functions to specified value columns of a DataFrame and adds the results as
    new columns.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The input data to augment. Grouped inputs are processed per group before
        the EWM columns are appended.
    date_column : str
        The name of the column containing date information in the input
        DataFrame or GroupBy object.
    value_column : Union[str, list]
        The `value_column` parameter is used to specify the column(s) on which
        the Exponential Weighted Moving (EWM) calculations will be performed. It
        can be either a string or a list of strings, representing the name(s) of
        the column(s) in the input DataFrame or GroupBy
    window_func : Union[str, list], optional
        The `window_func` parameter is used to specify the Exponential Weighted
        Moving (EWM) window function(s) to apply. It can be a string or a list
        of strings. The possible values are:

        - 'mean': Calculate the exponentially weighted mean.
        - 'median': Calculate the exponentially weighted median.
        - 'std': Calculate the exponentially weighted standard deviation.
        - 'var': Calculate the exponentially weighted variance.

    alpha : float or sequence of floats, optional
        The `alpha` parameter represents the smoothing factor for the Exponential
        Weighted Moving (EWM) window function. It controls the rate at which the
        weights decrease exponentially as the data points move further away from
        the current point. Pass a single value to compute one EWM or a sequence
        of values to generate multiple EWM columns in a single call. This option
        is mutually exclusive with specifying decay parameters such as `com`,
        `span`, or `halflife` through ``**kwargs``.
    engine : {"auto", "pandas", "polars", "cudf"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the input
        data while allowing explicit overrides. Polars and cudf inputs currently
        execute through a pandas fallback.
    reduce_memory : bool, optional
        Attempt to reduce memory usage before/after computation when operating
        on pandas data. If a polars input is supplied a warning is emitted and
        no conversion occurs.
    **kwargs:
        Additional arguments that are directly passed to the pandas EWM method.
        For more details, refer to the "Notes" section below.

    Returns
    -------
    DataFrame
        The function `augment_ewm` returns a DataFrame augmented with the
        results of the Exponential Weighted Moving (EWM) calculations.

    Notes
    ------
    Any additional arguments provided through **kwargs are directly passed
    to the pandas EWM method. These arguments can include parameters like
    'com', 'span', 'halflife', 'ignore_na', 'adjust' and more.

    For a comprehensive list and detailed description of these parameters:

    - Refer to the official pandas documentation:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html

    - Or, within an interactive Python environment, use:
        `?pandas.DataFrame.ewm` to display the method's docstring.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    # Pandas example (engine inferred)
    ewm_df = (
        df
            .groupby('id')
            .augment_ewm(
                date_column = 'date',
                value_column = 'value',
                window_func = [
                    'mean',
                    'std',
                ],
                alpha = 0.1,
            )
    )

    # Polars example using the tk accessor
    ewm_pl = (
        pl.from_pandas(df)
        .group_by('id')
        .tk.augment_ewm(
            date_column='date',
            value_column='value',
            window_func='mean',
            alpha=0.1,
        )
    )
    ```
    """
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)

    engine_resolved = normalize_engine(engine, data)

    if reduce_memory and not isinstance(
        data, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)
    ):
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if engine_resolved == "cudf" and cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required for engine='cudf', but it is not installed."
        )

    if engine_resolved == "polars":
        target_engine = "polars"
    else:
        target_engine = engine_resolved

    conversion: FrameConversion = convert_to_engine(data, target_engine)
    prepared_data = conversion.data

    if reduce_memory and target_engine == "cudf":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if target_engine == "pandas":
        result = _augment_ewm_pandas(
            data=prepared_data,
            date_column=date_column,
            value_column=value_column,
            window_func=window_func,
            alpha=alpha,
            reduce_memory=reduce_memory,
            **kwargs,
        )
    elif target_engine == "polars":
        try:
            result = _augment_ewm_polars(
                data=prepared_data,
                date_column=date_column,
                value_column=value_column,
                window_func=window_func,
                alpha=alpha,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
                **kwargs,
            )
        except NotImplementedError as exc:
            warnings.warn(
                f"{exc} Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_ewm_pandas(
                data=pandas_input,
                date_column=date_column,
                value_column=value_column,
                window_func=window_func,
                alpha=alpha,
                reduce_memory=reduce_memory,
                **kwargs,
            )
    elif target_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_ewm. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_ewm_pandas(
                data=pandas_input,
                date_column=date_column,
                value_column=value_column,
                window_func=window_func,
                alpha=alpha,
                reduce_memory=reduce_memory,
                **kwargs,
            )
        else:
            result = _augment_ewm_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                value_columns=(
                    [value_column] if isinstance(value_column, str) else list(value_column)
                ),
                window_funcs=[window_func] if isinstance(window_func, str) else list(window_func),
                alpha=alpha,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
                **kwargs,
            )
    else:
        raise ValueError(f"Unhandled engine for augment_ewm: {target_engine}")

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _ensure_list_like(values: Union[Any, Sequence[Any]]) -> List[Any]:
    if isinstance(values, (list, tuple, set)):
        return list(values)
    if isinstance(values, range):
        return list(values)
    if isinstance(values, np.ndarray):
        return values.tolist()
    if isinstance(values, pd.Series):
        return values.tolist()
    if isinstance(values, pd.Index):
        return values.tolist()
    polars_series = getattr(pl, "Series", None)
    if polars_series is not None and isinstance(values, polars_series):
        return values.to_list()
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return list(values)
    return [values]


def _prepare_decay_configs(
    alpha: Optional[Union[float, Sequence[float]]],
    kwargs: dict,
) -> List[Tuple[str, Any, dict]]:
    decay_param_keys = ("com", "span", "halflife")
    non_decay_kwargs = {k: v for k, v in kwargs.items() if k not in decay_param_keys}
    provided_decay_params = [key for key in decay_param_keys if key in kwargs]

    if "alpha" in kwargs:
        raise ValueError(
            "Do not supply 'alpha' through **kwargs. Use the 'alpha' argument instead."
        )

    if alpha is not None:
        if provided_decay_params:
            raise ValueError(
                "Specify either 'alpha' or one of 'com', 'span', or 'halflife', but not both."
            )
        alpha_values = _ensure_list_like(alpha)
        if not alpha_values:
            raise ValueError("'alpha' must contain at least one value.")
        return [
            ("alpha", alpha_value, {**non_decay_kwargs, "alpha": alpha_value})
            for alpha_value in alpha_values
        ]

    if not provided_decay_params:
        raise ValueError(
            "No valid decay parameter provided. Specify 'alpha' through function arguments, or one of 'com', 'span', or 'halflife' through **kwargs."
        )

    if len(provided_decay_params) > 1:
        raise ValueError(
            "Multiple decay parameters provided. Specify only one of 'com', 'span', or 'halflife'."
        )

    decay_key = provided_decay_params[0]
    decay_values = _ensure_list_like(kwargs[decay_key])
    if not decay_values:
        raise ValueError(f"Parameter '{decay_key}' must contain at least one value.")

    return [
        (decay_key, decay_value, {**non_decay_kwargs, decay_key: decay_value})
        for decay_value in decay_values
    ]


def _augment_ewm_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    window_func: Union[str, List[str]],
    alpha: Optional[Union[float, Sequence[float]]],
    reduce_memory: bool,
    **kwargs,
) -> pd.DataFrame:
    value_columns = (
        [value_column] if isinstance(value_column, str) else list(value_column)
    )
    window_funcs = [window_func] if isinstance(window_func, str) else list(window_func)

    base_frame = (
        data
        if isinstance(data, pd.DataFrame)
        else resolve_pandas_groupby_frame(data)
    )

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        sorted_frame = base_frame.sort_values(by=[*group_names, date_column])
        grouped = sorted_frame.groupby(group_names)
    else:
        group_names = None
        sorted_frame = base_frame.sort_values(by=[date_column])
        grouped = [([], sorted_frame)]

    decay_configs = _prepare_decay_configs(alpha, kwargs)

    result_dfs = []
    for _, group_df in grouped:
        for col in value_columns:
            for decay_label, decay_value, ewm_params in decay_configs:
                ewm_obj = group_df[col].ewm(**ewm_params)
                for func in window_funcs:
                    result_col_name = f"{col}_ewm_{func}_{decay_label}_{decay_value}"
                    result_series = _apply_ewm_function(ewm_obj, func)
                    group_df[result_col_name] = result_series
        result_dfs.append(group_df)

    result = pd.concat(result_dfs, copy=False)
    if group_names is not None:
        result = result.sort_index(level=group_names)
    else:
        result = result.sort_index()

    if reduce_memory:
        result = reduce_memory_usage(result)

    return result
def _augment_ewm_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    *,
    date_column: str,
    value_column: Union[str, List[str]],
    window_func: Union[str, List[str]],
    alpha: Optional[Union[float, Sequence[float]]],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
    **kwargs,
) -> pl.DataFrame:
    value_columns = (
        [value_column] if isinstance(value_column, str) else list(value_column)
    )
    window_funcs = [window_func] if isinstance(window_func, str) else list(window_func)

    supported_funcs = {"mean", "std", "var"}
    unsupported_funcs = [func for func in window_funcs if func not in supported_funcs]
    if unsupported_funcs:
        raise NotImplementedError(
            "Polars EWM backend currently supports functions: {'mean', 'std', 'var'}."
        )

    allowed_params = {"alpha", "com", "span", "halflife", "adjust", "min_periods"}
    decay_configs = _prepare_decay_configs(alpha, kwargs)
    for _, _, params in decay_configs:
        extra = set(params.keys()) - allowed_params
        if extra:
            raise NotImplementedError(
                f"Polars EWM does not support parameters: {sorted(extra)}."
            )

    resolved_groups = list(group_columns) if group_columns else []
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    sort_columns = resolved_groups + [date_column]
    frame_sorted = frame.sort(sort_columns)
    lf = frame_sorted.lazy()

    def _maybe_over(expr: pl.Expr) -> pl.Expr:
        return expr.over(resolved_groups) if resolved_groups else expr

    exprs: List[pl.Expr] = []
    for col in value_columns:
        for decay_label, decay_value, params in decay_configs:
            filtered_params = {k: params[k] for k in params if k in allowed_params}
            for func in window_funcs:
                if func == "mean":
                    expr = pl.col(col).ewm_mean(**filtered_params)
                elif func == "std":
                    expr = pl.col(col).ewm_std(**filtered_params)
                elif func == "var":
                    expr = pl.col(col).ewm_var(**filtered_params)
                else:
                    continue

                alias = f"{col}_ewm_{func}_{decay_label}_{decay_value}"
                exprs.append(_maybe_over(expr).alias(alias))

    if exprs:
        result = lf.with_columns(exprs).collect()
    else:
        result = frame_sorted

    if row_id_column and row_id_column in result.columns:
        result = result.sort(row_id_column)

    return result


def _apply_ewm_function(ewm_obj, func: Union[str, callable]) -> pd.Series:
    if isinstance(func, str):
        method = getattr(ewm_obj, func, None)
        if method is None:
            raise ValueError(f"Invalid function name: {func}")
        return method()
    raise TypeError(f"Invalid function type: {type(func)}")


def _augment_ewm_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    value_columns: List[str],
    window_funcs: List[str],
    alpha: Optional[Union[float, Sequence[float]]],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
    **kwargs,
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf ewm backend.")

    supported_funcs = {"mean", "std", "var"}
    invalid = [func for func in window_funcs if func not in supported_funcs]
    if invalid:
        raise ValueError(
            f"Unsupported cudf EWM function(s): {invalid}. "
            "Supported functions are {'mean', 'std', 'var'}."
        )

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)

    for col in value_columns:
        if not cudf.api.types.is_numeric_dtype(df_sorted[col]):
            df_sorted[col] = df_sorted[col].astype("float64")

    decay_configs = _prepare_decay_configs(alpha, kwargs)

    group_list: Optional[List[str]]
    if group_columns:
        group_list = list(group_columns)
    else:
        group_list = None

    for col in value_columns:
        if group_list is None:
            for decay_label, decay_value, ewm_params in decay_configs:
                ewm_obj = df_sorted[col].ewm(**ewm_params)
                for func in window_funcs:
                    result_name = f"{col}_ewm_{func}_{decay_label}_{decay_value}"
                    df_sorted[result_name] = getattr(ewm_obj, func)()
        else:
            # Allocate storage for the result column and fill group-by-group.
            for decay_label, decay_value, ewm_params in decay_configs:
                for func in window_funcs:
                    result_name = f"{col}_ewm_{func}_{decay_label}_{decay_value}"
                    result_holder = cudf.Series(
                        np.nan, index=df_sorted.index, dtype="float64"
                    )
                    for _, group_df in df_sorted.groupby(group_list, sort=False):
                        group_indices = group_df.index
                        group_series = group_df[col]
                        ewm_obj = group_series.ewm(**ewm_params)
                        result_holder.loc[group_indices] = getattr(ewm_obj, func)()
                    df_sorted[result_name] = result_holder

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted

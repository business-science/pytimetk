import pandas as pd
import polars as pl
import pandas_flavor as pf
import inspect
import warnings

from typing import Callable, List, Optional, Sequence, Tuple, Union

try:  # Optional cudf dependency for GPU acceleration
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore

from pathos.multiprocessing import ProcessingPool
from functools import partial

from pytimetk._polars_compat import ensure_polars_rolling_kwargs
from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads
from pytimetk.utils.polars_helpers import update_dict
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
    restore_output_type,
    conversion_to_pandas,
)


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_expanding(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    window_func: Union[
        str, List[Union[str, Tuple[str, Callable]]], Tuple[str, Callable]
    ] = "mean",
    min_periods: Optional[int] = None,
    engine: Optional[str] = "auto",
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Apply one or more Series-based expanding functions to one or more columns of a DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each group.
    value_column : Union[str, list]
        Column(s) to which the expanding window functions should be applied. Can be
        a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_expanding` function specifies
        the function(s) to be applied to the expanding windows of the value column(s).

        1. It can be either:
            - A string representing the name of a standard function (e.g., 'mean', 'sum').

        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name for
              the function and the function itself.
            - Each custom function should accept a Pandas Series as its input and
              operate on that series. Example: ("range", lambda x: x.max() - x.min())

        (See more Examples below.)

        Note: If your function needs to operate on multiple columns (i.e., it
              requires access to a DataFrame rather than just a Series), consider
              using the `augment_expanding_apply` function in this library.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the window
        size. If set, a value will be produced even if fewer observations are
        present than the window size.
    engine : {"auto", "pandas", "polars", "cudf"}, optional, default "auto"
        Specifies the backend computation library for augmenting expanding window
        functions. When "auto" the backend is inferred from the input data type.
        Use "pandas" or "polars" to force a specific backend.
    threads : int, optional, default 1
        Number of threads to use for parallel processing. If `threads` is set to
        1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
    show_progress : bool, optional, default True
        If `True`, a progress bar will be displayed during parallel processing.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    **kwargs : additional keyword arguments
        Additional arguments passed to the `pandas.Series.expanding` method when
        using the Pandas engine.

    Returns
    -------
    pd.DataFrame
        The `augment_expanding` function returns a DataFrame with new columns for
        each applied function, window size, and value column.

    Notes
    -----

    ## Performance

    ### Polars Engine (3X faster than Pandas)

    In most cases, the `polars` engine will be faster than the `pandas` engine. Speed tests indicate 3X or more.

    ### Parallel Processing (Pandas Engine Only)

    This function uses parallel processing to speed up computation for large
    datasets with many time series groups:

    Parallel processing has overhead and may not be faster on small datasets.

    To use parallel processing, set `threads = -1` to use all available processors.

    Examples
    --------

    ```{python}
    # Example 1 - Pandas Backend for Expanding Window Functions
    # This example demonstrates the use of string-named functions
    # on an expanding window using the Pandas backend for computations.

    import pytimetk as tk
    import pandas as pd
    import numpy as np

    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date',
                value_column = 'value',
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in standard deviation function,
                     ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function

                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
                threads = 1,  # Disable parallel processing
                show_progress = True,  # Display a progress bar
                )
    )
    display(expanded_df)
    ```


    ```{python}
    # Example 2 - Polars Backend for Expanding Window Functions using Built-Ins
    #             (538X Faster than Pandas)
    #  This example demonstrates the use of string-named functions and configurable
    #  functions using the Polars backend for computations. Configurable functions,
    #  like pl_quantile, allow the use of specific parameters associated with their
    #  corresponding polars.Expr.rolling_<function_name> method.
    #  For instance, pl_quantile corresponds to polars.Expr.rolling_quantile.

    import pytimetk as tk
    import pandas as pd
    import polars as pl
    import numpy as np

    from pytimetk.utils.polars_helpers import pl_quantile
    from pytimetk.utils.pandas_helpers import pd_quantile

    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        pl.from_pandas(df)
            .group_by('id')
            .tk.augment_expanding(
                date_column = 'date',
                value_column = 'value',
                window_func = [
                    'mean',  # Built-in mean function
                    'std',   # Built-in std function
                    ('quantile_75', pl_quantile(quantile=0.75)),
                ],
                min_periods = 1,
            )
    )
    display(expanded_df)
    ```

    ```{python}
    # Example 3 - Lambda Functions for Expanding Window Functions are faster in Pandas than Polars
    # This example demonstrates the use of lambda functions of the form lambda x: x
    # Identity lambda functions, while convenient, have signficantly slower performance.
    # When using lambda functions the Pandas backend will likely be faster than Polars.

    import pytimetk as tk
    import pandas as pd
    import numpy as np

    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    expanded_df = (
        df
            .groupby('id')
            .augment_expanding(
                date_column = 'date',
                value_column = 'value',
                window_func = [

                    ('range', lambda x: x.max() - x.min()),  # Identity lambda function: can be slower, especially in Polars
                ],
                min_periods = 1,
                engine = 'pandas',  # Utilize pandas for the underlying computations
            )
    )
    display(expanded_df)
    ```
    """
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "cudf" and cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required for engine='cudf', but it is not installed."
        )

    conversion_engine = engine_resolved
    conversion: FrameConversion = convert_to_engine(data, conversion_engine)
    prepared_data = conversion.data

    if reduce_memory and conversion_engine == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and conversion_engine in ("polars", "cudf"):
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    value_columns: List[str] = (
        [value_column] if isinstance(value_column, str) else list(value_column)
    )

    window_funcs = list(window_func) if isinstance(window_func, list) else [window_func]

    min_periods_resolved = 1 if min_periods is None else min_periods
    threads_resolved = get_threads(threads)

    if engine_resolved == "pandas":
        prepared_data, idx_unsorted = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_expanding_pandas(
            prepared_data,
            date_column,
            value_columns,
            window_funcs,
            min_periods_resolved,
            threads_resolved,
            show_progress,
            **kwargs,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Expanding augmentation must return a pandas DataFrame.")

        result.index = idx_unsorted

        if reduce_memory:
            result = reduce_memory_usage(result)

        result = result.sort_index()

        restored = restore_output_type(result, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored.sort_index()

        return restored

    if engine_resolved == "cudf":
        fallback_reason: Optional[str] = None
        supported_funcs = {
            "mean",
            "sum",
            "min",
            "max",
            "std",
            "var",
            "count",
            "median",
        }

        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data

        if not isinstance(cudf_df, cudf.DataFrame):
            fallback_reason = (
                "Unsupported cudf object type encountered during conversion."
            )
        elif kwargs:
            fallback_reason = (
                "additional expanding keyword arguments are not supported for cudf yet"
            )
        elif any(
            not isinstance(func, str) or func not in supported_funcs
            for func in window_funcs
        ):
            fallback_reason = (
                "custom expanding functions are not supported for cudf yet"
            )

        if fallback_reason is not None:
            warnings.warn(
                f"augment_expanding cudf path: {fallback_reason}. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_expanding_pandas(
                pandas_input,
                date_column,
                value_columns,
                window_funcs,
                min_periods_resolved,
                threads_resolved,
                show_progress,
                **kwargs,
            )
        else:
            result = _augment_expanding_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                value_columns=value_columns,
                window_funcs=[func for func in window_funcs if isinstance(func, str)],
                min_periods=min_periods_resolved,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )

        restored = restore_output_type(result, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored.sort_index()

        return restored

    if engine_resolved == "polars":
        result_polars = _augment_expanding_polars(
            prepared_data,
            date_column,
            value_columns,
            window_funcs,
            min_periods_resolved,
            conversion.group_columns,
            conversion.row_id_column,
        )

        restored = restore_output_type(result_polars, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored.sort_index()

        return restored

    raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")


def _augment_expanding_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    value_columns: List[str],
    window_funcs: List[str],
    min_periods: Optional[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf expanding backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    resolved_min = 1 if min_periods is None else min_periods

    for col in value_columns:
        if not cudf.api.types.is_numeric_dtype(df_sorted[col]):
            df_sorted[col] = df_sorted[col].astype("float64")

        if group_columns:
            expanding_obj = (
                df_sorted.groupby(list(group_columns), sort=False)[col]
                .expanding(min_periods=resolved_min)
            )
            for func in window_funcs:
                if func == "quantile":
                    result_series = expanding_obj.quantile(0.5).reset_index(drop=True)
                    new_column_name = f"{col}_expanding_quantile_50"
                else:
                    result_series = getattr(expanding_obj, func)().reset_index(drop=True)
                    new_column_name = f"{col}_expanding_{func}"
                df_sorted[new_column_name] = result_series
        else:
            expanding_obj = df_sorted[col].expanding(min_periods=resolved_min)
            for func in window_funcs:
                new_column_name = f"{col}_expanding_{func}"
                if func == "quantile":
                    result_series = expanding_obj.quantile(0.5)
                    new_column_name = f"{col}_expanding_quantile_50"
                else:
                    result_series = getattr(expanding_obj, func)()
                df_sorted[new_column_name] = result_series

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_expanding_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    min_periods: Optional[int] = None,
    threads: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Augments the given dataframe with expanding calculations using the Pandas library.
    """

    # Create a fresh copy of the data, leaving the original untouched
    data_copy = (
        data.copy()
        if isinstance(data, pd.DataFrame)
        else resolve_pandas_groupby_frame(data).copy()
    )

    # Group data if it's a GroupBy object; otherwise, prepare it for the expanding calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(
            group_names
        )
    else:
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]

    # Apply Series-based expanding window functions
    if threads == 1:
        func = partial(
            _process_expanding_window,
            value_columns=value_columns,
            window_funcs=window_funcs,
            min_periods=min_periods,
            **kwargs,
        )

        # Use tqdm to display progress for the loop
        result_dfs = [
            func(group)
            for _, group in conditional_tqdm(
                grouped,
                total=len(grouped),
                desc="Calculating Expanding...",
                display=show_progress,
            )
        ]
    else:
        # Prepare to use pathos.multiprocessing
        pool = ProcessingPool(threads)

        # Use partial to "freeze" arguments for _process_single_roll
        func = partial(
            _process_expanding_window,
            value_columns=value_columns,
            window_funcs=window_funcs,
            min_periods=min_periods,
            **kwargs,
        )

        result_dfs = list(
            conditional_tqdm(
                pool.map(func, (group for _, group in grouped)),
                total=len(grouped),
                desc="Calculating Expanding...",
                display=show_progress,
            )
        )

    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index

    return result_df


def _process_expanding_window(
    group_df: pd.DataFrame,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    min_periods: Optional[int],
    **kwargs,
):
    result_dfs = []
    for col in value_columns:
        for func in window_funcs:
            resolved_min_periods = min_periods
            if isinstance(func, tuple):
                # Ensure the tuple is of length 2 and begins with a string
                if len(func) != 2:
                    raise ValueError(
                        f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}."
                    )
                if not isinstance(func[0], str):
                    raise TypeError(
                        f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}."
                    )

                user_func_name, func = func
                new_column_name = f"{col}_expanding_{user_func_name}"
                resolved_min_periods = min_periods

                # Try handling a lambda function of the form lambda x: x
                if (
                    inspect.isfunction(func)
                    and len(inspect.signature(func).parameters) == 1
                ):
                    try:
                        # Construct expanding window column
                        group_df[new_column_name] = (
                            group_df[col]
                            .expanding(min_periods=resolved_min_periods, **kwargs)
                            .apply(func, raw=True)
                        )
                    except Exception as e:
                        raise Exception(
                            f"An error occurred during the operation of the `{user_func_name}` function in Pandas. Error: {e}"
                        )

                # Try handling a configurable function (e.g. pd_quantile)
                elif isinstance(func, tuple) and func[0] == "configurable":
                    try:
                        # Configurable function should return 4 objects
                        _, func_name, default_kwargs, user_kwargs = func
                    except Exception as e:
                        raise ValueError(
                            f"Unexpected function format. Expected a tuple with format ('configurable', func_name, default_kwargs, user_kwargs). Received: {func}. Original error: {e}"
                        )

                    try:
                        # Define local values that may be required by configurable functions.
                        # If adding a new configurable function in utils.pandas_helpers that necessitates
                        # additional local values, consider updating this dictionary accordingly.
                        local_values = {}
                        # Combine local values with user-provided parameters for the configurable function
                        user_kwargs.update(local_values)
                        # Update the default configurable parameters (without adding new keys)
                        default_kwargs = update_dict(default_kwargs, user_kwargs)
                    except Exception as e:
                        raise ValueError(
                            "Error encountered while updating parameters for the configurable function `{func_name}` passed to `window_func`: {e}"
                        )

                    try:
                        # Get the expanding window function
                        expanding_function = getattr(
                            group_df[col].expanding(
                                min_periods=resolved_min_periods, **kwargs
                            ),
                            func_name,
                            None,
                        )
                    except Exception as e:
                        raise AttributeError(
                            f"The function `{func_name}` tried to access a non-existent attribute or method in Pandas. Error: {e}"
                        )

                    if expanding_function:
                        try:
                            # Apply expanding function to data and store in new column
                            group_df[new_column_name] = expanding_function(
                                **default_kwargs
                            )
                        except Exception as e:
                            raise Exception(
                                f"Failed to construct the expanding window column using function `{user_func_name}`. Error: {e}"
                            )
                else:
                    raise TypeError(
                        f"Unexpected function format for `{user_func_name}`."
                    )

            elif isinstance(func, str):
                new_column_name = f"{col}_expanding_{func}"
                # Get the expanding function (like mean, sum, etc.) specified by `func` for the given column and window settings
                if func == "quantile":
                    new_column_name = f"{col}_expanding_{func}_50"
                    group_df[new_column_name] = (
                        group_df[col]
                        .expanding(min_periods=resolved_min_periods, **kwargs)
                        .quantile(q=0.5)
                    )
                    warnings.warn(
                        "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                        "For more control over the quantile value, consider using the function `pd_quantile()`. "
                        "For example: ('quantile_75', pd_quantile(q=0.75))."
                    )
                else:
                    expanding_function = getattr(
                        group_df[col].expanding(
                            min_periods=resolved_min_periods, **kwargs
                        ),
                        func,
                        None,
                    )
                    # Apply expanding function to data and store in new column
                    if expanding_function:
                        group_df[new_column_name] = expanding_function()
                    else:
                        raise ValueError(f"Invalid function name: {func}")
            else:
                raise TypeError(f"Invalid function type: {type(func)}")

        result_dfs.append(group_df)

    return pd.concat(result_dfs)


def _augment_expanding_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    min_periods: Optional[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    window_size = sorted_frame.height
    expanding_exprs: List[pl.Expr] = []

    for col in value_columns:
        for func in window_funcs:
            resolved_min_periods = min_periods
            if isinstance(func, tuple):
                if len(func) != 2:
                    raise ValueError(
                        f"Expected tuple of length 2, but `window_func` received tuple of length {len(func)}."
                    )
                if not isinstance(func[0], str):
                    raise TypeError(
                        f"Expected first element of tuple to be type 'str', but `window_func` received {type(func[0])}."
                    )

                user_func_name, func_impl = func
                new_column_name = f"{col}_expanding_{user_func_name}"
                resolved_min_periods = min_periods

                if (
                    inspect.isfunction(func_impl)
                    and len(inspect.signature(func_impl).parameters) == 1
                ):
                    rolling_kwargs = {
                        "window_size": window_size,
                        "min_samples": resolved_min_periods,
                    }
                    rolling_kwargs = ensure_polars_rolling_kwargs(rolling_kwargs)

                    try:
                        expr = (
                            pl.col(col)
                            .cast(pl.Float64)
                            .rolling_map(
                                function=func_impl,
                                **rolling_kwargs,
                            )
                        )
                    except Exception as e:
                        raise Exception(
                            f"An error occurred during the operation of the `{user_func_name}` function in Polars. Error: {e}"
                        )
                elif isinstance(func_impl, tuple) and func_impl[0] == "configurable":
                    try:
                        _, func_name, default_kwargs, user_kwargs = func_impl
                        default_kwargs = dict(default_kwargs)
                        user_kwargs = dict(user_kwargs)
                    except Exception as e:
                        raise ValueError(
                            f"Unexpected function format. Expected a tuple with format ('configurable', func_name, default_kwargs, user_kwargs). Received: {func_impl}. Original error: {e}"
                        )

                    try:
                        local_values = {
                            "window_size": window_size,
                            "min_samples": resolved_min_periods,
                        }
                        merged_defaults = update_dict(
                            default_kwargs, {**user_kwargs, **local_values}
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Error encountered while updating parameters for the configurable function `{func_name}` passed to `window_func`: {e}"
                        )

                    merged_defaults = ensure_polars_rolling_kwargs(merged_defaults)

                    try:
                        expr = getattr(pl.col(col), f"rolling_{func_name}")(
                            **merged_defaults
                        )
                    except AttributeError as e:
                        raise AttributeError(
                            f"The function `{user_func_name}` tried to access a non-existent attribute or method in Polars. Error: {e}"
                        )
                    except Exception as e:
                        raise Exception(
                            f"Error during the execution of `{user_func_name}` in Polars. Error: {e}"
                        )
                else:
                    raise TypeError(
                        f"Unexpected function format for `{user_func_name}`."
                    )

                expr = expr.alias(new_column_name)
            elif isinstance(func, str):
                func_name = func
                new_column_name = f"{col}_expanding_{func_name}"
                if not hasattr(pl.col(col), f"rolling_{func_name}"):
                    raise ValueError(
                        f"{func_name} is not a recognized function for Polars."
                    )

                params = {
                    "window_size": window_size,
                    "min_samples": resolved_min_periods,
                }
                params = ensure_polars_rolling_kwargs(params)

                if func_name == "quantile":
                    new_column_name = f"{col}_expanding_{func}_50"
                    expr = getattr(pl.col(col), f"rolling_{func_name}")(
                        quantile=0.5,
                        interpolation="midpoint",
                        **params,
                    )
                    warnings.warn(
                        "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                        "For more control over the quantile value, consider using the function `pl_quantile()`. "
                        "For example: ('quantile_75', pl_quantile(quantile=0.75))."
                    )
                else:
                    expr = getattr(pl.col(col), f"rolling_{func_name}")(**params)

                expr = expr.alias(new_column_name)
            else:
                raise TypeError(f"Invalid function type: {type(func)}")

            if resolved_groups:
                expr = expr.over(resolved_groups)

            expanding_exprs.append(expr)

    augmented = sorted_frame.with_columns(expanding_exprs)
    augmented = augmented.sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented

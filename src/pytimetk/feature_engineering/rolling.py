import pandas as pd
import polars as pl
import pandas_flavor as pf
import inspect
import warnings

from typing import Callable, List, Optional, Sequence, Tuple, Union

try:  # Optional cudf dependency for GPU acceleration
    import cudf  # type: ignore
    from cudf.core.groupby.groupby import DataFrameGroupBy as CudfDataFrameGroupBy
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore
    CudfDataFrameGroupBy = None  # type: ignore

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
def augment_rolling(
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
    window: Union[int, Tuple[int, int], List[int]] = 2,
    min_periods: Optional[int] = None,
    engine: Optional[str] = "auto",
    center: bool = False,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Apply one or more Series-based rolling functions and window sizes to one or more columns of a DataFrame.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input data to be processed. Can be a Pandas DataFrame or a GroupBy
        object.
    date_column : str
        Name of the datetime column. Data is sorted by this column within each
        group.
    value_column : Union[str, list]
        Column(s) to which the rolling window functions should be applied. Can
        be a single column name or a list.
    window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
        The `window_func` parameter in the `augment_rolling` function specifies
        the function(s) to be applied to the rolling windows of the value
        column(s).

        1. It can be either:
            - A string representing the name of a standard function (e.g.,
              'mean', 'sum').

        2. For custom functions:
            - Provide a list of tuples. Each tuple should contain a custom name
              for the function and the function itself.
            - Each custom function should accept a Pandas Series as its input
              and operate on that series.
              Example: ("range", lambda x: x.max() - x.min())

        (See more Examples below.)

        Note: If your function needs to operate on multiple columns (i.e., it
              requires access to a DataFrame rather than just a Series),
              consider using the `augment_rolling_apply` function in this library.
    window : Union[int, tuple, list], optional, default 2
        Specifies the size of the rolling windows.
        - An integer applies the same window size to all columns in `value_column`.
        - A tuple generates windows from the first to the second value (inclusive).
        - A list of integers designates multiple window sizes for each respective
          column.
    min_periods : int, optional, default None
        Minimum observations in the window to have a value. Defaults to the
        window size. If set, a value will be produced even if fewer observations
        are present than the window size.
    center : bool, optional, default False
        If `True`, the rolling window will be centered on the current value. For
        even-sized windows, the window will be left-biased. Otherwise, it uses a trailing window.
    threads : int, optional, default 1
        Number of threads to use for parallel processing. If `threads` is set to
        1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
    show_progress : bool, optional, default True
        If `True`, a progress bar will be displayed during parallel processing.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : {"auto", "pandas", "polars", "cudf"}, optional, default "auto"
        Specifies the backend computation library for augmenting rolling window
        functions. When "auto" the backend is inferred from the input data type.
        Use "pandas" or "polars" to force a specific backend.

    Returns
    -------
    pd.DataFrame
        The `augment_rolling` function returns a DataFrame with new columns for
        each applied function, window size, and value column.

    Notes
    -----
    ## Performance

    This function uses parallel processing to speed up computation for large
    datasets with many time series groups:

    Parallel processing has overhead and may not be faster on small datasets.

    To use parallel processing, set `threads = -1` to use all available processors.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np

    df = tk.load_dataset("m4_daily", parse_dates = ['date'])
    ```

    ```{python}
    # Example 1 - Using a single window size and a single function name, pandas engine
    # This example demonstrates the use of both string-named functions and lambda
    # functions on a rolling window. We specify a list of window sizes: [2,7].
    # As a result, the output will have computations for both window sizes 2 and 7.
    # Note - It's preferred to use built-in or configurable functions instead of
    # lambda functions for performance reasons.

    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date',
                value_column = 'value',
                window = [2,7],  # Specifying multiple window sizes
                window_func = [
                    'mean',  # Built-in mean function
                    ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                ],
                threads = 1,  # Disabling parallel processing
                engine = 'pandas'  # Using pandas engine
            )
    )
    display(rolled_df)
    ```

    ```{python}
    # Example 2 - Multiple groups, pandas engine
    # Example showcasing the use of string function names and lambda functions
    # applied on rolling windows. The `window` tuple (1,3) will generate window
    # sizes of 1, 2, and 3.
    # Note - It's preferred to use built-in or configurable functions instead of
    # lambda functions for performance reasons.

    rolled_df = (
        df
            .groupby('id')
            .augment_rolling(
                date_column = 'date',
                value_column = 'value',
                window = (1,3),  # Specifying a range of window sizes
                window_func = [
                    'mean',  # Using built-in mean function
                    ('std', lambda x: x.std())  # Lambda function for standard deviation
                ],
                threads = 1,  # Disabling parallel processing
                engine = 'pandas'  # Using pandas engine
            )
    )
    display(rolled_df)
    ```

    ```{python}
    # Example 3 - Multiple groups, polars engine

    import polars as pl


    rolled_df = (
        pl.from_pandas(df)
            .group_by('id')
            .tk.augment_rolling(
                date_column = 'date',
                value_column = 'value',
                window = (1,3),  # Specifying a range of window sizes
                window_func = [
                    'mean',  # Using built-in mean function
                    'std',  # Using built-in standard deviation function
                ],
            )
    )
    display(rolled_df)
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

    if not isinstance(window, (int, tuple, list)):
        raise TypeError("`window` must be an integer, tuple, or list.")
    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, tuple):
        windows = list(range(window[0], window[1] + 1))
    else:
        windows = [int(w) for w in window]

    window_funcs = list(window_func) if isinstance(window_func, list) else [window_func]

    threads_resolved = get_threads(threads)

    if engine_resolved == "pandas":
        if isinstance(prepared_data, pd.core.groupby.generic.DataFrameGroupBy):
            original_index = resolve_pandas_groupby_frame(prepared_data).index.copy()
        else:
            original_index = prepared_data.index.copy()

        prepared_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )

        result = _augment_rolling_pandas(
            prepared_data,
            date_column,
            value_columns,
            window_funcs,
            windows,
            min_periods,
            center,
            threads_resolved,
            show_progress,
            **kwargs,
        )
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Rolling augmentation must return a pandas DataFrame.")

        if result.index.has_duplicates:
            result = result.loc[~result.index.duplicated(keep="last")]

        result = result.reindex(original_index)

        if reduce_memory:
            result = reduce_memory_usage(result)

        restored = restore_output_type(result, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored

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

        allowed_cudf_kwargs = {"min_periods"}
        unsupported_kwargs = set(kwargs) - allowed_cudf_kwargs
        min_periods_override: Optional[int] = kwargs.get("min_periods")

        builtin_funcs: List[str] = []
        custom_window_funcs: List[Tuple[str, Callable]] = []
        for func in window_funcs:
            if isinstance(func, str):
                if func not in supported_funcs:
                    fallback_reason = (
                        f"cudf rolling does not support '{func}' aggregation."
                    )
                    break
                builtin_funcs.append(func)
            elif (
                isinstance(func, tuple)
                and len(func) == 2
                and isinstance(func[0], str)
                and callable(func[1])
            ):
                custom_window_funcs.append(func)
            else:
                fallback_reason = "Unsupported rolling function specification for cudf."
                break

        cudf_df: Optional["cudf.DataFrame"] = None
        if isinstance(prepared_data, cudf.DataFrame):
            cudf_df = prepared_data
        elif (
            CudfDataFrameGroupBy is not None
            and isinstance(prepared_data, CudfDataFrameGroupBy)
        ):
            cudf_df = prepared_data.obj.copy(deep=True)
        if isinstance(prepared_data, tuple) and len(prepared_data) == 2:
            # Defensive: convert_to_engine never returns tuple, but keep for safety
            fallback_reason = "Unsupported cudf object returned by engine conversion."
        elif cudf_df is None:
            fallback_reason = "Unsupported cudf object type for augment_rolling."
        elif unsupported_kwargs:
            fallback_reason = (
                "Unsupported cudf rolling kwargs: "
                + ", ".join(sorted(unsupported_kwargs))
            )

        if fallback_reason is not None:
            warnings.warn(
                f"augment_rolling cudf path: {fallback_reason}. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_rolling_pandas(
                pandas_input,
                date_column,
                value_columns,
                window_funcs,
                windows,
                min_periods,
                center,
                threads_resolved,
                show_progress,
                **kwargs,
            )
        else:
            result = _augment_rolling_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                value_columns=value_columns,
                window_funcs=builtin_funcs,
                windows=windows,
                min_periods=min_periods,
                min_periods_override=min_periods_override,
                center=center,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )

            if custom_window_funcs:
                # Use pandas implementation to compute custom functions and merge back
                pandas_view = result.to_pandas()
                pandas_sorted, _ = sort_dataframe(
                    pandas_view, date_column, keep_grouped_df=True
                )
                pandas_augmented = _augment_rolling_pandas(
                    pandas_sorted,
                    date_column,
                    value_columns,
                    custom_window_funcs,
                    windows,
                    min_periods,
                    center,
                    threads_resolved,
                    show_progress,
                    **kwargs,
                )
                pandas_augmented = pandas_augmented.reindex(pandas_view.index)
                custom_column_names: List[str] = []
                for col in value_columns:
                    for name, _ in custom_window_funcs:
                        for window_size in windows:
                            custom_column_names.append(
                                f"{col}_rolling_{name}_win_{window_size}"
                            )
                for column_name in custom_column_names:
                    if column_name in pandas_augmented.columns:
                        result[column_name] = cudf.Series(pandas_augmented[column_name])

        restored = restore_output_type(result, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored.sort_index()

        return restored

    if engine_resolved == "polars":
        result_polars = _augment_rolling_polars(
            prepared_data,
            date_column,
            value_columns,
            window_funcs,
            windows,
            min_periods,
            center,
            conversion.group_columns,
            conversion.row_id_column,
        )

        restored = restore_output_type(result_polars, conversion)

        if isinstance(restored, pd.DataFrame):
            return restored.sort_index()

        return restored

    raise ValueError("Invalid engine. Use 'pandas', 'polars', or 'cudf'.")


def _augment_rolling_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    value_columns: List[str],
    window_funcs: List[str],
    windows: List[int],
    min_periods: Optional[int],
    min_periods_override: Optional[int],
    center: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf rolling backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)

    for col in value_columns:
        if not cudf.api.types.is_numeric_dtype(df_sorted[col]):
            df_sorted[col] = df_sorted[col].astype("float64")

        for window_size in windows:
            resolved_min = (
                min_periods_override
                if min_periods_override is not None
                else (min_periods if min_periods is not None else window_size)
            )

            if group_columns:
                rolling_obj = (
                    df_sorted.groupby(list(group_columns), sort=False)[col]
                    .rolling(
                        window=window_size,
                        min_periods=resolved_min,
                        center=center,
                    )
                )
                for func in window_funcs:
                    new_column_name = f"{col}_rolling_{func}_win_{window_size}"
                    result_series = getattr(rolling_obj, func)().reset_index(drop=True)
                    df_sorted[new_column_name] = result_series
            else:
                rolling_obj = df_sorted[col].rolling(
                    window=window_size,
                    min_periods=resolved_min,
                    center=center,
                )
                for func in window_funcs:
                    new_column_name = f"{col}_rolling_{func}_win_{window_size}"
                    result_series = getattr(rolling_obj, func)()
                    df_sorted[new_column_name] = result_series

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_rolling_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    windows: List[int],
    min_periods: Optional[int] = None,
    center: bool = False,
    threads: int = 1,
    show_progress: bool = True,
    **kwargs,
) -> pd.DataFrame:
    # Create a fresh copy of the data, leaving the original untouched
    data_copy = (
        data.copy()
        if isinstance(data, pd.DataFrame)
        else resolve_pandas_groupby_frame(data).copy()
    )

    # Group data if it's a GroupBy object; otherwise, prepare it for the rolling calculations
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.groupby(group_names)

        # Check if the data is grouped and threads are set to 1. If true, handle it without parallel processing.
        if threads == 1:
            func = partial(
                _process_single_roll,
                value_columns=value_columns,
                window_funcs=window_funcs,
                windows=windows,
                min_periods=min_periods,
                center=center,
                **kwargs,
            )

            # Use tqdm to display progress for the loop
            result_dfs = [
                func(group)
                for _, group in conditional_tqdm(
                    grouped,
                    total=len(grouped),
                    desc="Calculating Rolling...",
                    display=show_progress,
                )
            ]
        else:
            # Prepare to use pathos.multiprocessing
            pool = ProcessingPool(threads)

            # Use partial to "freeze" arguments for _process_single_roll
            func = partial(
                _process_single_roll,
                value_columns=value_columns,
                window_funcs=window_funcs,
                windows=windows,
                min_periods=min_periods,
                center=center,
                **kwargs,
            )

            result_dfs = list(
                conditional_tqdm(
                    pool.map(func, (group for _, group in grouped)),
                    total=len(grouped),
                    desc="Calculating Rolling...",
                    display=show_progress,
                )
            )
    else:
        result_dfs = [
            _process_single_roll(
                data_copy,
                value_columns,
                window_funcs,
                windows,
                min_periods,
                center,
                **kwargs,
            )
        ]

    result_df = pd.concat(result_dfs).sort_index()  # Sort by the original index
    return result_df


def _process_single_roll(
    group_df: pd.DataFrame,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    windows: List[int],
    min_periods: Optional[int],
    center: bool,
    **kwargs,
) -> pd.DataFrame:
    result = group_df.copy()
    for value_col in value_columns:
        min_periods_state = min_periods
        for window_size in windows:
            resolved_min_periods = (
                window_size if min_periods_state is None else min_periods_state
            )
            for func in window_funcs:
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

                    user_func_name, func_impl = func
                    new_column_name = (
                        f"{value_col}_rolling_{user_func_name}_win_{window_size}"
                    )

                    # Try handling a lambda function of the form lambda x: x
                    if (
                        inspect.isfunction(func_impl)
                        and len(inspect.signature(func_impl).parameters) == 1
                    ):
                        try:
                            # Construct rolling window column
                            result[new_column_name] = (
                                result[value_col]
                                .rolling(
                                    window=window_size,
                                    min_periods=resolved_min_periods,
                                    center=center,
                                    **kwargs,
                                )
                                .apply(func_impl, raw=True)
                            )
                        except Exception as e:
                            raise Exception(
                                f"An error occurred during the operation of the `{user_func_name}` function in Pandas. Error: {e}"
                            )

                    # Try handling a configurable function (e.g. pd_quantile)
                    elif (
                        isinstance(func_impl, tuple) and func_impl[0] == "configurable"
                    ):
                        try:
                            # Configurable function should return 4 objects
                            _, func_name, default_kwargs, user_kwargs = func_impl
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
                            # Get the rolling window function
                            rolling_function = getattr(
                                group_df[value_col].rolling(
                                    window=window_size,
                                    min_periods=resolved_min_periods,
                                    center=center,
                                    **kwargs,
                                ),
                                func_name,
                                None,
                            )
                        except Exception as e:
                            raise AttributeError(
                                f"The function `{func_name}` tried to access a non-existent attribute or method in Pandas. Error: {e}"
                            )

                        if rolling_function:
                            try:
                                # Apply rolling function to data and store in new column
                                result[new_column_name] = rolling_function(
                                    **default_kwargs
                                )
                            except Exception as e:
                                raise Exception(
                                    f"Failed to construct the rolling window column using function `{user_func_name}`. Error: {e}"
                                )
                    else:
                        raise TypeError(
                            f"Unexpected function format for `{user_func_name}`."
                        )

                    min_periods_state = resolved_min_periods
                    continue

                if isinstance(func, str):
                    new_column_name = f"{value_col}_rolling_{func}_win_{window_size}"
                    if func == "quantile":
                        new_column_name = (
                            f"{value_col}_rolling_{func}_50_win_{window_size}"
                        )
                        result[new_column_name] = (
                            result[value_col]
                            .rolling(
                                window=window_size,
                                min_periods=resolved_min_periods,
                                center=center,
                                **kwargs,
                            )
                            .quantile(q=0.5)
                        )
                        warnings.warn(
                            "You passed 'quantile' as a string-based function, so it defaulted to a 50 percent quantile (0.5). "
                            "For more control over the quantile value, consider using the function `pd_quantile()`. "
                            "For example: ('quantile_75', pd_quantile(q=0.75))."
                        )
                    else:
                        rolling_function = getattr(
                            result[value_col].rolling(
                                window=window_size,
                                min_periods=resolved_min_periods,
                                center=center,
                                **kwargs,
                            ),
                            func,
                            None,
                        )
                        if rolling_function:
                            result[new_column_name] = rolling_function()
                        else:
                            raise ValueError(f"Invalid function name: {func}")
                    min_periods_state = resolved_min_periods
                    continue

                raise TypeError(f"Invalid function type: {type(func)}")
    return result


def _augment_rolling_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_columns: List[str],
    window_funcs: List[Union[str, Tuple[str, Callable]]],
    windows: List[int],
    min_periods: Optional[int],
    center: bool,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    rolling_exprs: List[pl.Expr] = []

    for col in value_columns:
        min_periods_state = min_periods
        for window_size in windows:
            resolved_min_periods = (
                window_size if min_periods_state is None else min_periods_state
            )
            for func in window_funcs:
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
                    new_column_name = (
                        f"{col}_rolling_{user_func_name}_win_{window_size}"
                    )

                    if (
                        inspect.isfunction(func_impl)
                        and len(inspect.signature(func_impl).parameters) == 1
                    ):
                        rolling_kwargs = {
                            "window_size": window_size,
                            "min_samples": resolved_min_periods,
                        }
                        if center:
                            rolling_kwargs["center"] = True
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
                    elif (
                        isinstance(func_impl, tuple) and func_impl[0] == "configurable"
                    ):
                        try:
                            _, func_name, default_kwargs, user_kwargs = func_impl
                            user_kwargs = dict(user_kwargs)
                            default_kwargs = dict(default_kwargs)
                        except Exception as e:
                            raise ValueError(
                                f"Unexpected function format. Expected a tuple with format ('configurable', func_name, default_kwargs, user_kwargs). Received: {func_impl}. Original error: {e}"
                            )

                        try:
                            local_values = {
                                "window_size": window_size,
                                "min_samples": resolved_min_periods,
                            }
                            if center:
                                local_values["center"] = True
                            merged_defaults = update_dict(
                                default_kwargs,
                                {**user_kwargs, **local_values},
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
                    new_column_name = f"{col}_rolling_{func_name}_win_{window_size}"
                    if not hasattr(pl.col(col), f"rolling_{func_name}"):
                        raise ValueError(
                            f"{func_name} is not a recognized function for Polars."
                        )

                    params = {
                        "window_size": window_size,
                        "min_samples": resolved_min_periods,
                    }
                    if center:
                        params["center"] = True
                    params = ensure_polars_rolling_kwargs(params)

                    if func_name == "quantile":
                        new_column_name = f"{col}_rolling_{func}_50_win_{window_size}"
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

                rolling_exprs.append(expr)
            min_periods_state = resolved_min_periods

    augmented = sorted_frame.with_columns(rolling_exprs)
    augmented = augmented.sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented

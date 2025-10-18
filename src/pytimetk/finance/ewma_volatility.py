import pandas as pd
import polars as pl
import numpy as np

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union

try:  # Optional cudf dependency
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
    ensure_row_id_column,
    normalize_engine,
    resolve_pandas_groupby_frame,
    resolve_polars_group_columns,
    restore_output_type,
    conversion_to_pandas,
)
from pytimetk.utils.polars_helpers import collect_lazyframe
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_ewma_volatility(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    decay_factor: float = 0.94,
    window: Union[int, Tuple[int, int], List[int]] = 20,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Calculate Exponentially Weighted Moving Average (EWMA) volatility for a financial time series.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input time-series data. Grouped inputs are processed per group before
        the indicator is appended.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices to calculate volatility.
    decay_factor : float, optional
        Smoothing factor (lambda) for EWMA, between 0 and 1. Higher values give more weight to past data. Default is 0.94 (RiskMetrics standard).
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window to initialize EWMA calculation. For each window value the EWMA volatility is only computed when at least that many observations are available.
        You may provide a single integer or multiple values (via tuple or list). Default is 20.
    reduce_memory : bool, optional
        If True, reduces memory usage before calculation. Default is False.
    engine : {"auto", "pandas", "polars", "cudf"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with added columns:
        - {close_column}_ewma_vol_{window}_{decay_factor}: EWMA volatility calculated using a minimum number of periods equal to each specified window.

    Notes
    -----
    EWMA volatility emphasizes recent price movements and is computed recursively as:

        σ²_t = (1 - λ) * r²_t + λ * σ²_{t-1}

    where r_t is the log return. By using the `min_periods` (set to the provided window value) we ensure that the EWMA is only calculated after enough observations have accumulated.

    References:

    - https://www.investopedia.com/articles/07/ewma.asp

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])

    # Example 1 - Calculate EWMA volatility for a single stock

    df.query("symbol == 'AAPL'").augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```

    ```{python}
    # Example 2 - Calculate EWMA volatility for multiple stocks
    df.groupby('symbol').augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```

    ```{python}
    # Example 3 - Calculate EWMA volatility using Polars engine
    pl_df = pl.from_pandas(df.query("symbol == 'AAPL'"))
    pl_df.tk.augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```

    ```{python}
    # Example 4 - Calculate EWMA volatility for multiple stocks using Polars engine
    pl_df_full = pl.from_pandas(df)
    pl_df_full.group_by('symbol').tk.augment_ewma_volatility(
        date_column='date',
        close_column='close',
        decay_factor=0.94,
        window=[20, 50]
    ).glimpse()
    ```
    """

    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    if not 0 < decay_factor < 1:
        raise ValueError("decay_factor must be between 0 and 1.")

    window_list = _normalize_windows(window)

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

    if conversion_engine == "pandas":
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_ewma_volatility_pandas(
            data=sorted_data,
            close_column=close_column,
            decay_factor=decay_factor,
            windows=window_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_ewma_volatility. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_ewma_volatility_pandas(
                data=pandas_input,
                close_column=close_column,
                decay_factor=decay_factor,
                windows=window_list,
            )
        else:
            result = _augment_ewma_volatility_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                decay_factor=decay_factor,
                windows=window_list,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_ewma_volatility_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            decay_factor=decay_factor,
            windows=window_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_ewma_volatility_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
    decay_factor: float,
    windows: List[int],
) -> pd.DataFrame:
    """Pandas implementation of EWMA volatility calculation with varying minimum periods."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        ratio = df[close_column] / df[close_column].shift(1)
        ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
        df["log_returns"] = np.log(ratio)
        df["squared_returns"] = df["log_returns"] ** 2

        for win in windows:
            col_name = f"{close_column}_ewma_vol_{win}_{decay_factor:.2f}"
            ewma_variance = df["squared_returns"].ewm(
                alpha=1 - decay_factor, adjust=False, min_periods=win
            ).mean()
            df[col_name] = np.sqrt(ewma_variance)

        return df.drop(columns=["log_returns", "squared_returns"])

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        df = resolve_pandas_groupby_frame(data).copy()
        shifted = df.groupby(group_names)[close_column].shift(1)
        ratio = df[close_column] / shifted
        ratio = ratio.replace([0, np.inf, -np.inf], np.nan)
        df["log_returns"] = np.log(ratio)
        df["squared_returns"] = df["log_returns"] ** 2

        for win in windows:
            col_name = f"{close_column}_ewma_vol_{win}_{decay_factor:.2f}"
            ewma_variance = (
                df.groupby(group_names)["squared_returns"]
                .ewm(alpha=1 - decay_factor, adjust=False, min_periods=win)
                .mean()
            )
            df[col_name] = (
                ewma_variance.apply(np.sqrt).reset_index(level=0, drop=True)
            )

        return df.drop(columns=["log_returns", "squared_returns"])

    raise TypeError("Unsupported data type passed to _augment_ewma_volatility_pandas.")


def _augment_ewma_volatility_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    decay_factor: float,
    windows: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required to execute the cudf ewma volatility backend."
        )

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        group_list = list(group_columns)
        prev_close = df_sorted.groupby(group_list, sort=False)[close_column].shift(1)
    else:
        group_list = None
        prev_close = df_sorted[close_column].shift(1)

    ratio = df_sorted[close_column] / prev_close
    ratio = ratio.where(prev_close != 0)
    log_returns = cudf.Series(np.log(ratio))
    df_sorted["__ewma_log_returns"] = log_returns
    df_sorted["__ewma_squared_returns"] = log_returns ** 2

    alpha = 1 - decay_factor

    for win in windows:
        col_name = f"{close_column}_ewma_vol_{win}_{decay_factor:.2f}"
        if group_list:
            result_series = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")
            for _, group_df in df_sorted.groupby(group_list, sort=False):
                idx = group_df.index
                variance = group_df["__ewma_squared_returns"].ewm(
                    alpha=alpha,
                    adjust=False,
                    min_periods=win,
                ).mean()
                result_series.loc[idx] = variance.sqrt()
            df_sorted[col_name] = result_series
        else:
            variance = df_sorted["__ewma_squared_returns"].ewm(
                alpha=alpha,
                adjust=False,
                min_periods=win,
            ).mean()
            df_sorted[col_name] = variance.sqrt()

    df_sorted = df_sorted.drop(columns=["__ewma_log_returns", "__ewma_squared_returns"])

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_ewma_volatility_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    decay_factor: float,
    windows: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    """Polars implementation of EWMA volatility calculation with varying minimum periods."""

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    def _maybe_over(expr: pl.Expr) -> pl.Expr:
        if resolved_groups:
            return expr.over(resolved_groups)
        return expr

    lazy_frame = sorted_frame.lazy()
    temp_columns = [
        "__ewma_log_price",
        "__ewma_prev_log_price",
        "__ewma_log_return",
        "__ewma_sq_return",
    ]

    lazy_frame = lazy_frame.with_columns(
        pl.col(close_column).log().alias("__ewma_log_price")
    )

    lazy_frame = lazy_frame.with_columns(
        _maybe_over(pl.col("__ewma_log_price").shift(1)).alias(
            "__ewma_prev_log_price"
        )
    )

    lazy_frame = lazy_frame.with_columns(
        (
            pl.col("__ewma_log_price") - pl.col("__ewma_prev_log_price")
        ).alias("__ewma_log_return")
    )

    lazy_frame = lazy_frame.with_columns(
        pl.when(pl.col("__ewma_log_return").is_finite())
        .then(pl.col("__ewma_log_return"))
        .otherwise(None)
        .alias("__ewma_log_return")
    )

    lazy_frame = lazy_frame.with_columns(
        (pl.col("__ewma_log_return") ** 2).alias("__ewma_sq_return")
    )

    for win in windows:
        vol_expr = _maybe_over(
            pl.col("__ewma_sq_return")
            .ewm_mean(alpha=1 - decay_factor, adjust=False, min_periods=win)
        ).sqrt()
        lazy_frame = lazy_frame.with_columns(
            vol_expr.alias(f"{close_column}_ewma_vol_{win}_{decay_factor:.2f}")
        )

    lazy_frame = lazy_frame.drop(temp_columns)

    result = collect_lazyframe(lazy_frame).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result


def _normalize_windows(window: Union[int, Tuple[int, int], List[int]]) -> List[int]:
    if isinstance(window, int):
        return [window]
    if isinstance(window, tuple):
        if len(window) != 2:
            raise ValueError("Expected tuple of length 2 for `window`.")
        start, end = window
        return list(range(start, end + 1))
    if isinstance(window, list):
        return [int(w) for w in window]
    raise TypeError(
        f"Invalid window specification: type: {type(window)}. Please use int, tuple, or list."
    )

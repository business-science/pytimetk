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
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from pytimetk.utils.polars_helpers import collect_lazyframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_hurst_exponent(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    window: Union[int, Tuple[int, int], List[int]] = 100,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Calculate the Hurst Exponent on a rolling window for a financial time series. Used for detecting trends and mean-reversion.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input time-series data. Grouped inputs are processed per group before
        the exponent is appended.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices to calculate the Hurst Exponent.
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window for Hurst Exponent calculation. Accepts int, tuple (start, end), or list. Default is 100.
    reduce_memory : bool, optional
        If True, reduces memory usage before calculation. Default is False.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with added columns:
        - {close_column}_hurst_{window}: Hurst Exponent for each window size

    Notes
    -----
    The Hurst Exponent measures the long-term memory of a time series:

    - H < 0.5: Mean-reverting behavior
    - H ≈ 0.5: Random walk (no persistence)
    - H > 0.5: Trending or persistent behavior
    Computed using a simplified R/S analysis over rolling windows.

    References:

    - https://en.wikipedia.org/wiki/Hurst_exponent

    Examples:
    ---------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock Hurst Exponent with pandas engine
    hurst_df = (
        df.query("symbol == 'AAPL'")
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=[100, 200]
        )
    )
    hurst_df.glimpse()
    ```

    ```{python}
    # Example 2 - Multiple stocks with groupby using pandas engine
    hurst_df = (
        df.groupby('symbol')
        .augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=100
        )
    )
    hurst_df.glimpse()
    ```

    ```{python}
    # Example 3 - Single stock Hurst Exponent with polars engine
    pl_single = pl.from_pandas(df.query("symbol == 'AAPL'"))
    hurst_df = pl_single.tk.augment_hurst_exponent(
        date_column='date',
        close_column='close',
        window=[100, 200]
    )
    hurst_df.glimpse()
    ```

    ```{python}
    # Example 4 - Multiple stocks with groupby using polars engine
    pl_grouped = pl.from_pandas(df)
    hurst_df = (
        pl_grouped.group_by('symbol')
        .tk.augment_hurst_exponent(
            date_column='date',
            close_column='close',
            window=100,
        )
    )
    hurst_df.glimpse()
    ```
    """

    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    windows = _normalize_windows(window)

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
        result = _augment_hurst_exponent_pandas(
            data=sorted_data,
            close_column=close_column,
            windows=windows,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_hurst_exponent. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_hurst_exponent_pandas(
                data=pandas_input,
                close_column=close_column,
                windows=windows,
            )
        else:
            result = _augment_hurst_exponent_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                windows=windows,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_hurst_exponent_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _hurst_from_array(values: np.ndarray, min_size: int = 8) -> float:
    if values.size < min_size or np.all(values == values[0]):
        return np.nan

    mean = np.mean(values)
    centered = values - mean
    cumulative = np.cumsum(centered)
    r = cumulative.max() - cumulative.min()
    s = np.std(values)

    if s == 0 or r == 0:
        return np.nan

    rs = r / s
    hurst = np.log(rs) / np.log(len(values))
    if hurst < 0 or hurst > 1:
        return np.nan
    return float(hurst)


def _augment_hurst_exponent_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
    windows: List[int],
) -> pd.DataFrame:
    """Pandas implementation of Hurst Exponent calculation."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names: Optional[List[str]] = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        df = resolve_pandas_groupby_frame(data).copy()
    else:
        raise TypeError("Unsupported data type passed to _augment_hurst_exponent_pandas.")

    col = close_column

    for window in windows:
        if group_names:
            df[f"{col}_hurst_{window}"] = (
                df.groupby(group_names)[col]
                .rolling(window=window, min_periods=window)
                .apply(lambda x: _hurst_from_array(x), raw=True)
                .reset_index(level=0, drop=True)
            )
        else:
            df[f"{col}_hurst_{window}"] = (
                df[col]
                .rolling(window=window, min_periods=window)
                .apply(lambda x: _hurst_from_array(x), raw=True)
            )

    return df


def _augment_hurst_exponent_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    windows: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required to execute the cudf hurst exponent backend."
        )

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        group_list = list(group_columns)
        groups = list(df_sorted.groupby(group_list, sort=False))
    else:
        group_list = None
        groups = [(None, df_sorted)]

    for window in windows:
        result_series = cudf.Series(np.nan, index=df_sorted.index, dtype="float64")
        for _, group_df in groups:
            values = group_df[close_column].to_numpy()
            hurst_values = np.full(len(values), np.nan, dtype="float64")
            for idx in range(window - 1, len(values)):
                window_slice = values[idx - window + 1 : idx + 1]
                hurst_values[idx] = _hurst_from_array(window_slice)
            result_series.loc[group_df.index] = hurst_values
        df_sorted[f"{close_column}_hurst_{window}"] = result_series

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_hurst_exponent_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    """Polars implementation of Hurst Exponent calculation."""

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    col = close_column

    def hurst_udf(series):
        """User-defined function for Hurst Exponent in Polars."""
        series = series.to_numpy()
        n = len(series)
        if n < 8 or np.all(series == series[0]):  # Minimum size or constant
            return np.nan

        mean = np.mean(series)
        y = series - mean
        z = np.cumsum(y)
        r = np.max(z) - np.min(z)
        s = np.std(series)

        if s == 0 or r == 0:
            return np.nan

        rs = r / s
        h = np.log(rs) / np.log(n)
        return h if 0 <= h <= 1 else np.nan

    def _maybe_over(expr: pl.Expr) -> pl.Expr:
        if resolved_groups:
            return expr.over(resolved_groups)
        return expr

    lazy_frame = sorted_frame.lazy()

    for window in windows:
        expr = _maybe_over(
            pl.col(col).rolling_map(
                hurst_udf,
                window_size=window,
                min_periods=window,
            )
        ).alias(f"{col}_hurst_{window}")
        lazy_frame = lazy_frame.with_columns(expr)

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

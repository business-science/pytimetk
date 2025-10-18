import pandas as pd
import polars as pl

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


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_bbands(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 20,
    std_dev: Union[int, float, List[float]] = 2,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame, "cudf.DataFrame"]:
    """
    Calculate Bollinger Bands for pandas or polars data.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        bands are appended.
    date_column : str
        Name of the column containing date information. Used to preserve the
        original ordering of results.
    close_column : str
        Name of the closing price column used to compute the moving average and
        standard deviation.
    periods : int, tuple, or list, optional
        Rolling window lengths. An integer adds a single window, a tuple
        ``(start, end)`` expands to every integer in the inclusive range, and a
        list provides explicit windows. Defaults to ``20``.
    std_dev : float, int, or list, optional
        Number(s) of standard deviations used when constructing the upper and
        lower bands. Integers are converted to floats. Defaults to ``2``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with middle/upper/lower band columns appended for each
        ``period`` and ``std_dev`` combination. The return type matches the
        input backend (pandas or polars).

    Notes
    -----
    The middle band is the rolling mean of ``close_column``. The upper band is
    the middle band plus ``std_dev`` times the rolling standard deviation, and
    the lower band subtracts the same quantity. Rolling statistics are
    calculated with a minimum window equal to ``period`` which matches the
    behaviour of the pandas implementation.

    Examples
    --------
    ```{python}
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Augment a pandas DataFrame (engine inferred)
    bbands_df = df.groupby("symbol").augment_bbands(
        date_column="date",
        close_column="close",
        periods=[20, 40],
        std_dev=[1.5, 2.0],
    )

    # Polars DataFrame using the tk accessor
    bbands_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_bbands(
            date_column="date",
            close_column="close",
            periods=(10, 15),
            std_dev=2,
        )
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    period_list = _normalize_periods(periods)
    std_list = _normalize_std_dev(std_dev)

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
        result = _augment_bbands_pandas(
            data=sorted_data,
            close_column=close_column,
            periods=period_list,
            std_dev=std_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_bbands. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_bbands_pandas(
                data=pandas_input,
                close_column=close_column,
                periods=period_list,
                std_dev=std_list,
            )
        else:
            result = _augment_bbands_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                periods=period_list,
                std_dev=std_list,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_bbands_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            periods=period_list,
            std_dev=std_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_bbands_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    periods: List[int],
    std_dev: List[float],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError("cudf is required to execute the cudf bbands backend.")

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    group_list = list(group_columns) if group_columns else None

    for period in periods:
        if group_list:
            rolling_obj = (
                df_sorted.groupby(group_list, sort=False)[close_column]
                .rolling(window=period, min_periods=period)
            )
            ma_series = rolling_obj.mean().reset_index(drop=True)
            std_series = rolling_obj.std().reset_index(drop=True)
        else:
            ma_series = df_sorted[close_column].rolling(window=period, min_periods=period).mean()
            std_series = df_sorted[close_column].rolling(window=period, min_periods=period).std()

        for sd in std_dev:
            fmt = _format_sd(sd)
            offset = std_series * float(sd)
            df_sorted[f"{close_column}_bband_middle_{period}_{fmt}"] = ma_series
            df_sorted[f"{close_column}_bband_upper_{period}_{fmt}"] = ma_series + offset
            df_sorted[f"{close_column}_bband_lower_{period}_{fmt}"] = ma_series - offset

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_bbands_pandas(
    data,
    close_column: str,
    periods: List[int],
    std_dev: List[float],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for period in periods:
            ma = df[close_column].rolling(period).mean()
            std = df[close_column].rolling(period).std()
            for sd in std_dev:
                fmt = _format_sd(sd)
                df[f"{close_column}_bband_middle_{period}_{fmt}"] = ma
                df[f"{close_column}_bband_upper_{period}_{fmt}"] = ma + (std * sd)
                df[f"{close_column}_bband_lower_{period}_{fmt}"] = ma - (std * sd)
        return df

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        base_df = resolve_pandas_groupby_frame(data).copy()

        for period in periods:
            rolling = (
                base_df.groupby(group_names)[close_column]
                .rolling(period)
                .agg(["mean", "std"])
                .rename(columns={"mean": "_mean", "std": "_std"})
            )
            rolling = rolling.reset_index(level=0, drop=True)

            for sd in std_dev:
                fmt = _format_sd(sd)
                base_df[f"{close_column}_bband_middle_{period}_{fmt}"] = rolling["_mean"]
                base_df[f"{close_column}_bband_upper_{period}_{fmt}"] = (
                    rolling["_mean"] + rolling["_std"] * sd
                )
                base_df[f"{close_column}_bband_lower_{period}_{fmt}"] = (
                    rolling["_mean"] - rolling["_std"] * sd
                )

        return base_df

    raise TypeError("Unsupported data type passed to _augment_bbands_pandas.")


def _augment_bbands_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    periods: List[int],
    std_dev: List[float],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    band_exprs = []
    for period in periods:
        mean_expr = pl.col(close_column).rolling_mean(window_size=period)
        std_expr = pl.col(close_column).rolling_std(window_size=period)

        if resolved_groups:
            mean_expr = mean_expr.over(resolved_groups)
            std_expr = std_expr.over(resolved_groups)

        for sd in std_dev:
            fmt = _format_sd(sd)
            middle_alias = f"{close_column}_bband_middle_{period}_{fmt}"
            upper_alias = f"{close_column}_bband_upper_{period}_{fmt}"
            lower_alias = f"{close_column}_bband_lower_{period}_{fmt}"

            band_exprs.append(mean_expr.alias(middle_alias))
            band_exprs.append((mean_expr + std_expr * sd).alias(upper_alias))
            band_exprs.append((mean_expr - std_expr * sd).alias(lower_alias))

    augmented = sorted_frame.with_columns(band_exprs).sort(row_col)

    if generated:
        augmented = augmented.drop(row_col)

    return augmented


def _normalize_periods(periods: Union[int, Tuple[int, int], List[int]]) -> List[int]:
    if isinstance(periods, int):
        return [periods]
    if isinstance(periods, tuple):
        if len(periods) != 2:
            raise ValueError("Expected tuple of length 2 for `periods`.")
        start, end = periods
        return list(range(start, end + 1))
    if isinstance(periods, list):
        return [int(p) for p in periods]
    raise TypeError(
        f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list."
    )


def _normalize_std_dev(std_dev: Union[int, float, List[float]]) -> List[float]:
    if isinstance(std_dev, (int, float)):
        return [float(std_dev)]
    if isinstance(std_dev, list):
        return [float(sd) for sd in std_dev]
    raise TypeError(
        f"Invalid std_dev specification: type: {type(std_dev)}. Please use float or list."
    )


def _format_sd(sd: float) -> str:
    return f"{sd:.1f}"

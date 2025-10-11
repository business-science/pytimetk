import numpy as np
import pandas as pd
import polars as pl

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union

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
    resolve_polars_group_columns,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_adx(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 14,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate Average Directional Index (ADX), +DI, and -DI using pandas or polars backends.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        indicators are appended.
    date_column : str
        Name of the column containing date information.
    high_column : str
        Name of the column containing high prices.
    low_column : str
        Name of the column containing low prices.
    close_column : str
        Name of the column containing closing prices. Indicator columns are
        prefixed with this name.
    periods : int, tuple, or list, optional
        Lookback windows for smoothing. Accepts an integer, a tuple specifying
        an inclusive range, or a list of explicit periods. Defaults to ``14``.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with the following columns appended for each ``period``:

        - ``{close_column}_plus_di_{period}``
        - ``{close_column}_minus_di_{period}``
        - ``{close_column}_adx_{period}``

        The return type matches the input backend.

    Notes
    -----
    The implementation follows Wilder's smoothing approach using exponential
    moving averages with ``alpha = 1 / period`` for the true range (TR) and
    directional movement (+DM, -DM) components. Division by zero is guarded by
    returning ``NaN`` when the denominator is zero.

    Examples
    --------
    ```python
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Pandas example (engine inferred)
    adx_df = (
        df.groupby("symbol")
        .augment_adx(
            date_column="date",
            high_column="high",
            low_column="low",
            close_column="close",
            periods=[14, 28],
        )
    )

    # Polars example (engine inferred)
    import polars as pl
    adx_pl = tk.augment_adx(
        data=pl.from_pandas(df.query("symbol == 'AAPL'")),
        date_column="date",
        high_column="high",
        low_column="low",
        close_column="close",
        periods=14,
    )
    ```
    """

    check_dataframe_or_groupby(data)
    check_value_column(data, high_column)
    check_value_column(data, low_column)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    periods_list = _normalize_periods(periods)

    engine_resolved = normalize_engine(engine, data)
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
        sorted_data, _ = sort_dataframe(
            prepared_data, date_column, keep_grouped_df=True
        )
        result = _augment_adx_pandas(
            data=sorted_data,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=periods_list,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_adx_polars(
            data=prepared_data,
            date_column=date_column,
            high_column=high_column,
            low_column=low_column,
            close_column=close_column,
            periods=periods_list,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_adx_pandas(
    data,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        grouped = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        grouped = data.grouper.names
        df = data.obj.copy()
    else:
        raise TypeError("Unsupported data type passed to _augment_adx_pandas.")

    col = close_column

    df["tr"] = pd.concat(
        [
            df[high_column] - df[low_column],
            (df[high_column] - df[col].shift(1)).abs(),
            (df[low_column] - df[col].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["plus_dm"] = np.where(
        (df[high_column] - df[high_column].shift(1))
        > (df[low_column].shift(1) - df[low_column]),
        np.maximum(df[high_column] - df[high_column].shift(1), 0),
        0,
    )
    df["minus_dm"] = np.where(
        (df[low_column].shift(1) - df[low_column])
        > (df[high_column] - df[high_column].shift(1)),
        np.maximum(df[low_column].shift(1) - df[low_column], 0),
        0,
    )

    if grouped is not None:
        grouped_obj = df.groupby(grouped)

    for period in periods:
        alpha = 1 / period
        if grouped is not None:
            tr_smooth = grouped_obj["tr"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
            plus_dm_smooth = grouped_obj["plus_dm"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
            minus_dm_smooth = grouped_obj["minus_dm"].transform(
                lambda s: s.ewm(alpha=alpha, adjust=False).mean()
            )
        else:
            tr_smooth = df["tr"].ewm(alpha=alpha, adjust=False).mean()
            plus_dm_smooth = df["plus_dm"].ewm(alpha=alpha, adjust=False).mean()
            minus_dm_smooth = df["minus_dm"].ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        df[f"{col}_plus_di_{period}"] = plus_di
        df[f"{col}_minus_di_{period}"] = minus_di
        df[f"{col}_adx_{period}"] = adx

    df.drop(columns=["tr", "plus_dm", "minus_dm"], inplace=True)
    return df


def _augment_adx_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    high_column: str,
    low_column: str,
    close_column: str,
    periods: List[int],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    delta_high = pl.col(high_column) - pl.col(high_column).shift(1)
    delta_low = pl.col(low_column).shift(1) - pl.col(low_column)

    tr_expr = pl.max_horizontal(
        pl.col(high_column) - pl.col(low_column),
        (pl.col(high_column) - pl.col(close_column).shift(1)).abs(),
        (pl.col(low_column) - pl.col(close_column).shift(1)).abs(),
    )

    zero = pl.lit(0.0)

    plus_dm_expr = pl.when(delta_high > delta_low).then(
        pl.max_horizontal(delta_high, zero)
    ).otherwise(0.0)
    minus_dm_expr = pl.when(delta_low > delta_high).then(
        pl.max_horizontal(delta_low, zero)
    ).otherwise(0.0)

    def compute(frame: pl.DataFrame) -> pl.DataFrame:

        df = frame.with_columns(
            [
                tr_expr.alias("tr"),
                plus_dm_expr.alias("plus_dm"),
                minus_dm_expr.alias("minus_dm"),
            ]
        )

        for period in periods:
            alpha = 1 / period
            plus_alias = f"{close_column}_plus_di_{period}"
            minus_alias = f"{close_column}_minus_di_{period}"
            adx_alias = f"{close_column}_adx_{period}"

            df = df.with_columns(
                [
                    (
                        100
                        * (
                            pl.col("plus_dm").ewm_mean(alpha=alpha, adjust=False)
                            / pl.col("tr").ewm_mean(alpha=alpha, adjust=False)
                        )
                    ).alias(plus_alias),
                    (
                        100
                        * (
                            pl.col("minus_dm").ewm_mean(alpha=alpha, adjust=False)
                            / pl.col("tr").ewm_mean(alpha=alpha, adjust=False)
                        )
                    ).alias(minus_alias),
                ]
            )

            df = df.with_columns(
                (
                    100
                    * (pl.col(plus_alias) - pl.col(minus_alias)).abs()
                    / (pl.col(plus_alias) + pl.col(minus_alias))
                )
                .ewm_mean(alpha=alpha, adjust=False)
                .alias(adx_alias)
            )

        return df.drop(["tr", "plus_dm", "minus_dm"])

    if resolved_groups:
        group_key = resolved_groups if len(resolved_groups) > 1 else resolved_groups[0]
        result = (
            sorted_frame.group_by(group_key, maintain_order=True)
            .map_groups(compute)
            .sort(row_col)
        )
    else:
        result = compute(sorted_frame).sort(row_col)

    if generated:
        result = result.drop(row_col)

    return result


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

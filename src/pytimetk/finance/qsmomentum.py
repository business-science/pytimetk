import pandas as pd
import numpy as np
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
def augment_qsmomentum(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    roc_fast_period: Union[int, Tuple[int, int], List[int]] = 21,
    roc_slow_period: Union[int, Tuple[int, int], List[int]] = 252,
    returns_period: Union[int, Tuple[int, int], List[int]] = 126,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate Quant Science Momentum (QSM) for pandas or polars inputs.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input financial data. Grouped inputs are processed per group before the
        momentum columns are appended.
    date_column : str
        Name of the column containing date information.
    close_column : str
        Column containing closing prices used to compute QSM.
    roc_fast_period : int, tuple, or list, optional
        Lookback window(s) for the fast Rate of Change (ROC). Accepts a single
        integer, an inclusive tuple range, or a list of explicit periods.
    roc_slow_period : int, tuple, or list, optional
        Lookback window(s) for the slow ROC component.
    returns_period : int, tuple, or list, optional
        Lookback window(s) used when calculating the rolling standard deviation
        of returns.
    reduce_memory : bool, optional
        Attempt to reduce memory usage when operating on pandas data. If a
        polars input is supplied a warning is emitted and no conversion occurs.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with ``{close_column}_qsmom_{fast}_{slow}_{returns}`` columns
        appended for every valid combination. The return type matches the input
        backend.

    Notes
    -----
    QSM measures the difference between slow and fast ROC values normalised by
    the rolling volatility of returns. Only combinations where ``fast < slow``
    and ``returns_period <= slow`` are evaluated. If no combinations satisfy
    these rules a ``ValueError`` is raised to surface the configuration issue.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import polars as pl

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    # Calculate QSM for multiple ROCs using pandas backend
    qsm_pd = (
        df.groupby("symbol")
        .augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period=252,
            returns_period=126,
        )
    )

    # Compute QSM on a polars DataFrame via the tk accessor
    qsm_pl = (
        pl.from_pandas(df.query("symbol == 'AAPL'"))
        .tk.augment_qsmomentum(
            date_column="date",
            close_column="close",
            roc_fast_period=[5, 21],
            roc_slow_period=252,
            returns_period=126,
        )
    )
    ```
    """

    # --- Common checks ---
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)

    # Normalize params to lists
    if isinstance(roc_fast_period, int):
        roc_fast_period = [roc_fast_period]
    elif isinstance(roc_fast_period, tuple):
        roc_fast_period = list(range(roc_fast_period[0], roc_fast_period[1] + 1))
    elif not isinstance(roc_fast_period, list):
        raise ValueError("roc_fast_period must be an int, tuple or list")

    roc_fast_values = _normalize_periods(roc_fast_period, label="roc_fast_period")
    roc_slow_values = _normalize_periods(roc_slow_period, label="roc_slow_period")
    returns_values = _normalize_periods(returns_period, label="returns_period")

    valid_combos = [
        (fp, sp, rp)
        for fp in roc_fast_values
        for sp in roc_slow_values
        for rp in returns_values
        if fp < sp and rp <= sp
    ]

    if not valid_combos:
        raise ValueError(
            "augment_qsmomentum generated no columns. "
            "Ensure roc_fast_period < roc_slow_period and returns_period <= roc_slow_period. "
            f"Got fast={roc_fast_values}, slow={roc_slow_values}, returns={returns_values}."
        )

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
        result = _augment_qsmomentum_pandas(
            data=sorted_data,
            close_column=close_column,
            combos=valid_combos,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_qsmomentum_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            combos=valid_combos,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _calculate_qsmomentum_pandas(
    close, roc_fast_period, roc_slow_period, returns_period
):
    close = pd.Series(close)
    returns = close.pct_change().iloc[-returns_period:]
    std_returns = np.std(returns)

    # Check if the standard deviation is too small:
    if np.abs(std_returns) < 1e-10:
        return np.nan

    # Calculate the rates of change with a small epsilon added to the denominator
    roc_slow_calc = (close.iloc[-roc_fast_period] - close.iloc[-roc_slow_period]) / (
        close.iloc[-roc_slow_period] + 1e-10
    )
    roc_fast_calc = (close.iloc[-1] - close.iloc[-roc_fast_period]) / (
        close.iloc[-roc_fast_period] + 1e-10
    )

    mom = (roc_slow_calc - roc_fast_calc) / std_returns
    return mom


def _calculate_qsmomentum_polars(
    close, roc_fast_period, roc_slow_period, returns_period
):
    close = pl.Series(close)
    returns = close.pct_change()
    returns_last_returns_period = returns.slice(-returns_period, returns_period)
    std_returns = returns_last_returns_period.std()

    # Check if the standard deviation is too small (or undefined)
    if std_returns is None or std_returns < 1e-10:
        return np.nan

    roc_slow_calc = (close[-roc_fast_period] - close[-roc_slow_period]) / (
        close[-roc_slow_period] + 1e-10
    )
    roc_fast_calc = (close[-1] - close[-roc_fast_period]) / (
        close[-roc_fast_period] + 1e-10
    )

    mom = (roc_slow_calc - roc_fast_calc) / std_returns
    return mom
def _augment_qsmomentum_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
    combos: Sequence[Tuple[int, int, int]],
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names: Optional[List[str]] = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = list(data.grouper.names)
        df = data.obj.copy()
    else:
        raise TypeError("Unsupported data type passed to _augment_qsmomentum_pandas.")

    for fp, sp, rp in combos:
        column = f"{close_column}_qsmom_{fp}_{sp}_{rp}"
        if group_names:
            df[column] = (
                df.groupby(group_names)[close_column]
                .rolling(window=sp, min_periods=sp)
                .apply(
                    lambda window: _calculate_qsmomentum_pandas(
                        window, fp, sp, rp
                    ),
                    raw=False,
                )
                .reset_index(level=0, drop=True)
            )
        else:
            df[column] = (
                df[close_column]
                .rolling(window=sp, min_periods=sp)
                .apply(
                    lambda window: _calculate_qsmomentum_pandas(
                        window, fp, sp, rp
                    ),
                    raw=False,
                )
            )

    return df


def _augment_qsmomentum_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    combos: Sequence[Tuple[int, int, int]],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    def compute(frame: pl.DataFrame) -> pl.DataFrame:
        df = frame
        for fp, sp, rp in combos:
            expr = pl.col(close_column).rolling_map(
                lambda series: _calculate_qsmomentum_polars(series, fp, sp, rp),
                window_size=sp,
                min_periods=sp,
            ).alias(f"{close_column}_qsmom_{fp}_{sp}_{rp}")
            df = df.with_columns(expr)
        return df

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


def _normalize_periods(
    periods: Union[int, Tuple[int, int], List[int]],
    label: str,
) -> List[int]:
    if isinstance(periods, int):
        return [periods]
    if isinstance(periods, tuple):
        if len(periods) != 2:
            raise ValueError(f"Expected tuple of length 2 for `{label}`.")
        start, end = periods
        return list(range(start, end + 1))
    if isinstance(periods, list):
        return [int(p) for p in periods]
    raise TypeError(
        f"Invalid {label} specification: type: {type(periods)}. Please use int, tuple, or list."
    )

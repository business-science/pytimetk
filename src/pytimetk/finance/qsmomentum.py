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
    """The function `augment_qsmomentum` calculates Quant Science Momentum for financial data.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter in the `augment_qsmomentum` function is expected to be a pandas DataFrame or a
        pandas DataFrameGroupBy object. This parameter represents the input data on which the momentum
        calculations will be performed.
    date_column : str
        The `date_column` parameter in the `augment_qsmomentum` function refers to the column in your input
        data that contains the dates associated with the financial data. This column is used for time-based
        operations and calculations within the function.
    close_column : str
        The `close_column` parameter in the `augment_qsmomentum` function refers to the column in the input
        DataFrame that contains the closing prices of the financial instrument or asset for which you want
        to calculate the momentum.
    roc_fast_period : Union[int, Tuple[int, int], List[int]], optional
        The `roc_fast_period` parameter in the `augment_qsmomentum` function determines the period used for
        calculating the fast Rate of Change (ROC) momentum indicator.
    roc_slow_period : Union[int, Tuple[int, int], List[int]], optional
        The `roc_slow_period` parameter in the `augment_qsmomentum` function represents the period used for
        calculating the slow rate of change (ROC) in momentum analysis.
    returns_period : Union[int, Tuple[int, int], List[int]], optional
        The `returns_period` parameter in the `augment_qsmomentum` function determines the period over
        which the returns are calculated.
    reduce_memory : bool, optional
        The `reduce_memory` parameter in the `augment_qsmomentum` function is a boolean flag that indicates
        whether memory reduction techniques should be applied to the input data before and after the
        momentum calculation process. If set to `True`, memory reduction methods will be used to optimize
        memory usage, potentially reducing
    engine : str, optional
        The `engine` parameter in the `augment_qsmomentum` function specifies the computation engine to be
        sed for calculating momentum. It can have two possible values: "pandas" or "polars".

    Returns
    -------
        The function `augment_qsmomentum` returns a pandas DataFrame that has been augmented with columns
        representing the Quant Science Momentum (QSM) calculated based on the specified parameters
        such as roc_fast_period, roc_slow_period, and returns_period.

    Notes
    -----

    The Quant Science Momentum (QSM) is a momentum indicator that is calculated based on the Slow Rate of Change (ROC) usually over a 252-day period and the Fast Rate of Change (ROC) usually over a 21-day period.

    The QSM is calculated as the difference between the slow and fast ROCs divided by the standard deviation of the returns over a specified period.

    This provides a measure of momentum that is normalized by the rolling volatility of the returns.

    Examples
    --------
    ``` {python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates = ['date'])

    df.glimpse()
    ```

    ``` {python}
    # PANDAS QS MOMENTUM CALCULATION
    df_qsmom = (
        df
            .query('symbol == "GOOG"')
            .augment_qsmomentum(
                date_column = 'date',
                close_column = 'close',
                roc_fast_period = [1, 5, 21],
                roc_slow_period = 252,
                returns_period = 126,
                engine = "pandas"
            )
    )

    df_qsmom.dropna().glimpse()
    ```

    ``` {python}
    # POLARS QS MOMENTUM CALCULATION
    df_qsmom = (
        df
            .query('symbol == "GOOG"')
            .augment_qsmomentum(
                date_column = 'date',
                close_column = 'close',
                roc_fast_period = [1, 5, 21],
                roc_slow_period = 252,
                returns_period = 126,
                engine = "polars"
            )
    )

    df_qsmom.dropna().glimpse()
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

import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Union
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
def augment_fip_momentum(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    window: Union[int, List[int]] = 252,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
    fip_method: str = "original",
    skip_window: int = 0,  # new parameter to skip the first n periods
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Calculate the "Frog In The Pan" (FIP) momentum metric over one or more rolling windows
    using either the pandas or polars engine, augmenting the DataFrame with FIP columns.

    The FIP momentum is defined as:

    - For `fip_method = 'original'`: FIP = Total Return * (percent of negative returns - percent of positive returns)
    - For `fip_method = 'modified'`: FIP = sign(Total Return) * (percent of positive returns - percent of negative returns)

    An optional parameter, `skip_window`, allows you to skip the first n periods (e.g., one month)
    to mitigate the effects of mean reversion.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        Input pandas DataFrame or grouped DataFrame containing time series data.
    date_column : str
        Name of the column with dates or timestamps.
    close_column : str
        Name of the column with closing prices to calculate returns.
    window : Union[int, List[int]], optional
        Size of the rolling window(s) as an integer or list of integers (default is 252).
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame. Default is False.
    engine : str, optional
        Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
    fip_method : str, optional
        Type of FIP calculation:
        - 'original': Original FIP calculation (default) where negative FIP indicates greater momentum.
        - 'modified': Modified FIP where positive FIP indicates greater momentum.
    skip_window : int, optional
        Number of initial periods to skip (set to NA) for each rolling calculation. Default is 0.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with FIP momentum columns:

        - {close_column}_fip_momentum_{w}: Rolling FIP momentum for each window w


    Notes
    -----

    - For 'original', a positive FIP may indicate inconsistency in the trend.
    - For 'modified', a positive FIP indicates stronger momentum in the direction of the trend (upward or downward).

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Single window with original FIP
    fip_df = (
        df.query("symbol == 'AAPL'")
        .augment_fip_momentum(
            date_column='date',
            close_column='close',
            window=252
        )
    )
    fip_df.tail()
    ```

    ```{python}
    # Multiple windows, polars engine, modified FIP
    pl_df = pl.from_pandas(df)
    fip_df = (
        pl_df.group_by('symbol')
        .tk.augment_fip_momentum(
            date_column='date',
            close_column='close',
            window=[63, 252],
            fip_method='modified',
        )
    )
    fip_df.tail()
    ```
    """

    if isinstance(window, int):
        windows = [window]
    elif isinstance(window, (list, tuple)):
        windows = window
    else:
        raise ValueError("`window` must be an integer or list/tuple of integers")

    if not all(isinstance(w, int) and w > 0 for w in windows):
        raise ValueError("All window values must be positive integers")

    if fip_method not in ["original", "modified"]:
        raise ValueError("`fip_method` must be 'original' or 'modified'")

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

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
        result = _augment_fip_momentum_pandas(
            data=sorted_data,
            close_column=close_column,
            windows=windows,
            fip_method=fip_method,
            skip_window=skip_window,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_fip_momentum_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            fip_method=fip_method,
            skip_window=skip_window,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_fip_momentum_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    close_column: str,
    windows: List[int],
    fip_method: str,
    skip_window: int,
) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
    else:
        raise TypeError(
            "Unsupported data type passed to _augment_fip_momentum_pandas."
        )

    col = close_column
    df[f"{col}_returns"] = df[col].pct_change()

    def calc_fip(ser, window, fip_method):
        returns = ser.dropna()
        if len(returns) < window // 2:
            return np.nan

        total_return = np.prod(1 + returns) - 1
        pct_positive = (returns > 0).mean()
        pct_negative = (returns < 0).mean()
        if fip_method == "original":
            return total_return * (pct_negative - pct_positive)
        elif fip_method == "modified":
            return np.sign(total_return) * (pct_positive - pct_negative)

    if group_names:
        for w in windows:
            out_series = pd.Series(index=df.index, dtype=float)
            # Process each group separately to preserve original index types
            for name, group_df in df.groupby(group_names):
                roll = (
                    group_df[f"{col}_returns"]
                    .rolling(w)
                    .apply(lambda x: calc_fip(x, w, fip_method), raw=False)
                )
                if skip_window > 0:
                    roll.iloc[:skip_window] = np.nan
                out_series.loc[roll.index] = roll
            df[f"{col}_fip_momentum_{w}"] = out_series
    else:
        for w in windows:
            roll = (
                df[f"{col}_returns"]
                .rolling(w)
                .apply(lambda x: calc_fip(x, w, fip_method), raw=False)
            )
            if skip_window > 0:
                roll.iloc[:skip_window] = np.nan
            df[f"{col}_fip_momentum_{w}"] = roll

    df.drop(columns=[f"{col}_returns"], inplace=True)
    return df


def _augment_fip_momentum_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    fip_method: str,
    skip_window: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    def fip_calc(values: np.ndarray, w: int, method: str) -> float:
        valid = ~np.isnan(values)
        if np.sum(valid) < max(1, w // 2):
            return np.nan
        total_return = np.prod(1 + values[valid]) - 1
        pct_positive = np.sum(values[valid] > 0) / np.sum(valid)
        pct_negative = np.sum(values[valid] < 0) / np.sum(valid)
        if method == "original":
            return total_return * (pct_negative - pct_positive)
        return np.sign(total_return) * (pct_positive - pct_negative)

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    def compute(frame: pl.DataFrame) -> pl.DataFrame:
        df = frame.with_columns(
            (pl.col(close_column) / pl.col(close_column).shift(1) - 1).alias(
                "__fip_returns"
            )
        )
        if skip_window > 0:
            df = df.with_row_count("__fip_row")

        for w in windows:
            fip_expr = (
                pl.col("__fip_returns")
                .rolling_map(
                    lambda x: fip_calc(np.array(x), w, fip_method),
                    window_size=w,
                    min_periods=max(1, w // 2),
                )
                .alias(f"{close_column}_fip_momentum_{w}")
            )
            df = df.with_columns(fip_expr)
            if skip_window > 0:
                df = df.with_columns(
                    pl.when(pl.col("__fip_row") < skip_window)
                    .then(None)
                    .otherwise(pl.col(f"{close_column}_fip_momentum_{w}"))
                    .alias(f"{close_column}_fip_momentum_{w}")
                )

        drop_cols = ["__fip_returns"]
        if skip_window > 0:
            drop_cols.append("__fip_row")
        return df.drop(drop_cols)

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

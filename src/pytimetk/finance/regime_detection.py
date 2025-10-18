import pandas as pd
import polars as pl
import numpy as np

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union
from joblib import Parallel, delayed

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

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
def augment_regime_detection(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    close_column: str,
    window: Union[int, Tuple[int, int], List[int]] = 252,
    n_regimes: int = 2,
    method: str = "hmm",
    step_size: int = 1,
    n_iter: int = 100,
    n_jobs: int = -1,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Detect regimes in a financial time series using a specified method (e.g., HMM).

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input time-series data. Grouped inputs are processed per group before
        the regime labels are appended.
    date_column : str
        Column name containing dates or timestamps.
    close_column : str
        Column name with closing prices for regime detection.
    window : Union[int, Tuple[int, int], List[int]], optional
        Size of the rolling window to fit the regime detection model. Default is 252.
    n_regimes : int, optional
        Number of regimes to detect (e.g., 2 for bull/bear). Default is 2.
    method : str, optional
        Method for regime detection. Currently supports 'hmm'. Default is 'hmm'.
    step_size : int, optional
        Step size between HMM fits (e.g., 10 fits every 10 rows). Default is 1.
    n_iter : int, optional
        Number of iterations for HMM fitting. Default is 100.
    n_jobs : int, optional
        Number of parallel jobs for group processing (-1 uses all cores). Default is -1.
    reduce_memory : bool, optional
        If True, reduces memory usage. Default is False.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the
        input data while allowing explicit overrides.

    Returns
    -------
    DataFrame
        DataFrame with added columns:
        - {close_column}_regime_{window}: Integer labels for detected regimes (e.g., 0, 1).

    Notes
    -----
    - Uses Hidden Markov Model (HMM) to identify latent regimes based on log returns.
    - Regimes reflect distinct statistical states (e.g., high/low volatility, trending).
    - Requires 'hmmlearn' package. Install with `pip install hmmlearn`.

    Examples
    --------
    ```python
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Example 1 - Single stock regime detection with pandas engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.query("symbol == 'AAPL'")
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=252,
            n_regimes=2
        )
    )
    regime_df.head().glimpse()
    ```

    ```python
    # Example 2 - Multiple stocks with groupby using pandas engine
    # Requires hmmlearn: pip install hmmlearn
    regime_df = (
        df.groupby('symbol')
        .augment_regime_detection(
            date_column='date',
            close_column='close',
            window=[252, 504],  # One year and two years
            n_regimes=3
        )
    )
    regime_df.groupby('symbol').tail(1).glimpse()
    ```

    ```python
    # Example 3 - Single stock regime detection with polars engine
    # Requires hmmlearn: pip install hmmlearn
    pl_single = pl.from_pandas(df.query("symbol == 'AAPL'"))
    regime_df = pl_single.tk.augment_regime_detection(
        date_column='date',
        close_column='close',
        window=252,
        n_regimes=2
    )
    regime_df.glimpse()
    ```

    ```python
    # Example 4 - Multiple stocks with groupby using polars engine
    # Requires hmmlearn: pip install hmmlearn
    pl_df = pl.from_pandas(df)
    regime_df = (
        pl_df.group_by('symbol')
        .tk.augment_regime_detection(
            date_column='date',
            close_column='close',
            window=504,
            n_regimes=3,
        )
    )
    regime_df.groupby('symbol').tail(1).glimpse()
    ```
    """

    # Check for hmmlearn availability
    if method.lower() == "hmm" and GaussianHMM is None:
        raise ImportError(
            "The 'hmm' method requires the 'hmmlearn' package, which is not installed. "
            "Please install it using: `pip install hmmlearn`"
        )

    if method.lower() == "hmm" and GaussianHMM is None:
        raise ImportError(
            "The 'hmm' method requires the 'hmmlearn' package, which is not installed. "
            "Please install it using: `pip install hmmlearn`"
        )

    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)

    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
    if method.lower() != "hmm":
        raise ValueError("Only 'hmm' method is currently supported.")
    if step_size < 1:
        raise ValueError("step_size must be at least 1.")

    windows = _normalize_windows(window)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "cudf":
        warnings.warn(
            "augment_regime_detection does not yet offer a native cudf implementation. Falling back to pandas.",
            RuntimeWarning,
            stacklevel=2,
        )
        conversion_engine = "pandas"
    else:
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
        result = _augment_regime_detection_pandas(
            data=sorted_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            n_regimes=n_regimes,
            step_size=step_size,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    else:
        result = _augment_regime_detection_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            n_regimes=n_regimes,
            step_size=step_size,
            n_iter=n_iter,
            n_jobs=n_jobs,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_regime_detection_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    n_regimes: int,
    step_size: int,
    n_iter: int,
    n_jobs: int,
) -> pd.DataFrame:
    """Pandas implementation of regime detection using HMM."""

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy()

    col = close_column

    df["log_returns"] = np.log(df[col] / df[col].shift(1))
    df["log_returns"] = df["log_returns"].replace([np.inf, -np.inf], np.nan)

    def detect_regimes(series, window, n_regimes, step_size, n_iter):
        n = len(series)
        regimes = np.full(n, np.nan, dtype=float)
        for i in range(window - 1, n, step_size):
            window_data = series[max(0, i - window + 1) : i + 1].dropna()
            if len(window_data) < max(window // 2, n_regimes * 10):
                continue
            window_data = window_data.values.reshape(-1, 1)
            if not np.all(np.isfinite(window_data)):
                continue
            try:
                model = GaussianHMM(
                    n_components=n_regimes, covariance_type="diag", n_iter=n_iter
                )
                model.fit(window_data)
                predicted = model.predict(window_data)
                # Fill regimes from i-step_size+1 to i with the last predicted value
                start_idx = max(0, i - step_size + 1)
                regimes[start_idx : i + 1] = predicted[
                    -step_size if i > window - 1 else -start_idx :
                ]
            except ValueError as e:
                print(f"Warning: HMM fit failed at index {i} with error: {e}")
                continue
        return pd.Series(regimes, index=series.index)

    for window in windows:
        if group_names:
            # Parallelize across groups
            results = Parallel(n_jobs=n_jobs)(
                delayed(detect_regimes)(
                    group["log_returns"], window, n_regimes, step_size, n_iter
                )
                for _, group in df.groupby(group_names)
            )
            df[f"{col}_regime_{window}"] = pd.concat(results).reindex(df.index)
        else:
            df[f"{col}_regime_{window}"] = detect_regimes(
                df["log_returns"], window, n_regimes, step_size, n_iter
            )

    df = df.drop(columns=["log_returns"])
    return df


def _augment_regime_detection_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    n_regimes: int,
    step_size: int,
    n_iter: int,
    n_jobs: int,
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    """Polars implementation of regime detection using HMM (via pandas)."""

    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    pandas_df = sorted_frame.to_pandas()

    if resolved_groups:
        pandas_groupby = pandas_df.groupby(resolved_groups, sort=False)
        result_pd = _augment_regime_detection_pandas(
            data=pandas_groupby,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            n_regimes=n_regimes,
            step_size=step_size,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )
    else:
        result_pd = _augment_regime_detection_pandas(
            data=pandas_df,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            n_regimes=n_regimes,
            step_size=step_size,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )

    result = pl.from_pandas(result_pd).sort(row_col)

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

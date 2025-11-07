import pandas as pd
import polars as pl
import numpy as np

import pandas_flavor as pf
import warnings
from typing import List, Optional, Sequence, Tuple, Union
from joblib import Parallel, delayed

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - optional dependency
    GaussianHMM = None

import importlib
import importlib.util

_POMEGRANATE_MODEL = None
_POMEGRANATE_DIST = None


def _ensure_pomegranate_available():
    """
    Lazily import pomegranate regardless of major version.
    Returns (HiddenMarkovModel, NormalDistribution) classes.
    """
    global _POMEGRANATE_MODEL, _POMEGRANATE_DIST
    if _POMEGRANATE_MODEL is not None and _POMEGRANATE_DIST is not None:
        return _POMEGRANATE_MODEL, _POMEGRANATE_DIST

    last_error = None

    try:
        from pomegranate import HiddenMarkovModel as legacy_hmm
        from pomegranate import NormalDistribution as legacy_norm

        _POMEGRANATE_MODEL = legacy_hmm
        _POMEGRANATE_DIST = legacy_norm
    except ImportError as exc:
        last_error = exc

    if _POMEGRANATE_MODEL is None or _POMEGRANATE_DIST is None:
        message = (
            "The 'pomegranate' backend requires the legacy pomegranate>=0.14,<1.0 "
            "release which exposes HiddenMarkovModel/NormalDistribution. "
            "Install it with `pip install 'pomegranate<1.0'`."
        )
        if last_error is not None:
            message += f" Original error: {last_error}"
        raise ImportError(message)

    return _POMEGRANATE_MODEL, _POMEGRANATE_DIST


try:  # Optional cudf dependency
    import cudf  # type: ignore
except ImportError:  # pragma: no cover - cudf optional
    cudf = None  # type: ignore

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
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
from pytimetk.utils.selection import ColumnSelector
from pytimetk.feature_engineering._shift_utils import resolve_shift_columns

HMMLEARN_AVAILABLE = GaussianHMM is not None
POMEGRANATE_AVAILABLE = importlib.util.find_spec("pomegranate") is not None


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_regime_detection(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: Union[str, ColumnSelector],
    close_column: Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]]],
    window: Union[int, Tuple[int, int], List[int]] = 252,
    n_regimes: int = 2,
    method: str = "hmm",
    step_size: int = 1,
    n_iter: int = 100,
    n_jobs: int = -1,
    reduce_memory: bool = False,
    hmm_backend: str = "auto",
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Detect regimes in a financial time series using a specified method (e.g., HMM).

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Input time-series data. Grouped inputs are processed per group before
        the regime labels are appended.
    date_column : str or ColumnSelector
        Column (or selector) containing dates or timestamps.
    close_column : str, ColumnSelector, or list
        Column(s) with closing prices used for regime detection. Must resolve to
        exactly one column.
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
    hmm_backend : {"auto", "pomegranate", "hmmlearn"}, optional
        Backend library used for the HMM implementation. ``"auto"`` (default)
        prefers the faster ``pomegranate`` backend when installed, otherwise
        falls back to ``hmmlearn``.
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
    - Requires 'hmmlearn' package. Install with `pip install hmmlearn` or the faster optional `pomegranate` backend via `pip install 'pytimetk[regime_backends]'` (equivalent to `pip install 'pomegranate<1.0'`).

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk

    df = tk.load_dataset("stocks_daily", parse_dates=["date"])

    df
    ```

    ```{python}
    # Regime detection - pandas single stock (requires hmm backend)
    regime_single = (
        df
        .query("symbol == 'AAPL'")
        .augment_regime_detection(
            date_column="date",
            close_column="close",
            window=252,
            n_regimes=2,
        )
    )

    regime_single.glimpse()
    ```

    ```{python}
    # Regime detection - pandas grouped (requires hmm backend)
    regime_grouped = (
        df
        .groupby("symbol")
        .augment_regime_detection(
            date_column="date",
            close_column="close",
            window=[252, 504],
            n_regimes=3,
        )
    )

    regime_grouped.groupby("symbol").tail(1)
    ```

    ```{python}
    # Regime detection - polars engine (requires hmm backend)
    pl_single = pl.from_pandas(df.query("symbol == 'AAPL'"))
    regime_polars = (
        pl_single
        .tk.augment_regime_detection(
            date_column="date",
            close_column="close",
            window=252,
            n_regimes=2,
        )
    )

    regime_polars.glimpse()
    ```

    ```{python}
    # Pomegranate backend with column selectors
    from pytimetk.utils.selection import contains

    selector_demo = (
        df
        .groupby("symbol")
        .augment_regime_detection(
            date_column=contains("dat"),
            close_column=contains("clos"),
            window=252,
            n_regimes=4,
            hmm_backend="pomegranate", # pomegranate<=1.0.0 required
        )
    )

    selector_demo.groupby("symbol").tail(1)
    ```

    ``` {python}
    # Visualizing regimes
    SYMBOLS = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'NVDA']
    SYMBOL = 'NVDA'

    (
        selector_demo
        .query(f"symbol == '{SYMBOL}'")
        .plot_timeseries(
            date_column="date",
            value_column="close",
            color_column=contains("regime_"),
            smooth=False,
            title=f"{SYMBOL} Close Price with Detected Regimes",
        )
    )
    ```
    """

    method_lc = method.lower()
    if method_lc != "hmm":
        raise ValueError("Only 'hmm' method is currently supported.")

    backend = hmm_backend.lower()
    if backend not in {"auto", "pomegranate", "hmmlearn"}:
        raise ValueError(
            "Invalid `hmm_backend`. Choose from {'auto', 'pomegranate', 'hmmlearn'}."
        )
    if backend == "auto":
        if HMMLEARN_AVAILABLE:
            backend = "hmmlearn"
        elif POMEGRANATE_AVAILABLE:
            backend = "pomegranate"
        else:
            backend = "hmmlearn"

    if backend == "pomegranate" and not POMEGRANATE_AVAILABLE:
        raise ImportError(
            "The 'pomegranate' backend requires the 'pomegranate' package. "
            "Install it with `pip install pomegranate`."
        )
    if backend == "hmmlearn" and not HMMLEARN_AVAILABLE:
        raise ImportError(
            "The 'hmmlearn' backend requires the 'hmmlearn' package. "
            "Install it with `pip install hmmlearn`."
        )

    check_dataframe_or_groupby(data)
    date_column, close_columns = resolve_shift_columns(
        data,
        date_column=date_column,
        value_column=close_column,
        require_numeric=True,
    )
    if len(close_columns) != 1:
        raise ValueError("`close_column` selector must resolve to exactly one column.")
    close_column = close_columns[0]

    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2.")
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
            hmm_backend=backend,
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
            hmm_backend=backend,
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
    hmm_backend: str,
    n_jobs: int,
) -> pd.DataFrame:
    """Pandas implementation of regime detection using HMM."""

    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=False)
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy(deep=False)

    col = close_column

    if group_names:
        prev = df.groupby(group_names)[col].shift(1)
    else:
        prev = df[col].shift(1)
    df["log_returns"] = np.log(df[col] / prev)
    df["log_returns"] = df["log_returns"].replace([np.inf, -np.inf], np.nan)

    pome_model_cls = None
    pome_dist_cls = None
    if hmm_backend == "pomegranate":
        pome_model_cls, pome_dist_cls = _ensure_pomegranate_available()

    def detect_regimes(series, window, n_regimes, step_size, n_iter):
        values = series.to_numpy(dtype=float, copy=False)
        n = len(values)
        regimes = np.full(n, np.nan, dtype=float)
        min_obs = max(window // 2, n_regimes * 10)
        hmm_model = None
        hmm_params = None
        pom_model = None
        for i in range(window - 1, n, step_size):
            start = max(0, i - window + 1)
            window_values = values[start : i + 1]
            finite_idx = np.where(np.isfinite(window_values))[0]
            if len(finite_idx) < min_obs:
                continue
            window_data = window_values[finite_idx].reshape(-1, 1)
            try:
                if hmm_backend == "hmmlearn":
                    if hmm_model is None:
                        hmm_model = GaussianHMM(
                            n_components=n_regimes,
                            covariance_type="diag",
                            n_iter=n_iter,
                            tol=1e-3,
                        )
                    if hmm_params is not None:
                        hmm_model.startprob_ = hmm_params["startprob"]
                        hmm_model.transmat_ = hmm_params["transmat"]
                        hmm_model.means_ = hmm_params["means"]
                        hmm_model.covars_ = hmm_params["covars"]
                        hmm_model.init_params = ""
                    else:
                        hmm_model.init_params = "stmc"
                    hmm_model.fit(window_data)
                    predicted = hmm_model.predict(window_data)
                    hmm_params = {
                        "startprob": hmm_model.startprob_.copy(),
                        "transmat": hmm_model.transmat_.copy(),
                        "means": hmm_model.means_.copy(),
                        "covars": hmm_model.covars_.copy(),
                    }
                else:
                    sequence = window_data.ravel().tolist()
                    if pom_model is None:
                        pom_model = pome_model_cls.from_samples(
                            pome_dist_cls,
                            n_components=n_regimes,
                            X=[sequence],
                            algorithm="baum-welch",
                            max_iterations=n_iter,
                            stop_threshold=1e-3,
                        )
                    else:
                        pom_model.fit(
                            [sequence],
                            algorithm="baum-welch",
                            max_iterations=n_iter,
                            stop_threshold=1e-3,
                        )
                    predicted = np.asarray(pom_model.predict(sequence))
            except ValueError:
                continue
            tail_len = min(step_size, len(finite_idx))
            target_positions = finite_idx[-tail_len:] + start
            regimes[target_positions] = predicted[-tail_len:]
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
    hmm_backend: str,
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
            hmm_backend=hmm_backend,
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
            hmm_backend=hmm_backend,
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

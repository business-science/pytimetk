from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf

from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import ccf as sm_ccf
from statsmodels.tsa.stattools import pacf as sm_pacf

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
)
from pytimetk.utils.datetime_helpers import resolve_lag_sequence
from pytimetk.utils.selection import resolve_column_selection, ColumnSelector
from pytimetk.feature_engineering._shift_utils import resolve_shift_columns
from pytimetk.utils.dataframe_ops import resolve_pandas_groupby_frame


@dataclass
class _ACFConfig:
    date_column: str
    value_column: str
    ccf_columns: List[str]
    lags: Union[str, int, Sequence[int], np.ndarray, range, slice]


def _prepare_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _acf_diagnostics_single(frame: pd.DataFrame, config: _ACFConfig) -> pd.DataFrame:
    sorted_frame = frame.sort_values(config.date_column).reset_index(drop=True)
    lags_array = resolve_lag_sequence(
        config.lags,
        sorted_frame[config.date_column],
    )
    value_series = _prepare_numeric(sorted_frame[config.value_column])
    value_clean = value_series.dropna().to_numpy()

    if value_clean.size < 2:
        raise ValueError(
            "Not enough non-missing observations to compute autocorrelation diagnostics."
        )

    max_lag_from_data = min(int(lags_array.max()), value_clean.size - 1)
    pacf_capacity = max(value_clean.size // 2 - 1, 0)
    max_lag = min(
        max_lag_from_data, pacf_capacity if pacf_capacity > 0 else max_lag_from_data
    )

    lags_for_core = lags_array[lags_array <= max_lag]
    if lags_for_core.size == 0:
        lags_for_core = np.array([0], dtype=int)

    acf_values = sm_acf(value_clean, nlags=int(max_lag), fft=True)
    pacf_values = sm_pacf(value_clean, nlags=int(max_lag), method="ywmle")

    sample_size = float(value_clean.size)
    white_noise = 2.0 / np.sqrt(sample_size)

    records: List[dict] = []
    for lag in lags_for_core:
        records.append(
            {
                "lag": int(lag),
                "metric": "ACF",
                "value": float(acf_values[lag]),
                "white_noise_upper": white_noise,
                "white_noise_lower": -white_noise,
            }
        )
        records.append(
            {
                "lag": int(lag),
                "metric": "PACF",
                "value": float(pacf_values[lag]),
                "white_noise_upper": white_noise,
                "white_noise_lower": -white_noise,
            }
        )

    for column in config.ccf_columns:
        paired = sorted_frame[[config.value_column, column]].dropna()
        if paired.shape[0] < 2:
            continue
        x = _prepare_numeric(paired[config.value_column]).to_numpy()
        y = _prepare_numeric(paired[column]).to_numpy()

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]
        if x.size < 2:
            continue

        max_lag_ccf = min(max_lag, x.size - 1, max(x.size // 2 - 1, 0))
        lags_for_column = lags_for_core[lags_for_core <= max_lag_ccf]
        if lags_for_column.size == 0:
            continue

        ccf_values = sm_ccf(x, y, adjusted=False)
        for lag in lags_for_column:
            records.append(
                {
                    "lag": int(lag),
                    "metric": f"CCF_{column}",
                    "value": float(ccf_values[lag]),
                    "white_noise_upper": white_noise,
                    "white_noise_lower": -white_noise,
                }
            )

    result = pd.DataFrame.from_records(records)
    result.sort_values(["metric", "lag"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


@pf.register_groupby_method
@pf.register_dataframe_method
def acf_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector],
    ccf_columns: Optional[Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]], np.ndarray]] = None,
    lags: Union[str, int, Sequence[int], np.ndarray, range, slice] = 1000,
) -> pd.DataFrame:
    """
    Compute tidy autocorrelation, partial autocorrelation, and optional
    cross-correlation diagnostics for one or more time series.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Long-form time series data (optionally grouped via ``groupby``).
    date_column : str
        Name of the datetime column.
    value_column : str
        Numeric column used to compute ACF/PACF diagnostics.
    ccf_columns : str or sequence, optional
        Additional numeric columns to run cross-correlation against
        ``value_column``. Accepts literal column names or tidy selectors created
        with :mod:`pytimetk.utils.selection` (e.g. ``contains("driver")``).
    lags : int, sequence, slice, or str, optional
        Lag specification. Integers mirror ``range(0, lags)``,
        sequences/slices are used verbatim, and strings such as ``"30 days"`` or
        ``"3 months"`` are resolved relative to the supplied index. Defaults to
        ``1000``.

    Returns
    -------
    pd.DataFrame
        Diagnostics with columns:

        - grouping columns (when present)
        - ``metric`` (``"ACF"``, ``"PACF"``, or ``"CCF_<column>"``)
        - ``lag`` (non-negative integer)
        - ``value`` (correlation)
        - ``white_noise_upper`` / ``white_noise_lower`` (95% bounds)

    Examples
    --------
    ```{python}
    import numpy as np
    import pandas as pd
    import pytimetk as tk

    rng = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "id": ["A"] * 20 + ["B"] * 20,
            "date": list(rng[:20]) + list(rng[:20]),
            "value": np.sin(np.linspace(0, 4 * np.pi, 40)),
            "driver": np.cos(np.linspace(0, 4 * np.pi, 40)),
        }
    )

    diagnostics = tk.acf_diagnostics(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        ccf_columns="driver",
        lags="30 days",
    )
    diagnostics.head()
    ```
    """
    check_dataframe_or_groupby(data)
    date_column, value_columns = resolve_shift_columns(
        data,
        date_column=date_column,
        value_column=value_column,
        require_numeric=True,
    )
    value_column = value_columns[0]

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        base_frame = resolve_pandas_groupby_frame(data)
    else:
        base_frame = data

    if ccf_columns is not None:
        resolved_ccf = resolve_column_selection(
            base_frame,
            ccf_columns,
            allow_none=False,
            require_match=True,
            unique=True,
        )
    else:
        resolved_ccf = []

    config = _ACFConfig(
        date_column=date_column,
        value_column=value_column,
        ccf_columns=resolved_ccf,
        lags=lags,
    )

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        results: List[pd.DataFrame] = []
        for keys, frame in data:
            diag = _acf_diagnostics_single(frame.copy(), config)
            if not isinstance(keys, tuple):
                keys = (keys,)
            for name, value in zip(group_names, keys):
                diag[name] = value
            results.append(diag)
        combined = pd.concat(results, ignore_index=True)
        return combined[
            [
                *group_names,
                "metric",
                "lag",
                "value",
                "white_noise_upper",
                "white_noise_lower",
            ]
        ]

    diagnostics = _acf_diagnostics_single(base_frame.copy(), config)
    return diagnostics[
        ["metric", "lag", "value", "white_noise_upper", "white_noise_lower"]
    ]

from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pandas_flavor as pf
from statsmodels.tsa.seasonal import STL

from pytimetk.core.frequency import get_seasonal_frequency, get_trend_frequency
from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.datetime_helpers import parse_human_duration


def _median_interval(index: pd.Series) -> pd.Timedelta:
    idx = pd.to_datetime(index).dropna().sort_values()
    if idx.size < 2:
        raise ValueError("At least two timestamps are required for STL diagnostics.")
    diffs = pd.Series(idx).diff().dropna()
    median = diffs.median()
    if pd.isna(median) or median.total_seconds() == 0:
        raise ValueError("Unable to determine sampling interval for the time index.")
    return median


def _duration_to_period(
    duration: Union[pd.Timedelta, pd.DateOffset], index: pd.Series
) -> int:
    median_delta = _median_interval(index)
    median_seconds = median_delta.total_seconds()

    if isinstance(duration, pd.DateOffset):
        base = pd.Timestamp(pd.to_datetime(index).dropna().sort_values().iloc[0])
        delta = (base + duration) - base
    else:
        delta = duration

    if not isinstance(delta, pd.Timedelta):
        delta = pd.to_timedelta(delta)

    total_seconds = delta.total_seconds()
    if total_seconds <= 0:
        raise ValueError("Duration for STL parameters must be positive.")

    period = max(int(round(total_seconds / median_seconds)), 2)
    return period


def _resolve_period(
    value: Union[str, int, float, pd.Timedelta, pd.DateOffset, None],
    index: pd.Series,
    auto_callable,
) -> int:
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        inferred = auto_callable(index, numeric=True)
        try:
            return max(int(math.ceil(float(inferred))), 2)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Failed to auto-resolve STL period.") from exc

    if isinstance(value, (int, float, np.integer, np.floating)):
        return max(int(round(float(value))), 2)

    if isinstance(value, pd.DateOffset):
        return _duration_to_period(value, index)

    if isinstance(value, (pd.Timedelta, np.timedelta64)):
        return _duration_to_period(pd.to_timedelta(value), index)

    if isinstance(value, str):
        try:
            offset = pd.tseries.frequencies.to_offset(value)
            if offset.delta is not None:
                return _duration_to_period(offset.delta, index)
            return _duration_to_period(offset, index)
        except (ValueError, TypeError):
            parsed = parse_human_duration(value)
            return _duration_to_period(parsed, index)

    raise TypeError(
        "Unsupported STL period specification. Provide 'auto', a numeric value, "
        "a pandas offset string (e.g., '7D'), or a human-readable duration."
    )


def _stl_diagnostics_single(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: Union[str, int, float, pd.Timedelta, pd.DateOffset, None],
    trend: Union[str, int, float, pd.Timedelta, pd.DateOffset, None],
    robust: bool,
) -> pd.DataFrame:
    sorted_frame = frame.sort_values(date_column).reset_index(drop=True)
    value_series = pd.to_numeric(sorted_frame[value_column], errors="coerce")
    if value_series.isna().all():
        raise ValueError(
            "Value column contains only missing values; STL cannot be computed."
        )

    filled_values = value_series.interpolate(limit_direction="both")
    if filled_values.isna().any():
        # As a fallback, fill remaining NaNs with forward/backward fill
        filled_values = filled_values.fillna(method="ffill").fillna(method="bfill")

    period = _resolve_period(
        frequency, sorted_frame[date_column], get_seasonal_frequency
    )
    period = max(2, min(period, len(filled_values)))

    trend_window = _resolve_period(
        trend, sorted_frame[date_column], get_trend_frequency
    )
    trend_window = max(3, trend_window)
    if trend_window % 2 == 0:
        trend_window += 1
    trend_window = min(trend_window, len(filled_values) - (1 - len(filled_values) % 2))
    trend_window = max(trend_window, 3)

    seasonal_window = max(period, 7)
    if seasonal_window % 2 == 0:
        seasonal_window += 1

    stl = STL(
        filled_values,
        period=period,
        seasonal=seasonal_window,
        trend=trend_window,
        robust=robust,
    )
    result = stl.fit()

    decomposition = pd.DataFrame(
        {
            date_column: sorted_frame[date_column],
            "observed": value_series,
            "season": result.seasonal,
            "trend": result.trend,
            "remainder": result.resid,
        }
    )
    decomposition["seasadj"] = decomposition["observed"] - decomposition["season"]
    return decomposition


@pf.register_groupby_method
@pf.register_dataframe_method
def stl_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: str,
    frequency: Union[str, int, float, pd.Timedelta, pd.DateOffset, None] = "auto",
    trend: Union[str, int, float, pd.Timedelta, pd.DateOffset, None] = "auto",
    robust: bool = True,
) -> pd.DataFrame:
    """
    Generate STL decomposition diagnostics (observed, season, trend, remainder, seasadj).

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Time series data, optionally grouped.
    date_column : str
        Name of the datetime column.
    value_column : str
        Numeric measure to decompose.
    frequency : str, int, float, pd.Timedelta, pd.DateOffset, optional
        Seasonal period specification. ``"auto"`` (default) infers a period via
        :func:`pytimetk.get_seasonal_frequency`. Strings such as ``"7D"`` or
        ``"30 days"`` are supported.
    trend : str, int, float, pd.Timedelta, pd.DateOffset, optional
        Trend window specification. ``"auto"`` (default) infers a window via
        :func:`pytimetk.get_trend_frequency`.
    robust : bool, optional
        Apply a robust STL fit (down-weights outliers). Defaults to ``True``.

    Returns
    -------
    pd.DataFrame
        Decomposition with columns:

        - grouping columns (if present)
        - ``date``, ``observed``, ``season``, ``trend``, ``remainder``, ``seasadj``

    Examples
    --------
    ```{python}
    import numpy as np
    import pandas as pd
    import pytimetk as tk

    rng = pd.date_range("2020-01-01", periods=180, freq="D")
    values = np.sin(np.linspace(0, 8 * np.pi, len(rng))) + np.random.default_rng(123).normal(scale=0.1, size=len(rng))
    df = pd.DataFrame({"date": rng, "value": values})

    decomposition = tk.stl_diagnostics(
        data=df,
        date_column="date",
        value_column="value",
        frequency="7D",
        trend="30 days",
    )
    decomposition.head()
    ```
    """
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column, require_numeric_dtype=True)

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        results = []
        for keys, group_df in data:
            diag = _stl_diagnostics_single(
                group_df,
                date_column=date_column,
                value_column=value_column,
                frequency=frequency,
                trend=trend,
                robust=robust,
            )
            if not isinstance(keys, tuple):
                keys = (keys,)
            for name, value in zip(group_names, keys):
                diag[name] = value
            results.append(diag)

        combined = pd.concat(results, ignore_index=True)
        return combined[
            [
                *group_names,
                date_column,
                "observed",
                "season",
                "trend",
                "remainder",
                "seasadj",
            ]
        ]

    decomposition = _stl_diagnostics_single(
        data,
        date_column=date_column,
        value_column=value_column,
        frequency=frequency,
        trend=trend,
        robust=robust,
    )
    return decomposition[
        [date_column, "observed", "season", "trend", "remainder", "seasadj"]
    ]

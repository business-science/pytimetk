from __future__ import annotations

import pandas as pd
import pytimetk.utils.pandas_flavor_compat as pf

from typing import List, Sequence, Union

from pytimetk.feature_engineering.timeseries_signature import (
    augment_timeseries_signature,
)
from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
)
from pytimetk.utils.selection import ColumnSelector
from pytimetk.feature_engineering._shift_utils import resolve_shift_columns

_SEASONAL_FEATURE_MAP = {
    "second": "second",
    "minute": "minute",
    "hour": "hour",
    "wday.lbl": "wday_lbl",
    "wday_lbl": "wday_lbl",
    "week": "yweek",
    "month.lbl": "month_lbl",
    "month_lbl": "month_lbl",
    "quarter": "quarter",
    "year": "year",
}

_FEATURE_PERIOD_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "wday.lbl": 86400,
    "week": 604800,
    "month.lbl": 2629800,  # ~30.44 days
    "quarter": 7889400,  # 3 months
    "year": 31557600,  # 365.25 days
}

_AUTO_FEATURE_BANDS = [
    (60, ["second", "minute", "hour", "wday.lbl", "week", "month.lbl"]),
    (3600, ["minute", "hour", "wday.lbl", "week", "month.lbl"]),
    (86400, ["hour", "wday.lbl", "week", "month.lbl", "quarter", "year"]),
    (604800, ["wday.lbl", "week", "month.lbl", "quarter", "year"]),
    (2678400, ["week", "month.lbl", "quarter", "year"]),
]


def _normalise_feature_name(feature: str) -> str:
    key = feature.strip().lower().replace(" ", "")
    replacements = {
        "wdaylbl": "wday.lbl",
        "weekday": "wday.lbl",
        "weekdaylbl": "wday.lbl",
        "monthlbl": "month.lbl",
    }
    return replacements.get(key, key)


def _auto_seasonal_features(index: pd.Series) -> List[str]:
    idx = pd.to_datetime(index).dropna().sort_values()
    if idx.size < 3:
        return ["year"]

    diffs = pd.Series(idx).diff().dropna()
    if diffs.empty:
        return ["year"]

    median_delta = diffs.median()
    median_seconds = median_delta.total_seconds()
    total_span_seconds = (idx.iloc[-1] - idx.iloc[0]).total_seconds()

    features: List[str] = []
    for threshold, choices in _AUTO_FEATURE_BANDS:
        if median_seconds < threshold:
            features = choices
            break
    if not features:
        features = ["month.lbl", "quarter", "year"]

    def has_two_periods(name: str) -> bool:
        period_seconds = _FEATURE_PERIOD_SECONDS.get(name, None)
        if period_seconds is None:
            return True
        return total_span_seconds >= 2 * period_seconds

    filtered = [name for name in features if has_two_periods(name)]
    return filtered or ["year"]


def _resolve_feature_set(
    feature_set: Union[str, Sequence[str], None],
    index: pd.Series,
) -> List[str]:
    if feature_set is None:
        feature_set = ["auto"]
    if isinstance(feature_set, str):
        feature_list = [feature_set]
    else:
        feature_list = list(feature_set)

    feature_list = [_normalise_feature_name(name) for name in feature_list]

    if "auto" in feature_list:
        base = _auto_seasonal_features(index)
        extras = [name for name in feature_list if name != "auto"]
        combined = base + extras
    else:
        combined = feature_list

    resolved: List[str] = []
    for name in combined:
        if name not in _SEASONAL_FEATURE_MAP:
            raise ValueError(
                f"Unknown seasonal feature '{name}'. "
                f"Supported values: {sorted(_SEASONAL_FEATURE_MAP.keys()) + ['auto']}"
            )
        if name not in resolved:
            resolved.append(name)
    return resolved


def _seasonal_diagnostics_single(
    frame: pd.DataFrame,
    date_column: str,
    value_column: str,
    feature_set: List[str],
) -> pd.DataFrame:
    sorted_frame = frame.sort_values(date_column).reset_index(drop=True)
    augmented = augment_timeseries_signature(
        sorted_frame[[date_column, value_column]].copy(),
        date_column=date_column,
    )

    column_map = {
        feature: f"{date_column}_{_SEASONAL_FEATURE_MAP[feature]}"
        for feature in feature_set
    }

    missing = [col for col in column_map.values() if col not in augmented.columns]
    if missing:
        raise ValueError(
            "The following seasonal feature columns could not be generated: "
            f"{missing}. Ensure the timestamp granularity supports these features."
        )

    tidy = augmented[[date_column, value_column, *column_map.values()]].melt(
        id_vars=[date_column, value_column],
        value_vars=list(column_map.values()),
        var_name="_feature",
        value_name="seasonal_value",
    )

    reverse_map = {v: k for k, v in column_map.items()}
    tidy["seasonal_feature"] = tidy["_feature"].map(reverse_map)
    tidy.drop(columns="_feature", inplace=True)

    return tidy[[date_column, value_column, "seasonal_feature", "seasonal_value"]]


@pf.register_groupby_method
@pf.register_dataframe_method
def seasonal_diagnostics(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: Union[str, ColumnSelector],
    value_column: Union[str, ColumnSelector],
    feature_set: Union[str, Sequence[str], None] = "auto",
) -> pd.DataFrame:
    """
    Prepare seasonal feature diagnostics akin to ``tk_seasonal_diagnostics``.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Time series data (long format) or grouped data.
    date_column : str
        Name of the datetime column.
    value_column : str
        Numeric measure to analyse.
    feature_set : str or sequence, optional
        One or more of ``["second", "minute", "hour", "wday.lbl", "week",
        "month.lbl", "quarter", "year"]``. The special value ``"auto"``
        selects features based on the timestamp scale and overall history.

    Returns
    -------
    pd.DataFrame
        Tidy data with:

        - grouping columns (when present)
        - ``date`` (or the supplied ``date_column``)
        - the original ``value_column``
        - ``seasonal_feature`` (e.g. ``"hour"``)
        - ``seasonal_value`` (the actual categorical value for that observation)

    Examples
    --------
    ```{python}
    import numpy as np
    import pandas as pd
    import pytimetk as tk

    rng = pd.date_range("2020-01-01", periods=48, freq="H")
    df = pd.DataFrame(
        {
            "id": ["A"] * 24 + ["B"] * 24,
            "date": list(rng[:24]) + list(rng[:24]),
            "value": np.random.default_rng(123).normal(size=48),
        }
    )

    diagnostics = tk.seasonal_diagnostics(
        data=df.groupby("id"),
        date_column="date",
        value_column="value",
        feature_set=["hour", "wday.lbl"],
    )
    diagnostics.head()
    ```

    ```{python}
    from pytimetk.utils.selection import contains

    selector_diagnostics = tk.seasonal_diagnostics(
        data=df,
        date_column=contains("dat"),
        value_column=contains("val"),
        feature_set=["hour"],
    )
    selector_diagnostics.head()
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
        group_names = data.grouper.names

        feature_union: List[str] = []
        for _, group_df in data:
            union_features = _resolve_feature_set(feature_set, group_df[date_column])
            for feature in union_features:
                if feature not in feature_union:
                    feature_union.append(feature)

        results: List[pd.DataFrame] = []
        for keys, group_df in data:
            diag = _seasonal_diagnostics_single(
                group_df,
                date_column=date_column,
                value_column=value_column,
                feature_set=feature_union,
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
                value_column,
                "seasonal_feature",
                "seasonal_value",
            ]
        ]

    feature_list = _resolve_feature_set(feature_set, data[date_column])
    diagnostics = _seasonal_diagnostics_single(
        data,
        date_column=date_column,
        value_column=value_column,
        feature_set=feature_list,
    )
    return diagnostics

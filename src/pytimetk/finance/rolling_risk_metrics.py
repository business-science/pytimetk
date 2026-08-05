import pandas as pd
import polars as pl
import pandas_flavor as pf
import numpy as np
import warnings
from numbers import Integral
from typing import List, Optional, Sequence, Union

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
from scipy import stats  # For skewness and kurtosis


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_rolling_risk_metrics(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
        "cudf.DataFrame",
        "cudf.core.groupby.groupby.DataFrameGroupBy",
    ],
    date_column: Union[str, ColumnSelector],
    close_column: Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]]],
    window: Union[int, List[int]] = 252,
    risk_free_rate: float = 0.0,
    benchmark_column: Optional[
        Union[str, ColumnSelector, Sequence[Union[str, ColumnSelector]]]
    ] = None,
    annualization_factor: int = 252,
    metrics: Optional[List[str]] = None,
    reduce_memory: bool = False,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """The augment_rolling_risk_metrics function calculates rolling risk-adjusted performance
    metrics for a financial time series using either pandas or polars engine, and returns
    the augmented DataFrame with columns for Sharpe Ratio, Sortino Ratio, and other metrics.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input data can be a pandas DataFrame or a pandas DataFrameGroupBy object
        containing the time series data for risk metric calculations.
    date_column : str or ColumnSelector
        The name or selector of the column containing dates or timestamps.
    close_column : str, ColumnSelector, or list
        The column(s) containing closing prices to calculate returns and risk
        metrics from. Must resolve to exactly one column.
    window : int, optional
        The rolling window size for calculations (e.g., 252 for annual). Default is 252.
    risk_free_rate : float, optional
        The assumed risk-free rate (e.g., 0.0 for 0%). Default is 0.0.
    benchmark_column : str, ColumnSelector, or None, optional
        Column containing benchmark returns (e.g., market index) for Treynor
        and Information Ratios. If provided it must resolve to one column.
        Default is None.
    annualization_factor : int, optional
        The factor to annualize returns and volatility (e.g., 252 for daily data). Default is 252.
    metrics : List[str] or None, optional
        The list of risk metrics to calculate. Choose from: 'sharpe_ratio', 'sortino_ratio',
        'treynor_ratio', 'information_ratio', 'omega_ratio', 'volatility_annualized',
        'skewness', 'kurtosis'. Default is None (all metrics).
    reduce_memory : bool, optional
        If True, reduces memory usage of the DataFrame before calculation. Default is False.
    engine : str, optional
        The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame augmented with columns:
        - {close_column}_sharpe_ratio_{window}: Rolling Sharpe Ratio
        - {close_column}_sortino_ratio_{window}: Rolling Sortino Ratio
        - {close_column}_treynor_ratio_{window}: Rolling Treynor Ratio (if benchmark provided)
        - {close_column}_information_ratio_{window}: Rolling Information Ratio (if benchmark provided)
        - {close_column}_omega_ratio_{window}: Rolling Omega Ratio
        - {close_column}_volatility_annualized_{window}: Rolling annualized volatility
        - {close_column}_skewness_{window}: Rolling skewness of returns
        - {close_column}_kurtosis_{window}: Rolling kurtosis of returns

    Notes
    -----
    This function computes returns from closing prices and calculates rolling risk metrics:

    - Sharpe Ratio: Excess return over risk-free rate divided by volatility
    - Sortino Ratio: Excess return over risk-free rate divided by downside deviation
    - Treynor Ratio: Excess return over risk-free rate divided by beta (requires benchmark)
    - Information Ratio: Excess return over benchmark divided by tracking error (requires benchmark)
    - Omega Ratio: Ratio of gains to losses above/below a threshold
    - Volatility: Annualized standard deviation of returns
    - Skewness: Asymmetry of return distribution
    - Kurtosis: Fat-tailedness of return distribution

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
    # Rolling risk metrics - single stock (pandas)
    risk_single = (
        df
        .query("symbol == 'AAPL'")
        .augment_rolling_risk_metrics(
            date_column="date",
            close_column="adjusted",
            window=252,
        )
    )

    risk_single.glimpse()
    ```

    ```{python}
    # Rolling risk metrics - polars grouped
    pl_df = pl.from_pandas(df)
    risk_polars = (
        pl_df
        .group_by("symbol")
        .tk.augment_rolling_risk_metrics(
            date_column="date",
            close_column="adjusted",
            window=60,
        )
    )

    risk_polars.glimpse()
    ```

    ```{python}
    # Rolling risk metrics - selective pandas metrics
    risk_selected = (
        df
        .groupby("symbol")
        .augment_rolling_risk_metrics(
            date_column="date",
            close_column="adjusted",
            window=252,
            metrics=["sharpe_ratio", "sortino_ratio", "volatility_annualized"],
        )
    )

    risk_selected.glimpse()
    ```

    ```{python}
    from pytimetk.utils.selection import contains

    selector_df = (
        df
        .groupby("symbol")
        .augment_rolling_risk_metrics(
            date_column=contains("dat"),
            close_column=contains("adj"),
            benchmark_column=contains("clos"),
            window=63,
            metrics=["sharpe_ratio"],
        )
    )

    selector_df.glimpse()
    ```
    """

    # Define all available metrics
    ALL_METRICS = [
        "sharpe_ratio",
        "sortino_ratio",
        "treynor_ratio",
        "information_ratio",
        "omega_ratio",
        "volatility_annualized",
        "skewness",
        "kurtosis",
    ]

    # Set default metrics to all if None
    if metrics is None:
        metrics = ALL_METRICS
    else:
        # Validate metrics
        invalid_metrics = [m for m in metrics if m not in ALL_METRICS]
        if invalid_metrics:
            raise ValueError(
                f"Invalid metrics: {invalid_metrics}. Choose from {ALL_METRICS}"
            )
        # Ensure benchmark-dependent metrics require benchmark_column
        benchmark_metrics = ["treynor_ratio", "information_ratio"]
        if any(m in metrics for m in benchmark_metrics) and benchmark_column is None:
            raise ValueError(
                "Metrics 'treynor_ratio' and 'information_ratio' require a benchmark_column"
            )

    # Convert a single integer to a list and reject invalid rolling windows early.
    windows = [window] if isinstance(window, Integral) else window
    if not isinstance(windows, list) or any(
        isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
        for value in windows
    ):
        raise ValueError("All `window` values must be positive integers.")
    windows = [int(value) for value in windows]

    # Existing checks...
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

    if benchmark_column is not None:
        _, benchmark_columns = resolve_shift_columns(
            data,
            date_column=date_column,
            value_column=benchmark_column,
            require_numeric=True,
        )
        if len(benchmark_columns) != 1:
            raise ValueError(
                "`benchmark_column` selector must resolve to exactly one column."
            )
        benchmark_column = benchmark_columns[0]

    engine_resolved = normalize_engine(engine, data)
    if (
        engine_resolved == "cudf" and cudf is None
    ):  # pragma: no cover - optional dependency
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
        result = _augment_rolling_risk_metrics_pandas(
            data=sorted_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            risk_free_rate=risk_free_rate,
            benchmark_column=benchmark_column,
            annualization_factor=annualization_factor,
            metrics=metrics,
        )
        if reduce_memory:
            result = reduce_memory_usage(result)
    elif conversion_engine == "cudf":
        cudf_df = prepared_data.obj if hasattr(prepared_data, "obj") else prepared_data
        if not isinstance(cudf_df, cudf.DataFrame):
            warnings.warn(
                "Unsupported cudf object encountered for augment_rolling_risk_metrics. Falling back to pandas.",
                RuntimeWarning,
                stacklevel=2,
            )
            pandas_input = conversion_to_pandas(conversion)
            result = _augment_rolling_risk_metrics_pandas(
                data=pandas_input,
                date_column=date_column,
                close_column=close_column,
                windows=windows,
                risk_free_rate=risk_free_rate,
                benchmark_column=benchmark_column,
                annualization_factor=annualization_factor,
                metrics=metrics,
            )
        else:
            result = _augment_rolling_risk_metrics_cudf_dataframe(
                cudf_df,
                date_column=date_column,
                close_column=close_column,
                windows=windows,
                risk_free_rate=risk_free_rate,
                benchmark_column=benchmark_column,
                annualization_factor=annualization_factor,
                metrics=metrics,
                group_columns=conversion.group_columns,
                row_id_column=conversion.row_id_column,
            )
    else:
        result = _augment_rolling_risk_metrics_polars(
            data=prepared_data,
            date_column=date_column,
            close_column=close_column,
            windows=windows,
            risk_free_rate=risk_free_rate,
            benchmark_column=benchmark_column,
            annualization_factor=annualization_factor,
            metrics=metrics,
            group_columns=conversion.group_columns,
            row_id_column=conversion.row_id_column,
        )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_rolling_risk_metrics_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    risk_free_rate: float,
    benchmark_column: Optional[str],
    annualization_factor: int,
    metrics: List[str],
) -> pd.DataFrame:
    """Pandas implementation of rolling risk metrics calculation with selective metrics."""
    if isinstance(data, pd.DataFrame):
        df = data.copy(deep=False)
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy(deep=False)

    col = close_column
    returns_col = "__rrm_returns"
    benchmark_returns_col = "__rrm_benchmark_returns"

    if group_names:
        previous_close = df.groupby(group_names, sort=False)[col].shift(1)
    else:
        previous_close = df[col].shift(1)
    df[returns_col] = np.log(df[col] / previous_close)

    benchmark_required = benchmark_column is not None and any(
        metric in metrics for metric in ["treynor_ratio", "information_ratio"]
    )
    if benchmark_required:
        if group_names:
            previous_benchmark = df.groupby(group_names, sort=False)[
                benchmark_column
            ].shift(1)
        else:
            previous_benchmark = df[benchmark_column].shift(1)
        df[benchmark_returns_col] = np.log(df[benchmark_column] / previous_benchmark)

    df["__rrm_downside_squared"] = df[returns_col].clip(upper=0).pow(2)
    df["__rrm_positive_returns"] = df[returns_col].clip(lower=0)
    df["__rrm_negative_returns"] = df[returns_col].clip(upper=0)

    if benchmark_required:
        df["__rrm_return_benchmark_product"] = (
            df[returns_col] * df[benchmark_returns_col]
        )
        df["__rrm_benchmark_squared"] = df[benchmark_returns_col].pow(2)
        df["__rrm_active_return"] = df[returns_col] - df[benchmark_returns_col]

    def rolling_aggregate(column: str, window: int, method: str) -> pd.Series:
        min_periods = max(1, window // 2)

        def aggregate(series: pd.Series) -> pd.Series:
            rolling = series.rolling(window, min_periods=min_periods)
            return getattr(rolling, method)()

        if group_names:
            return df.groupby(group_names, sort=False)[column].transform(aggregate)
        return aggregate(df[column])

    def rolling_moment(column: str, window: int, moment: str) -> pd.Series:
        min_periods = max(1, window // 2)
        function = stats.skew if moment == "skew" else stats.kurtosis

        def aggregate(series: pd.Series) -> pd.Series:
            return series.rolling(window, min_periods=min_periods).apply(
                lambda values: function(values, nan_policy="omit"), raw=True
            )

        if group_names:
            return df.groupby(group_names, sort=False)[column].transform(aggregate)
        return aggregate(df[column])

    annualization = np.sqrt(annualization_factor)

    for w in windows:
        mean_ret = rolling_aggregate(returns_col, w, "mean")

        if any(
            metric in metrics for metric in ["sharpe_ratio", "volatility_annualized"]
        ):
            std_ret = rolling_aggregate(returns_col, w, "std")

        if "sharpe_ratio" in metrics:
            df[f"{col}_sharpe_ratio_{w}"] = (
                (mean_ret - risk_free_rate) / std_ret * annualization
            )

        if "sortino_ratio" in metrics:
            downside_deviation = np.sqrt(
                rolling_aggregate("__rrm_downside_squared", w, "mean")
            ).replace(0, np.nan)
            df[f"{col}_sortino_ratio_{w}"] = (
                (mean_ret - risk_free_rate) / downside_deviation * annualization
            )

        if "volatility_annualized" in metrics:
            df[f"{col}_volatility_annualized_{w}"] = std_ret * annualization

        if "omega_ratio" in metrics:
            positive_sum = rolling_aggregate("__rrm_positive_returns", w, "sum")
            negative_sum = rolling_aggregate("__rrm_negative_returns", w, "sum")
            df[f"{col}_omega_ratio_{w}"] = positive_sum / (-negative_sum).replace(
                0, np.nan
            )

        if "skewness" in metrics:
            df[f"{col}_skewness_{w}"] = rolling_moment(returns_col, w, "skew")

        if "kurtosis" in metrics:
            df[f"{col}_kurtosis_{w}"] = rolling_moment(returns_col, w, "kurtosis")

        if benchmark_required:
            benchmark_mean = rolling_aggregate(benchmark_returns_col, w, "mean")

            if "treynor_ratio" in metrics:
                mean_product = rolling_aggregate(
                    "__rrm_return_benchmark_product", w, "mean"
                )
                mean_benchmark_squared = rolling_aggregate(
                    "__rrm_benchmark_squared", w, "mean"
                )
                covariance = mean_product - (mean_ret * benchmark_mean)
                benchmark_variance = mean_benchmark_squared - benchmark_mean.pow(2)
                beta = covariance / benchmark_variance.replace(0, np.nan)
                df[f"{col}_treynor_ratio_{w}"] = (
                    (mean_ret - risk_free_rate) / beta * annualization
                )

            if "information_ratio" in metrics:
                tracking_error = rolling_aggregate(
                    "__rrm_active_return", w, "std"
                ).replace(0, np.nan)
                df[f"{col}_information_ratio_{w}"] = (
                    mean_ret - benchmark_mean
                ) / tracking_error

    temporary_columns = [
        returns_col,
        "__rrm_downside_squared",
        "__rrm_positive_returns",
        "__rrm_negative_returns",
    ]
    if benchmark_required:
        temporary_columns.extend(
            [
                benchmark_returns_col,
                "__rrm_return_benchmark_product",
                "__rrm_benchmark_squared",
                "__rrm_active_return",
            ]
        )

    return df.drop(columns=temporary_columns)


def _augment_rolling_risk_metrics_cudf_dataframe(
    frame: "cudf.DataFrame",
    *,
    date_column: str,
    close_column: str,
    windows: List[int],
    risk_free_rate: float,
    benchmark_column: Optional[str],
    annualization_factor: int,
    metrics: List[str],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> "cudf.DataFrame":
    if cudf is None:  # pragma: no cover - optional dependency
        raise ImportError(
            "cudf is required to execute the cudf rolling risk metrics backend."
        )

    sort_columns: List[str] = [date_column]
    if group_columns:
        sort_columns = list(group_columns) + sort_columns

    df_sorted = frame.sort_values(sort_columns)
    df_sorted[close_column] = df_sorted[close_column].astype("float64")

    if group_columns:
        group_list = list(group_columns)
        prev_close = df_sorted.groupby(group_list, sort=False)[close_column].shift(1)
    else:
        group_list = None
        prev_close = df_sorted[close_column].shift(1)

    ratio = df_sorted[close_column] / prev_close
    ratio = ratio.where(prev_close != 0)
    df_sorted["__rrm_returns"] = cudf.Series(np.log(ratio))

    df_sorted["__rrm_pos"] = df_sorted["__rrm_returns"].where(
        df_sorted["__rrm_returns"] > 0, 0.0
    )
    df_sorted["__rrm_pos"] = df_sorted["__rrm_pos"].where(
        df_sorted["__rrm_returns"].notna()
    )
    df_sorted["__rrm_neg"] = df_sorted["__rrm_returns"].where(
        df_sorted["__rrm_returns"] < 0, 0.0
    )
    df_sorted["__rrm_neg"] = df_sorted["__rrm_neg"].where(
        df_sorted["__rrm_returns"].notna()
    )
    df_sorted["__rrm_neg_sq"] = df_sorted["__rrm_neg"] ** 2
    df_sorted["__rrm_returns_sq"] = df_sorted["__rrm_returns"] ** 2
    df_sorted["__rrm_returns_cu"] = df_sorted["__rrm_returns"] ** 3
    df_sorted["__rrm_returns_qu"] = df_sorted["__rrm_returns"] ** 4

    if benchmark_column is not None:
        df_sorted[benchmark_column] = df_sorted[benchmark_column].astype("float64")
        if group_list:
            prev_bench = df_sorted.groupby(group_list, sort=False)[
                benchmark_column
            ].shift(1)
        else:
            prev_bench = df_sorted[benchmark_column].shift(1)
        bench_ratio = df_sorted[benchmark_column] / prev_bench
        bench_ratio = bench_ratio.where(prev_bench != 0)
        df_sorted["__rrm_bench_returns"] = cudf.Series(np.log(bench_ratio))
        df_sorted["__rrm_bench_sq"] = df_sorted["__rrm_bench_returns"] ** 2
        df_sorted["__rrm_ret_bench"] = (
            df_sorted["__rrm_returns"] * df_sorted["__rrm_bench_returns"]
        )
        df_sorted["__rrm_diff_returns"] = (
            df_sorted["__rrm_returns"] - df_sorted["__rrm_bench_returns"]
        )

    if group_list:
        grouped_returns = df_sorted.groupby(group_list, sort=False)["__rrm_returns"]
        grouped_returns_sq = df_sorted.groupby(group_list, sort=False)[
            "__rrm_returns_sq"
        ]
        grouped_returns_cu = df_sorted.groupby(group_list, sort=False)[
            "__rrm_returns_cu"
        ]
        grouped_returns_qu = df_sorted.groupby(group_list, sort=False)[
            "__rrm_returns_qu"
        ]
        grouped_neg_sq = df_sorted.groupby(group_list, sort=False)["__rrm_neg_sq"]
        grouped_pos = df_sorted.groupby(group_list, sort=False)["__rrm_pos"]
        grouped_neg = df_sorted.groupby(group_list, sort=False)["__rrm_neg"]
        if benchmark_column is not None:
            grouped_bench_returns = df_sorted.groupby(group_list, sort=False)[
                "__rrm_bench_returns"
            ]
            grouped_bench_sq = df_sorted.groupby(group_list, sort=False)[
                "__rrm_bench_sq"
            ]
            grouped_ret_bench = df_sorted.groupby(group_list, sort=False)[
                "__rrm_ret_bench"
            ]
            grouped_diff_returns = df_sorted.groupby(group_list, sort=False)[
                "__rrm_diff_returns"
            ]

    for w in windows:
        min_periods = max(1, w // 2)
        if group_list:
            mean_ret = (
                grouped_returns.rolling(window=w, min_periods=min_periods)
                .mean()
                .reset_index(drop=True)
            )
            std_ret = (
                grouped_returns.rolling(window=w, min_periods=min_periods)
                .std()
                .reset_index(drop=True)
            )
            count = (
                grouped_returns.rolling(window=w, min_periods=min_periods)
                .count()
                .reset_index(drop=True)
            )
            neg_sq = (
                grouped_neg_sq.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            pos_sum = (
                grouped_pos.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            neg_sum = (
                grouped_neg.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            sum_returns = (
                grouped_returns.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            sum_sq = (
                grouped_returns_sq.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            sum_cu = (
                grouped_returns_cu.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            sum_qu = (
                grouped_returns_qu.rolling(window=w, min_periods=min_periods)
                .sum()
                .reset_index(drop=True)
            )
            if benchmark_column is not None:
                bench_mean = (
                    grouped_bench_returns.rolling(window=w, min_periods=min_periods)
                    .mean()
                    .reset_index(drop=True)
                )
                sum_bench_sq = (
                    grouped_bench_sq.rolling(window=w, min_periods=min_periods)
                    .sum()
                    .reset_index(drop=True)
                )
                sum_ret_bench = (
                    grouped_ret_bench.rolling(window=w, min_periods=min_periods)
                    .sum()
                    .reset_index(drop=True)
                )
                diff_std = (
                    grouped_diff_returns.rolling(window=w, min_periods=min_periods)
                    .std()
                    .reset_index(drop=True)
                )
        else:
            mean_ret = (
                df_sorted["__rrm_returns"]
                .rolling(window=w, min_periods=min_periods)
                .mean()
            )
            std_ret = (
                df_sorted["__rrm_returns"]
                .rolling(window=w, min_periods=min_periods)
                .std()
            )
            count = (
                df_sorted["__rrm_returns"]
                .rolling(window=w, min_periods=min_periods)
                .count()
            )
            neg_sq = (
                df_sorted["__rrm_neg_sq"]
                .rolling(window=w, min_periods=min_periods)
                .sum()
            )
            pos_sum = (
                df_sorted["__rrm_pos"].rolling(window=w, min_periods=min_periods).sum()
            )
            neg_sum = (
                df_sorted["__rrm_neg"].rolling(window=w, min_periods=min_periods).sum()
            )
            sum_returns = (
                df_sorted["__rrm_returns"]
                .rolling(window=w, min_periods=min_periods)
                .sum()
            )
            sum_sq = (
                df_sorted["__rrm_returns_sq"]
                .rolling(window=w, min_periods=min_periods)
                .sum()
            )
            sum_cu = (
                df_sorted["__rrm_returns_cu"]
                .rolling(window=w, min_periods=min_periods)
                .sum()
            )
            sum_qu = (
                df_sorted["__rrm_returns_qu"]
                .rolling(window=w, min_periods=min_periods)
                .sum()
            )
            if benchmark_column is not None:
                bench_mean = (
                    df_sorted["__rrm_bench_returns"]
                    .rolling(window=w, min_periods=min_periods)
                    .mean()
                )
                sum_bench_sq = (
                    df_sorted["__rrm_bench_sq"]
                    .rolling(window=w, min_periods=min_periods)
                    .sum()
                )
                sum_ret_bench = (
                    df_sorted["__rrm_ret_bench"]
                    .rolling(window=w, min_periods=min_periods)
                    .sum()
                )
                diff_std = (
                    df_sorted["__rrm_diff_returns"]
                    .rolling(window=w, min_periods=min_periods)
                    .std()
                )

        if "sharpe_ratio" in metrics:
            sharpe = ((mean_ret - risk_free_rate) / std_ret) * np.sqrt(
                annualization_factor
            )
            df_sorted[f"{close_column}_sharpe_ratio_{w}"] = sharpe

        if "volatility_annualized" in metrics:
            volatility = std_ret * np.sqrt(annualization_factor)
            df_sorted[f"{close_column}_volatility_annualized_{w}"] = volatility

        if "sortino_ratio" in metrics:
            downside_var = neg_sq / count
            downside_std = downside_var.pow(0.5)
            sortino = ((mean_ret - risk_free_rate) / downside_std) * np.sqrt(
                annualization_factor
            )
            sortino = sortino.where(downside_std > 0, np.nan)
            df_sorted[f"{close_column}_sortino_ratio_{w}"] = sortino

        if "omega_ratio" in metrics:
            omega = pos_sum / (-neg_sum)
            omega = omega.where(neg_sum < 0, np.nan)
            df_sorted[f"{close_column}_omega_ratio_{w}"] = omega

        if "skewness" in metrics or "kurtosis" in metrics:
            with cudf.option_context("mode.null_division", np.nan):
                avg = sum_returns / count
            variance = (sum_sq / count) - avg.pow(2)
            std_pop = variance.where(variance > 0, np.nan).pow(0.5)
            if "skewness" in metrics:
                mu3 = (sum_cu / count) - 3 * avg * (sum_sq / count) + 2 * avg.pow(3)
                skew = mu3 / std_pop.pow(3)
                df_sorted[f"{close_column}_skewness_{w}"] = skew
            if "kurtosis" in metrics:
                mu4 = (
                    (sum_qu / count)
                    - 4 * avg * (sum_cu / count)
                    + 6 * avg.pow(2) * (sum_sq / count)
                    - 3 * avg.pow(4)
                )
                kurt = mu4 / std_pop.pow(4) - 3
                df_sorted[f"{close_column}_kurtosis_{w}"] = kurt

        if benchmark_column is not None and (
            "treynor_ratio" in metrics or "information_ratio" in metrics
        ):
            cov = (sum_ret_bench / count) - (mean_ret * bench_mean)
            bench_var = (sum_bench_sq / count) - bench_mean.pow(2)
            beta = cov / bench_var
            beta = beta.where(bench_var != 0, np.nan)
            if "treynor_ratio" in metrics:
                treynor = ((mean_ret - risk_free_rate) / beta) * np.sqrt(
                    annualization_factor
                )
                df_sorted[f"{close_column}_treynor_ratio_{w}"] = treynor
            if "information_ratio" in metrics:
                info = (mean_ret - bench_mean) / diff_std
                df_sorted[f"{close_column}_information_ratio_{w}"] = info

    drop_cols = [
        "__rrm_returns",
        "__rrm_pos",
        "__rrm_neg",
        "__rrm_neg_sq",
        "__rrm_returns_sq",
        "__rrm_returns_cu",
        "__rrm_returns_qu",
    ]

    if benchmark_column is not None:
        drop_cols.extend(
            [
                "__rrm_bench_returns",
                "__rrm_bench_sq",
                "__rrm_ret_bench",
                "__rrm_diff_returns",
            ]
        )

    df_sorted = df_sorted.drop(columns=drop_cols)

    if row_id_column and row_id_column in df_sorted.columns:
        df_sorted = df_sorted.sort_values(row_id_column)

    return df_sorted


def _augment_rolling_risk_metrics_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    close_column: str,
    windows: List[int],
    risk_free_rate: float,
    benchmark_column: Optional[str],
    annualization_factor: int,
    metrics: List[str],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data
    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    df = frame_with_id.sort(sort_keys)
    col = close_column
    original_cols = df.columns

    returns_alias = "__rrm_returns"
    downside_squared_alias = "__rrm_downside_squared"
    positive_returns_alias = "__rrm_positive_returns"
    negative_returns_alias = "__rrm_negative_returns"
    benchmark_returns_alias = "__rrm_benchmark_returns"

    returns_expr = pl.col(col).log() - pl.col(col).log().shift(1)
    if resolved_groups:
        returns_expr = returns_expr.over(resolved_groups)
    df = df.with_columns(returns_expr.alias(returns_alias))
    df = df.with_columns(
        pl.when(pl.col(returns_alias).is_null())
        .then(None)
        .when(pl.col(returns_alias) < 0)
        .then(pl.col(returns_alias) ** 2)
        .otherwise(0.0)
        .alias(downside_squared_alias),
        pl.when(pl.col(returns_alias).is_null())
        .then(None)
        .otherwise(pl.col(returns_alias).clip(lower_bound=0))
        .alias(positive_returns_alias),
        pl.when(pl.col(returns_alias).is_null())
        .then(None)
        .otherwise(pl.col(returns_alias).clip(upper_bound=0))
        .alias(negative_returns_alias),
    )

    benchmark_required = benchmark_column is not None and any(
        m in metrics for m in ["treynor_ratio", "information_ratio"]
    )
    if benchmark_required:
        bench_returns = pl.col(benchmark_column).log() - pl.col(
            benchmark_column
        ).log().shift(1)
        if resolved_groups:
            bench_returns = bench_returns.over(resolved_groups)
        df = df.with_columns(bench_returns.alias(benchmark_returns_alias))

    # Loop over each window separately
    for w in windows:
        min_periods = max(1, w // 2)
        exprs = []
        returns = pl.col(returns_alias)
        mean_ret = returns.rolling_mean(w, min_periods=min_periods)

        if "sharpe_ratio" in metrics:
            std_ret = returns.rolling_std(w, min_periods=min_periods)
            exprs.append(
                (
                    (mean_ret - risk_free_rate)
                    / std_ret
                    * pl.lit(np.sqrt(annualization_factor))
                ).alias(f"{col}_sharpe_ratio_{w}")
            )
        if "volatility_annualized" in metrics:
            exprs.append(
                (
                    returns.rolling_std(w, min_periods=min_periods)
                    * pl.lit(np.sqrt(annualization_factor))
                ).alias(f"{col}_volatility_annualized_{w}")
            )
        if "sortino_ratio" in metrics:
            downside_deviation = (
                pl.col(downside_squared_alias)
                .rolling_mean(w, min_periods=min_periods)
                .sqrt()
            )
            exprs.append(
                pl.when(downside_deviation > 0)
                .then(
                    (mean_ret - risk_free_rate)
                    / downside_deviation
                    * pl.lit(np.sqrt(annualization_factor))
                )
                .otherwise(None)
                .alias(f"{col}_sortino_ratio_{w}")
            )
        if "omega_ratio" in metrics:
            positive_sum = pl.col(positive_returns_alias).rolling_sum(
                w, min_periods=min_periods
            )
            negative_sum = pl.col(negative_returns_alias).rolling_sum(
                w, min_periods=min_periods
            )
            exprs.append(
                pl.when(negative_sum < 0)
                .then(positive_sum / negative_sum.abs())
                .otherwise(None)
                .alias(f"{col}_omega_ratio_{w}")
            )
        if "skewness" in metrics or "kurtosis" in metrics:
            mean_1 = mean_ret
            mean_2 = (returns**2).rolling_mean(w, min_periods=min_periods)
            variance = mean_2 - mean_1**2

            if "skewness" in metrics:
                mean_3 = (returns**3).rolling_mean(w, min_periods=min_periods)
                third_moment = mean_3 - 3 * mean_1 * mean_2 + 2 * mean_1**3
                exprs.append(
                    pl.when(variance > 0)
                    .then(third_moment / (variance**1.5))
                    .otherwise(None)
                    .alias(f"{col}_skewness_{w}")
                )

            if "kurtosis" in metrics:
                mean_3 = (returns**3).rolling_mean(w, min_periods=min_periods)
                mean_4 = (returns**4).rolling_mean(w, min_periods=min_periods)
                fourth_moment = (
                    mean_4
                    - 4 * mean_1 * mean_3
                    + 6 * mean_1**2 * mean_2
                    - 3 * mean_1**4
                )
                exprs.append(
                    pl.when(variance > 0)
                    .then(fourth_moment / (variance**2) - 3)
                    .otherwise(None)
                    .alias(f"{col}_kurtosis_{w}")
                )

        if benchmark_required:
            benchmark_returns = pl.col(benchmark_returns_alias)
            benchmark_mean = benchmark_returns.rolling_mean(w, min_periods=min_periods)

            if "treynor_ratio" in metrics:
                mean_product = (returns * benchmark_returns).rolling_mean(
                    w, min_periods=min_periods
                )
                mean_benchmark_squared = (benchmark_returns**2).rolling_mean(
                    w, min_periods=min_periods
                )
                covariance = mean_product - (mean_ret * benchmark_mean)
                benchmark_variance = mean_benchmark_squared - benchmark_mean**2
                beta = covariance / benchmark_variance
                exprs.append(
                    pl.when(benchmark_variance != 0)
                    .then(
                        (mean_ret - risk_free_rate)
                        / beta
                        * pl.lit(np.sqrt(annualization_factor))
                    )
                    .otherwise(None)
                    .alias(f"{col}_treynor_ratio_{w}")
                )

            if "information_ratio" in metrics:
                tracking_error = (returns - benchmark_returns).rolling_std(
                    w, min_periods=min_periods
                )
                exprs.append(
                    pl.when(tracking_error != 0)
                    .then((mean_ret - benchmark_mean) / tracking_error)
                    .otherwise(None)
                    .alias(f"{col}_information_ratio_{w}")
                )

        # Apply the expressions for this window in a separate call
        if resolved_groups:
            df = df.with_columns([e.over(resolved_groups) for e in exprs])
        else:
            df = df.with_columns(exprs)

    temporary_columns = [
        returns_alias,
        downside_squared_alias,
        positive_returns_alias,
        negative_returns_alias,
    ]
    if benchmark_required:
        temporary_columns.append(benchmark_returns_alias)
    df = df.drop(temporary_columns)

    # Order columns
    metric_cols = [c for c in df.columns if c not in original_cols]
    df = df.select(original_cols + metric_cols)

    df = df.sort(row_col)

    if generated:
        df = df.drop(row_col)

    return df

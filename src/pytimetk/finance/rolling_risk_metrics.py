import pandas as pd
import polars as pl
import pandas_flavor as pf
import numpy as np
import warnings
from typing import List, Optional, Sequence, Union

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
    date_column: str,
    close_column: str,
    window: Union[int, List[int]] = 252,
    risk_free_rate: float = 0.0,
    benchmark_column: Optional[str] = None,
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
    date_column : str
        The name of the column containing dates or timestamps.
    close_column : str
        The column containing closing prices to calculate returns and risk metrics from.
    window : int, optional
        The rolling window size for calculations (e.g., 252 for annual). Default is 252.
    risk_free_rate : float, optional
        The assumed risk-free rate (e.g., 0.0 for 0%). Default is 0.0.
    benchmark_column : str or None, optional
        The column containing benchmark returns (e.g., market index) for Treynor and Information Ratios.
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

    df = tk.load_dataset('stocks_daily', parse_dates=['date'])

    # Single stock risk metrics
    risk_df = (
        df.query("symbol == 'AAPL'")
        .augment_rolling_risk_metrics(
            date_column='date',
            close_column='adjusted',
            window=252
        )
    )
    risk_df.head()
    ```

    ```{python}
    # Multiple stocks with groupby and benchmark (polars)
    pl_df = pl.from_pandas(df)
    risk_df = (
        pl_df.group_by('symbol')
        .tk.augment_rolling_risk_metrics(
            date_column='date',
            close_column='adjusted',
            # benchmark_column='market_adjusted_returns',  # Use if a benchmark returns column exists
            window=60,
        )
    )
    risk_df.head()
    ```

    ```{python}
    # Selective metrics
    risk_df = (
        df.groupby('symbol')
        .augment_rolling_risk_metrics(
            date_column='date',
            close_column='adjusted',
            window=252,
            metrics=['sharpe_ratio', 'sortino_ratio', 'volatility_annualized'],
        )
    )
    risk_df.tail()
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

    # Convert single int to list for consistency
    windows = [window] if isinstance(window, int) else window

    # Existing checks...
    check_dataframe_or_groupby(data)
    check_value_column(data, close_column)
    check_date_column(data, date_column)
    if benchmark_column is not None:
        check_value_column(data, benchmark_column)

    engine_resolved = normalize_engine(engine, data)
    if engine_resolved == "cudf" and cudf is None:  # pragma: no cover - optional dependency
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
        df = data.copy()
        group_names = None
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = resolve_pandas_groupby_frame(data).copy()

    col = close_column
    # Calculate log returns only if needed by selected metrics
    required_returns = any(
        m in metrics
        for m in [
            "sharpe_ratio",
            "sortino_ratio",
            "treynor_ratio",
            "information_ratio",
            "omega_ratio",
            "volatility_annualized",
            "skewness",
            "kurtosis",
        ]
    )
    if required_returns:
        df[f"{col}_returns"] = np.log(df[col] / df[col].shift(1))

    if benchmark_column and any(
        m in metrics for m in ["treynor_ratio", "information_ratio"]
    ):
        df[f"{benchmark_column}_returns"] = np.log(
            df[benchmark_column] / df[benchmark_column].shift(1)
        )

    # Define helper functions only if needed
    def roll_downside_std(ser, window_size, **kwargs):
        return ser.rolling(window_size, min_periods=window_size // 2).apply(
            lambda x: np.std(x[x < 0]) if np.any(x < 0) else np.nan, raw=True
        )

    def roll_omega(ser, window_size, **kwargs):
        return (
            ser.rolling(window_size, min_periods=window_size // 2)
            .apply(
                lambda x: (
                    np.sum(x[x > 0]) / np.abs(np.sum(x[x < 0]))
                    if np.sum(x[x < 0]) != 0
                    else np.inf
                )
                if np.sum(~np.isnan(x)) >= window_size // 2
                else np.nan,
                raw=True,
            )
            .replace([np.inf, -np.inf], np.nan)
        )

    def roll_beta(ser, bench_ser, window_size, **kwargs):
        cov = ser.rolling(window_size, min_periods=window_size // 2).cov(bench_ser)
        var = bench_ser.rolling(window_size, min_periods=window_size // 2).var()
        return cov / var.where(var != 0, np.nan)

    # Apply rolling calculations for each window
    if group_names:
        grouped = df.groupby(group_names)

        for w in windows:
            # Precompute only needed metrics
            mean_ret = (
                grouped[f"{col}_returns"].rolling(w, min_periods=w // 2).mean()
                if required_returns
                else None
            )
            std_ret = (
                grouped[f"{col}_returns"].rolling(w, min_periods=w // 2).std()
                if any(m in metrics for m in ["sharpe_ratio", "volatility_annualized"])
                else None
            )
            downside_std = (
                grouped[f"{col}_returns"].apply(
                    roll_downside_std, raw=False, window_size=w
                )
                if "sortino_ratio" in metrics
                else None
            )
            omega = (
                grouped[f"{col}_returns"].apply(roll_omega, raw=False, window_size=w)
                if "omega_ratio" in metrics
                else None
            )
            skew = (
                grouped[f"{col}_returns"]
                .rolling(w, min_periods=w // 2)
                .apply(lambda x: stats.skew(x, nan_policy="omit"), raw=True)
                if "skewness" in metrics
                else None
            )
            kurt = (
                grouped[f"{col}_returns"]
                .rolling(w, min_periods=w // 2)
                .apply(lambda x: stats.kurtosis(x, nan_policy="omit"), raw=True)
                if "kurtosis" in metrics
                else None
            )

            # Assign only selected metrics
            if "sharpe_ratio" in metrics:
                df[f"{col}_sharpe_ratio_{w}"] = (
                    (mean_ret - risk_free_rate)
                    / std_ret
                    * np.sqrt(annualization_factor)
                ).reset_index(level=0, drop=True)
            if "sortino_ratio" in metrics:
                df[f"{col}_sortino_ratio_{w}"] = (
                    (mean_ret - risk_free_rate)
                    / downside_std
                    * np.sqrt(annualization_factor)
                ).reset_index(level=0, drop=True)
            if "volatility_annualized" in metrics:
                df[f"{col}_volatility_annualized_{w}"] = (
                    std_ret * np.sqrt(annualization_factor)
                ).reset_index(level=0, drop=True)
            if "omega_ratio" in metrics:
                df[f"{col}_omega_ratio_{w}"] = omega.reset_index(level=0, drop=True)
            if "skewness" in metrics:
                df[f"{col}_skewness_{w}"] = skew.reset_index(level=0, drop=True)
            if "kurtosis" in metrics:
                df[f"{col}_kurtosis_{w}"] = kurt.reset_index(level=0, drop=True)

            if benchmark_column:
                bench_mean = (
                    grouped[f"{benchmark_column}_returns"]
                    .rolling(w, min_periods=w // 2)
                    .mean()
                    if "information_ratio" in metrics
                    else None
                )
                beta = (
                    grouped[f"{col}_returns"].apply(
                        lambda x: roll_beta(
                            x,
                            df.loc[x.index, f"{benchmark_column}_returns"],
                            window_size=w,
                        ),
                        raw=False,
                    )
                    if "treynor_ratio" in metrics
                    else None
                )
                tracking_error = (
                    grouped[f"{col}_returns"].apply(
                        lambda x: (x - df.loc[x.index, f"{benchmark_column}_returns"])
                        .rolling(w, min_periods=w // 2)
                        .std(),
                        raw=False,
                    )
                    if "information_ratio" in metrics
                    else None
                )

                if "treynor_ratio" in metrics:
                    df[f"{col}_treynor_ratio_{w}"] = (
                        (mean_ret - risk_free_rate)
                        / beta
                        * np.sqrt(annualization_factor)
                    ).reset_index(level=0, drop=True)
                if "information_ratio" in metrics:
                    df[f"{col}_information_ratio_{w}"] = (
                        (mean_ret - bench_mean) / tracking_error
                    ).reset_index(level=0, drop=True)
    else:
        for w in windows:
            mean_ret = (
                df[f"{col}_returns"].rolling(w, min_periods=w // 2).mean()
                if required_returns
                else None
            )
            std_ret = (
                df[f"{col}_returns"].rolling(w, min_periods=w // 2).std()
                if any(m in metrics for m in ["sharpe_ratio", "volatility_annualized"])
                else None
            )
            downside_std = (
                roll_downside_std(df[f"{col}_returns"], window_size=w)
                if "sortino_ratio" in metrics
                else None
            )
            omega = (
                roll_omega(df[f"{col}_returns"], window_size=w)
                if "omega_ratio" in metrics
                else None
            )
            skew = (
                df[f"{col}_returns"]
                .rolling(w, min_periods=w // 2)
                .apply(lambda x: stats.skew(x, nan_policy="omit"), raw=True)
                if "skewness" in metrics
                else None
            )
            kurt = (
                df[f"{col}_returns"]
                .rolling(w, min_periods=w // 2)
                .apply(lambda x: stats.kurtosis(x, nan_policy="omit"), raw=True)
                if "kurtosis" in metrics
                else None
            )

            if "sharpe_ratio" in metrics:
                df[f"{col}_sharpe_ratio_{w}"] = (
                    (mean_ret - risk_free_rate)
                    / std_ret
                    * np.sqrt(annualization_factor)
                )
            if "sortino_ratio" in metrics:
                df[f"{col}_sortino_ratio_{w}"] = (
                    (mean_ret - risk_free_rate)
                    / downside_std
                    * np.sqrt(annualization_factor)
                )
            if "volatility_annualized" in metrics:
                df[f"{col}_volatility_annualized_{w}"] = std_ret * np.sqrt(
                    annualization_factor
                )
            if "omega_ratio" in metrics:
                df[f"{col}_omega_ratio_{w}"] = omega
            if "skewness" in metrics:
                df[f"{col}_skewness_{w}"] = skew
            if "kurtosis" in metrics:
                df[f"{col}_kurtosis_{w}"] = kurt

            if benchmark_column:
                bench_mean = (
                    df[f"{benchmark_column}_returns"]
                    .rolling(w, min_periods=w // 2)
                    .mean()
                    if "information_ratio" in metrics
                    else None
                )
                beta = (
                    roll_beta(
                        df[f"{col}_returns"],
                        df[f"{benchmark_column}_returns"],
                        window_size=w,
                    )
                    if "treynor_ratio" in metrics
                    else None
                )
                tracking_error = (
                    (df[f"{col}_returns"] - df[f"{benchmark_column}_returns"])
                    .rolling(w, min_periods=w // 2)
                    .std()
                    if "information_ratio" in metrics
                    else None
                )

                if "treynor_ratio" in metrics:
                    df[f"{col}_treynor_ratio_{w}"] = (
                        (mean_ret - risk_free_rate)
                        / beta
                        * np.sqrt(annualization_factor)
                    )
                if "information_ratio" in metrics:
                    df[f"{col}_information_ratio_{w}"] = (
                        mean_ret - bench_mean
                    ) / tracking_error

    # Drop temporary returns columns if computed
    if required_returns:
        df.drop(columns=[f"{col}_returns"], inplace=True)
    if benchmark_column and any(
        m in metrics for m in ["treynor_ratio", "information_ratio"]
    ):
        df.drop(columns=[f"{benchmark_column}_returns"], inplace=True)

    return df


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
    df_sorted["__rrm_returns"] = cudf.Series(np.log(ratio)).fillna(0)

    df_sorted["__rrm_pos"] = df_sorted["__rrm_returns"].where(
        df_sorted["__rrm_returns"] > 0, 0.0
    )
    df_sorted["__rrm_neg"] = df_sorted["__rrm_returns"].where(
        df_sorted["__rrm_returns"] < 0, 0.0
    )
    df_sorted["__rrm_neg_sq"] = df_sorted["__rrm_neg"] ** 2
    df_sorted["__rrm_neg_mask"] = (
        df_sorted["__rrm_returns"] < 0
    ).astype("float64")
    df_sorted["__rrm_returns_sq"] = df_sorted["__rrm_returns"] ** 2
    df_sorted["__rrm_returns_cu"] = df_sorted["__rrm_returns"] ** 3
    df_sorted["__rrm_returns_qu"] = df_sorted["__rrm_returns"] ** 4

    if benchmark_column is not None:
        df_sorted[benchmark_column] = df_sorted[benchmark_column].astype("float64")
        if group_list:
            prev_bench = df_sorted.groupby(group_list, sort=False)[benchmark_column].shift(1)
        else:
            prev_bench = df_sorted[benchmark_column].shift(1)
        bench_ratio = df_sorted[benchmark_column] / prev_bench
        bench_ratio = bench_ratio.where(prev_bench != 0)
        df_sorted["__rrm_bench_returns"] = cudf.Series(np.log(bench_ratio)).fillna(0)
        df_sorted["__rrm_bench_sq"] = df_sorted["__rrm_bench_returns"] ** 2
        df_sorted["__rrm_ret_bench"] = (
            df_sorted["__rrm_returns"] * df_sorted["__rrm_bench_returns"]
        )
        df_sorted["__rrm_diff_returns"] = (
            df_sorted["__rrm_returns"] - df_sorted["__rrm_bench_returns"]
        )

    if group_list:
        grouped_returns = df_sorted.groupby(group_list, sort=False)["__rrm_returns"]
        grouped_returns_sq = df_sorted.groupby(group_list, sort=False)["__rrm_returns_sq"]
        grouped_returns_cu = df_sorted.groupby(group_list, sort=False)["__rrm_returns_cu"]
        grouped_returns_qu = df_sorted.groupby(group_list, sort=False)["__rrm_returns_qu"]
        grouped_neg_sq = df_sorted.groupby(group_list, sort=False)["__rrm_neg_sq"]
        grouped_neg_mask = df_sorted.groupby(group_list, sort=False)["__rrm_neg_mask"]
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
            neg_count = (
                grouped_neg_mask.rolling(window=w, min_periods=min_periods)
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
            mean_ret = df_sorted["__rrm_returns"].rolling(
                window=w, min_periods=min_periods
            ).mean()
            std_ret = df_sorted["__rrm_returns"].rolling(
                window=w, min_periods=min_periods
            ).std()
            count = df_sorted["__rrm_returns"].rolling(
                window=w, min_periods=min_periods
            ).count()
            neg_sq = df_sorted["__rrm_neg_sq"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            neg_count = df_sorted["__rrm_neg_mask"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            pos_sum = df_sorted["__rrm_pos"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            neg_sum = df_sorted["__rrm_neg"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            sum_returns = df_sorted["__rrm_returns"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            sum_sq = df_sorted["__rrm_returns_sq"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            sum_cu = df_sorted["__rrm_returns_cu"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            sum_qu = df_sorted["__rrm_returns_qu"].rolling(
                window=w, min_periods=min_periods
            ).sum()
            if benchmark_column is not None:
                bench_mean = df_sorted["__rrm_bench_returns"].rolling(
                    window=w, min_periods=min_periods
                ).mean()
                sum_bench_sq = df_sorted["__rrm_bench_sq"].rolling(
                    window=w, min_periods=min_periods
                ).sum()
                sum_ret_bench = df_sorted["__rrm_ret_bench"].rolling(
                    window=w, min_periods=min_periods
                ).sum()
                diff_std = df_sorted["__rrm_diff_returns"].rolling(
                    window=w, min_periods=min_periods
                ).std()

        if "sharpe_ratio" in metrics:
            sharpe = ((mean_ret - risk_free_rate) / std_ret) * np.sqrt(
                annualization_factor
            )
            df_sorted[f"{close_column}_sharpe_ratio_{w}"] = sharpe

        if "volatility_annualized" in metrics:
            volatility = std_ret * np.sqrt(annualization_factor)
            df_sorted[f"{close_column}_volatility_annualized_{w}"] = volatility

        if "sortino_ratio" in metrics:
            downside_var = (neg_sq / neg_count).where(neg_count > 0, np.nan)
            downside_std = downside_var.pow(0.5)
            sortino = (
                (mean_ret - risk_free_rate) / downside_std
            ) * np.sqrt(annualization_factor)
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
                mu3 = (
                    (sum_cu / count)
                    - 3 * avg * (sum_sq / count)
                    + 2 * avg.pow(3)
                )
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
        "__rrm_neg_mask",
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

    # Calculate returns and masks if needed
    required_returns = any(
        m in metrics
        for m in [
            "sharpe_ratio",
            "sortino_ratio",
            "treynor_ratio",
            "information_ratio",
            "omega_ratio",
            "volatility_annualized",
            "skewness",
            "kurtosis",
        ]
    )
    if required_returns:
        returns_expr = pl.col(col).log() - pl.col(col).log().shift(1)
        if resolved_groups:
            returns_expr = returns_expr.over(resolved_groups)
        df = df.with_columns(
            returns_expr.alias(f"{col}_returns"),
            (returns_expr > 0).cast(pl.Float64).alias("pos_mask"),
            (returns_expr < 0).cast(pl.Float64).alias("neg_mask"),
        )
    if benchmark_column and any(
        m in metrics for m in ["treynor_ratio", "information_ratio"]
    ):
        bench_returns = (
            pl.col(benchmark_column).log() - pl.col(benchmark_column).log().shift(1)
        )
        if resolved_groups:
            bench_returns = bench_returns.over(resolved_groups)
        df = df.with_columns(bench_returns.alias(f"{benchmark_column}_returns"))

    # Loop over each window separately
    for w in windows:
        exprs = []
        if "sharpe_ratio" in metrics:
            exprs.append(
                (
                    (
                        pl.col(f"{col}_returns").rolling_mean(w, min_periods=w // 2)
                        - risk_free_rate
                    )
                    / pl.col(f"{col}_returns").rolling_std(w, min_periods=w // 2)
                    * pl.lit(np.sqrt(annualization_factor))
                ).alias(f"{col}_sharpe_ratio_{w}")
            )
        if "volatility_annualized" in metrics:
            exprs.append(
                (
                    pl.col(f"{col}_returns").rolling_std(w, min_periods=w // 2)
                    * pl.lit(np.sqrt(annualization_factor))
                ).alias(f"{col}_volatility_annualized_{w}")
            )
        if "sortino_ratio" in metrics:
            # Note: we use the rolling_std on the product with the negative mask
            exprs.append(
                (
                    (
                        pl.col(f"{col}_returns").rolling_mean(w, min_periods=w // 2)
                        - risk_free_rate
                    )
                    / (pl.col(f"{col}_returns") * pl.col("neg_mask")).rolling_std(
                        w, min_periods=w // 2
                    )
                    * pl.lit(np.sqrt(annualization_factor))
                ).alias(f"{col}_sortino_ratio_{w}")
            )
        if "omega_ratio" in metrics:
            exprs.append(
                (
                    (pl.col(f"{col}_returns") * pl.col("pos_mask")).rolling_sum(
                        w, min_periods=w // 2
                    )
                    / (pl.col(f"{col}_returns") * pl.col("neg_mask"))
                    .rolling_sum(w, min_periods=w // 2)
                    .abs()
                )
                .replace([np.inf, -np.inf], np.nan)
                .alias(f"{col}_omega_ratio_{w}")
            )
        if "skewness" in metrics:
            exprs.append(
                pl.col(f"{col}_returns").rolling_skew(w).alias(f"{col}_skewness_{w}")
            )
        if "kurtosis" in metrics:
            # Fast rolling kurtosis (excess kurtosis = kurtosis - 3)
            # Compute rolling sums of powers over the returns column:
            S1 = pl.col(f"{col}_returns").rolling_sum(window_size=w, min_periods=w)
            S2 = (pl.col(f"{col}_returns") ** 2).rolling_sum(
                window_size=w, min_periods=w
            )
            S3 = (pl.col(f"{col}_returns") ** 3).rolling_sum(
                window_size=w, min_periods=w
            )
            S4 = (pl.col(f"{col}_returns") ** 4).rolling_sum(
                window_size=w, min_periods=w
            )
            mean_expr = S1 / w
            var_expr = S2 / w - mean_expr**2
            m4_expr = (
                (S4 / w)
                - 4 * mean_expr * (S3 / w)
                + 6 * mean_expr**2 * (S2 / w)
                - 3 * mean_expr**4
            )
            kurt_expr = m4_expr / (var_expr**2)
            excess_kurt_expr = kurt_expr - 3
            exprs.append(excess_kurt_expr.alias(f"{col}_kurtosis_{w}"))
            # For benchmark-dependent metrics, you would add similar expressions here.

        # Apply the expressions for this window in a separate call
        if resolved_groups:
            df = df.with_columns(
                [e.over(resolved_groups) for e in exprs]
            )
        else:
            df = df.with_columns(exprs)

    # Drop temporary columns
    if required_returns:
        df = df.drop([f"{col}_returns", "pos_mask", "neg_mask"])
    if benchmark_column and any(
        m in metrics for m in ["treynor_ratio", "information_ratio"]
    ):
        df = df.drop([f"{benchmark_column}_returns"])

    # Order columns
    metric_cols = [c for c in df.columns if c not in original_cols]
    df = df.select(original_cols + metric_cols)

    df = df.sort(row_col)

    if generated:
        df = df.drop(row_col)

    return df

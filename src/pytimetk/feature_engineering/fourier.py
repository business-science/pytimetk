import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
import warnings
from typing import Tuple
from typing import Union, List

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
)

from pytimetk.core.ts_summary import ts_summary
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    resolve_pandas_groupby_frame,
    restore_output_type,
)


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_fourier(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    max_order: int = 1,
    reduce_memory: bool = True,
    engine: str = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Adds Fourier transforms to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_fourier` function takes a Pandas DataFrame or GroupBy object, a date column, a value column or list of value columns, the number of periods for the Fourier series, and the maximum Fourier order, and adds Fourier-transformed columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to add Fourier-transformed columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to compute the Fourier transforms.
    periods : int or list, optional
        The `periods` parameter specifies how many timesteps between each peak in the fourier series. Default is 1.
    max_order : int, optional
        The `max_order` parameter specifies the maximum Fourier order to calculate. Default is 1.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for
        augmenting lags. It can be either "pandas" or "polars".

        - The default value is "pandas".

        - When "polars", the function will internally use the `polars` library.
        This can be faster than using "pandas" for large datasets.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with Fourier-transformed columns added to it.

    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])

    # Example 1 - Add Fourier transforms for a single column
    fourier_df = (
        df
            .query("id == 'D10'")
            .augment_fourier(
                date_column='date',
                periods=[1, 7],
                max_order=1
            )
    )
    fourier_df.head()

    fourier_df.plot_timeseries("date", "date_sin_1_7", x_axis_date_labels = "%B %d, %Y",)
    ```

    ``` {python}
    # Example 2 - Add Fourier transforms for grouped data
    fourier_df = (
        df
            .groupby("id")
            .augment_fourier(
                date_column='date',
                periods=[1, 7],
                max_order=1,
                engine= "pandas"
            )
    )
    fourier_df
    ```

    ``` {python}
    # Example 3 - Add Fourier transforms for grouped data
    fourier_df = (
        df
            .groupby("id")
            .augment_fourier(
                date_column='date',
                periods=[1, 7],
                max_order=1,
                engine= "polars"
            )
    )
    fourier_df
    ```

    ``` {python}
    # Example 4 - Polars DataFrame using the tk accessor
    import polars as pl


    pl_fourier = (
        pl.from_pandas(df)
          .group_by("id")
          .tk.augment_fourier(
              date_column='date',
              periods=[1, 7],
              max_order=1,
          )
    )
    ```

    """

    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    conversion: FrameConversion = convert_to_engine(data, "pandas")
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(
            f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list."
        )

    periods = [int(p) for p in periods]

    sorted_data, _ = sort_dataframe(prepared_data, date_column, keep_grouped_df=True)

    if isinstance(prepared_data, pd.core.groupby.generic.DataFrameGroupBy):
        base_df = resolve_pandas_groupby_frame(prepared_data).copy()
    else:
        base_df = prepared_data.copy()

    result = _augment_fourier_pandas(
        base_df,
        date_column,
        periods,
        max_order,
    )

    if reduce_memory and engine_resolved == "pandas":
        result = reduce_memory_usage(result)

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored

    return restored


def calc_fourier(x, period, type: str, K=1):
    term = K / period
    return (
        np.sin(2 * np.pi * term * x) if type == "sin" else np.cos(2 * np.pi * term * x)
    )


def date_to_seq_scale_factor(
    data: pd.DataFrame, date_var: str, engine: str = "pandas"
) -> pd.DataFrame:
    return ts_summary(data, date_column=date_var, engine=engine)["diff_median"]


def _augment_fourier_pandas(
    base_df: pd.DataFrame,
    date_column: str,
    periods: List[int],
    max_order: int,
) -> pd.DataFrame:
    sorted_df = base_df.sort_values(date_column)
    new_cols_sorted = _compute_fourier_columns(
        sorted_df, date_column, periods, max_order
    )
    if new_cols_sorted.empty:
        return base_df.copy()

    new_cols = new_cols_sorted.reindex(base_df.index)
    result_df = base_df.copy()
    result_df[new_cols.columns] = new_cols
    return result_df


def _compute_fourier_columns(
    frame: pd.DataFrame,
    date_column: str,
    periods: List[int],
    max_order: int,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    scale_factor_series = date_to_seq_scale_factor(frame, date_column)
    if scale_factor_series.empty:
        raise ValueError(
            "Unable to compute a scale factor for Fourier features. Check that the input data contains more than one observation."
        )

    scale_factor = scale_factor_series.iloc[0].total_seconds()
    if scale_factor == 0:
        raise ValueError(
            "Time difference between observations is zero. Try arranging data to have a positive time difference between observations. If working with time series groups, arrange by groups first, then date."
        )

    min_date = frame[date_column].min()
    radians = (
        2 * np.pi * (frame[date_column] - min_date).dt.total_seconds() / scale_factor
    )

    data = {}
    for type_val in ("sin", "cos"):
        for K_val in range(1, max_order + 1):
            for period_val in periods:
                col_name = f"{date_column}_{type_val}_{K_val}_{period_val}"
                data[col_name] = calc_fourier(
                    x=radians,
                    period=period_val,
                    type=type_val,
                    K=K_val,
                )

    return pd.DataFrame(data, index=frame.index)

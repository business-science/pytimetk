import numpy as np
import pandas as pd
import pandas_flavor as pf
import polars as pl
import warnings
from typing import List, Optional, Sequence, Union

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    ensure_row_id_column,
    normalize_engine,
    resolve_polars_group_columns,
    restore_output_type,
)


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_hilbert(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: Union[str, List[str]],
    reduce_memory: bool = True,
    engine: Optional[str] = "auto",
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Apply the Hilbert transform to specified columns of a DataFrame or
    DataFrameGroupBy object.

    Signal Processing: The Hilbert transform is used in various signal processing techniques, including phase and amplitude modulation and demodulation, and in the analysis of signals with time-varying amplitude and frequency.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Input DataFrame or DataFrameGroupBy object with one or more columns of
        real-valued signals.
    value_column : str or list
        List of column names in 'data' to which the Hilbert transform will be
        applied.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : {"auto", "pandas", "polars"}, optional
        Specifies the backend to use for the computation. When "auto" (default)
        the backend is inferred from the input data. Use "pandas" or "polars"
        to force a specific backend.

    Returns
    -------
    df_hilbert : DataFrame
        A new DataFrame with the 2 Hilbert-transformed columns added, 1 for the
        real and 1 for imaginary (original columns are preserved). Matches the
        backend of the input data.

    Notes
    -----
    The Hilbert transform is used in time series analysis primarily for:

    1. Creating Analytic Signals: Forms a complex-valued signal whose
    properties (magnitude and phase) provide valuable insights into the
    original signal's structure.

    2. Determining Instantaneous Phase/Frequency: Offers real-time signal
    characteristics, crucial for non-stationary signals whose properties
    change over time.

    3. Extracting Amplitude Envelope: Helps in identifying signal's
    amplitude variations, useful in various analysis tasks.

    4. Enhancing Signal Analysis: Assists in tasks like demodulation, trend
    analysis, feature extraction for machine learning, and improving
    signal-to-noise ratio, providing a deeper understanding of underlying
    patterns and trends.


    Examples
    --------
    ```{python}
    # Example 1: Using Pandas Engine on a pandas groupby object
    import pytimetk as tk
    import pandas as pd

    df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])


    df_hilbert = (
        df
            .groupby('id')
            .augment_hilbert(
                date_column = 'Date',
                value_column = ['Weekly_Sales'],
                engine = 'pandas'
            )
    )

    df_hilbert.head()
    ```

    ```{python}
    # Example 2: Using the polars accessor on a grouped table
    import pytimetk as tk
    import polars as pl


    df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
    df_hilbert = (
        pl.from_pandas(df)
            .group_by('id')
            .tk.augment_hilbert(
                date_column = 'Date',
                value_column = ['Weekly_Sales'],
            )
    )

    df_hilbert.head()
    ```

    ```{python}
    # Example 3: Using the polars accessor on a DataFrame
    import pytimetk as tk
    import polars as pl


    df = tk.load_dataset('taylor_30_min', parse_dates=['date'])
    df_hilbert = (
        pl.from_pandas(df)
            .tk.augment_hilbert(
                date_column = 'date',
                value_column = ['value'],
            )
    )

    df_hilbert.head()
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    conversion: FrameConversion = convert_to_engine(data, "pandas")
    prepared_data = conversion.data

    if reduce_memory and engine_resolved == "pandas":
        prepared_data = reduce_memory_usage(prepared_data)
    elif reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    prepared_data, idx_unsorted = sort_dataframe(
        prepared_data, date_column, keep_grouped_df=True
    )

    value_columns: List[str] = (
        [value_column] if isinstance(value_column, str) else list(value_column)
    )

    result = _augment_hilbert_pandas(
        prepared_data,
        date_column,
        value_columns,
    )

    if not isinstance(result, pd.DataFrame):
        raise TypeError("Hilbert augmentation must return a pandas DataFrame.")

    result.index = idx_unsorted
    result = result.sort_index()

    if reduce_memory and engine_resolved == "pandas":
        result = reduce_memory_usage(result)

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_hilbert_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_columns: List[str],
):
    # Type checks
    # if not isinstance(data, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
    #     raise TypeError("Input must be a pandas DataFrame or DataFrameGroupBy object")
    if not isinstance(value_columns, list) or not all(
        isinstance(col, str) for col in value_columns
    ):
        raise TypeError("value_column must be a list of strings")

    # If 'data' is a DataFrame, convert it to a groupby object with a dummy group
    if isinstance(data, pd.DataFrame):
        if any(col not in data.columns for col in value_columns):
            missing_cols = [col for col in value_columns if col not in data.columns]
            raise KeyError(f"Columns {missing_cols} do not exist in the DataFrame")
        data = data.sort_values(by=date_column)
        data = data.groupby(np.zeros(len(data)))

    # Function to apply Hilbert transform to each group
    def apply_hilbert(group):
        for col in value_columns:
            # Ensure the column exists in the DataFrame
            if col not in group.columns:
                raise KeyError(f"Column '{col}' does not exist in the group")

            # Get the signal from the DataFrame
            signal = group[col].values

            # Compute the FFT of the signal
            N = signal.size
            Xf = np.fft.fft(signal)

            # Create a zero-phase version of the signal with the negative
            # frequencies zeroed out
            h = np.zeros(N)
            if N % 2 == 0:
                h[0] = h[N // 2] = 1
                h[1 : N // 2] = 2
            else:
                h[0] = 1
                h[1 : (N + 1) // 2] = 2

            Xf *= h

            # Perform the inverse FFT
            x_analytic = np.fft.ifft(Xf)

            # Update the DataFrame
            group[f"{col}_hilbert_real"] = np.real(x_analytic)
            group[f"{col}_hilbert_imag"] = np.imag(x_analytic)
        return group

    # Apply the Hilbert transform to each group and concatenate the results
    df_hilbert = pd.concat(
        (apply_hilbert(group) for _, group in data), ignore_index=True
    )

    return df_hilbert


def _augment_hilbert_polars(
    data: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
    value_columns: List[str],
    group_columns: Optional[Sequence[str]],
    row_id_column: Optional[str],
) -> pl.DataFrame:
    resolved_groups = resolve_polars_group_columns(data, group_columns)
    frame = data.df if isinstance(data, pl.dataframe.group_by.GroupBy) else data

    frame_with_id, row_col, generated = ensure_row_id_column(frame, row_id_column)

    sort_keys = list(resolved_groups)
    sort_keys.append(date_column)
    sorted_frame = frame_with_id.sort(sort_keys)

    def apply_hilbert(pl_group: pl.DataFrame) -> pl.DataFrame:
        new_series = []
        for col in value_columns:
            signal = pl_group[col].to_numpy()
            N = signal.size
            Xf = np.fft.fft(signal)
            h = np.zeros(N)
            if N % 2 == 0:
                h[0] = h[N // 2] = 1
                h[1 : N // 2] = 2
            else:
                h[0] = 1
                h[1 : (N + 1) // 2] = 2
            Xf *= h
            x_analytic = np.fft.ifft(Xf)
            new_series.extend(
                [
                    pl.Series(f"{col}_hilbert_real", np.real(x_analytic)),
                    pl.Series(f"{col}_hilbert_imag", np.imag(x_analytic)),
                ]
            )
        return pl_group.with_columns(new_series)

    if resolved_groups:
        transformed = (
            sorted_frame.group_by(resolved_groups, maintain_order=True)
            .map_groups(apply_hilbert)
            .sort(sort_keys)
        )
    else:
        transformed = apply_hilbert(sorted_frame)

    transformed = transformed.sort(row_col)

    if generated:
        transformed = transformed.drop(row_col)

    return transformed

import numpy as np
import pandas as pd
import pandas_flavor as pf
import polars as pl
from typing import Union, List
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.polars_helpers import pandas_to_polars_frequency, pandas_to_polars_aggregation_mapping
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_hilbert(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str, 
    value_column: Union[str, List[str]], 
    reduce_memory: bool = True,
    engine: str = 'pandas'):

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
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        summarizing the data. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
        for summarizing the data. This can be faster than using "pandas" for 
        large datasets. 
    
    Returns
    -------
    df_hilbert : pd.DataFrame
        A new DataFrame with the 2 Hilbert-transformed columns added, 1 for the 
        real and 1 for imaginary (original columns are preserved).

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
    # Example 2: Using Polars Engine on a pandas groupby object
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
    df_hilbert = (
        df
            .groupby('id')
            .augment_hilbert(
                date_column = 'Date',
                value_column = ['Weekly_Sales'],
                engine = 'polars'
            )
    )

    df_hilbert.head()
    ```
    
    ```{python}
    # Example 3: Using Polars Engine on a pandas dataframe
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset('taylor_30_min', parse_dates=['date'])
    df_hilbert = (
        df
            .augment_hilbert(
                date_column = 'date',
                value_column = ['value'],
                engine = 'polars'
            )
    )

    df_hilbert.head()
    ```
    
    ```{python}
    # Example 4: Using Polars Engine on a groupby object
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset('taylor_30_min', parse_dates=['date'])
    df_hilbert_pd = (
        df
            .augment_hilbert(
                date_column = 'date',
                value_column = ['value'],
                engine = 'pandas'
            )
    )

    df_hilbert.head()
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df=True)
    
    if engine == 'pandas':
        ret = _augment_hilbert_pandas(data, date_column, value_column)
    elif engine == 'polars':
        ret = _augment_hilbert_polars(data, date_column, value_column)
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    return ret.sort_index()
    
    
# Monkey-patch the method to the DataFrameGroupBy class
pd.core.groupby.DataFrameGroupBy.augment_hilbert = augment_hilbert


def _augment_hilbert_pandas(                            
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
                            ):
    # Type checks
    # if not isinstance(data, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
    #     raise TypeError("Input must be a pandas DataFrame or DataFrameGroupBy object")
    if not isinstance(value_column, list) or not all(isinstance(col, str) for col in value_column):
        raise TypeError("value_column must be a list of strings")

    # If 'data' is a DataFrame, convert it to a groupby object with a dummy group
    if isinstance(data, pd.DataFrame):
        if any(col not in data.columns for col in value_column):
            missing_cols = [col for col in value_column if col not in data.columns]
            raise KeyError(f"Columns {missing_cols} do not exist in the DataFrame")
        data = data.sort_values(by=date_column)
        data = data.groupby(np.zeros(len(data)))

    
    # Function to apply Hilbert transform to each group
    def apply_hilbert(group):
        for col in value_column:
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
                h[1:N // 2] = 2
            else:
                h[0] = 1
                h[1:(N + 1) // 2] = 2

            Xf *= h
            
            # Perform the inverse FFT
            x_analytic = np.fft.ifft(Xf)
            
            # Update the DataFrame
            group[f'{col}_hilbert_real'] = np.real(x_analytic)
            group[f'{col}_hilbert_imag'] = np.imag(x_analytic)
        return group

    # Apply the Hilbert transform to each group and concatenate the results
    df_hilbert = pd.concat((apply_hilbert(group) for _, group in data), ignore_index=True)

    return df_hilbert





def _augment_hilbert_polars(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
) -> pd.DataFrame:
    if isinstance(value_column, str):
        value_column = [value_column]

    df_pl = pl.from_pandas(data.obj.copy() if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy) else data.copy())

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        groups = data.grouper.names if isinstance(data.grouper.names, list) else [data.grouper.names]
        
        # Define output schema
        original_schema = df_pl.schema
        new_columns = {
            f"{col}_hilbert_real": pl.Float64 for col in value_column
        } | {
            f"{col}_hilbert_imag": pl.Float64 for col in value_column
        }
        output_schema = {**original_schema, **new_columns}

        def apply_hilbert(pl_group: pl.DataFrame) -> pl.DataFrame:
            exprs = []
            for col in value_column:
                signal = pl_group[col].to_numpy()
                N = signal.size
                Xf = np.fft.fft(signal)
                h = np.zeros(N)
                if N % 2 == 0:
                    h[0] = h[N // 2] = 1
                    h[1:N // 2] = 2
                else:
                    h[0] = 1
                    h[1:(N + 1) // 2] = 2
                Xf *= h
                x_analytic = np.fft.ifft(Xf)
                exprs.extend([
                    pl.Series(f'{col}_hilbert_real', np.real(x_analytic)),
                    pl.Series(f'{col}_hilbert_imag', np.imag(x_analytic))
                ])
            return pl_group.with_columns(exprs)

        data = (
            df_pl.lazy()
                .sort([*groups, date_column])
                .group_by(groups, maintain_order=True)
                .map_groups(apply_hilbert, schema=output_schema)
                .collect(streaming=True)
        )
    else:
        data = df_pl.lazy().sort(date_column)
        exprs = []
        for col in value_column:
            signal = data.select(pl.col(col)).collect()[col].to_numpy()
            N = signal.size
            Xf = np.fft.fft(signal)
            h = np.zeros(N)
            if N % 2 == 0:
                h[0] = h[N // 2] = 1
                h[1:N // 2] = 2
            else:
                h[0] = 1
                h[1:(N + 1) // 2] = 2
            Xf *= h
            x_analytic = np.fft.ifft(Xf)
            exprs.extend([
                pl.Series(f'{col}_hilbert_real', np.real(x_analytic)),
                pl.Series(f'{col}_hilbert_imag', np.imag(x_analytic))
            ])
        data = data.with_columns(exprs).collect(streaming=True)

    return data.to_pandas()



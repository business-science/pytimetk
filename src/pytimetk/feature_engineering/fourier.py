import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Tuple
from typing import Union, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from pytimetk.core.ts_summary import ts_summary
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe


@pf.register_dataframe_method
def augment_fourier(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    max_order: int = 1,
    reduce_memory: bool = True,
    engine: str = 'pandas'
) -> pd.DataFrame:
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
    
    """

    if not engine in ['pandas', 'polars']: 
        raise ValueError(f"Supported engines are 'pandas' or 'polars'. Found {engine}. Please select an authorized engine.")
    
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    
    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):  
        data = data.obj.copy().reset_index(drop=True)  
        
    # Reduce memory usage
    if reduce_memory:
        data = reduce_memory_usage(data)

    if engine == "pandas":
        ret = _augment_fourier_pandas(data, date_column, periods, max_order)
    else:
        ret = _augment_fourier_polars(data, date_column, periods, max_order)
        
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_fourier = augment_fourier

def calc_fourier(x, period, type: str, K = 1): 
    term = K / period
    return np.sin(2 * np.pi * term * x) if type == "sin" else np.cos(2 * np.pi * term * x)


def date_to_seq_scale_factor(data: pd.DataFrame, date_var: str, engine: str = "pandas") -> pd.DataFrame:
    return ts_summary(data, date_column=date_var, engine=engine)['diff_median']

def _augment_fourier_pandas(
    data: pd.DataFrame, 
    date_column: str,
    periods: Union[int, Tuple[int, int], List[int]],
    max_order: int
) -> pd.DataFrame:

    df = data.copy()

    scale_factor = date_to_seq_scale_factor(df, date_column).iloc[0].total_seconds()
    if scale_factor == 0:
        raise ValueError("Time difference between observations is zero. Try arranging data to have a positive time difference between observations. If working with time series groups, arrange by groups first, then date.")
    
    # Calculate radians for the date values
    min_date = df[date_column].min()
    df['radians'] = 2 * np.pi * (df[date_column] - min_date).dt.total_seconds() / scale_factor
    
    type_vals = ["sin", "cos"]
    K_vals = range(1, max_order + 1)

    new_cols = [
        f'{date_column}_{type_val}_{K_val}_{period_val}'
        for type_val in type_vals
        for K_val in K_vals
        for period_val in periods
    ]

    df_new = np.array([
        calc_fourier(x=df['radians'], period=period_val, type=type_val, K=K_val)
        for type_val in type_vals
        for K_val in K_vals
        for period_val in periods
    ]).reshape(-1, len(df)).T

    df[new_cols] = df_new
    # Drop the temporary 'radians' column
    return df.drop(columns=['radians'])

def _augment_fourier_polars(
        data: pd.DataFrame, 
        date_column: str,
        periods: Union[int, Tuple[int, int], List[int]],
        max_order: int
    ) -> pd.DataFrame:
    """ Takes pandas objects as inputs and converts them into polars object internally.
    """

    # Convert to polars
    df = pl.from_pandas(data) 
    # .sort(by=[date_column], descending=False, nulls_last=True) 

    # Compute scale factor
    scale_factor = date_to_seq_scale_factor(data, date_column, engine="polars")[0].total_seconds()
    if scale_factor == 0:
        raise ValueError("Time difference between observations is zero. Try arranging data to have a positive time difference between observations. If working with time series groups, arrange by groups first, then date.")
    
    # Convert dates to numeric representation
    min_date = df[date_column].min()
    df = df.with_columns((2 * np.pi * (df[date_column] - min_date).dt.total_seconds() / scale_factor).rename('radians'))
    
    # Compute Fourier series
    for type_val in ("sin", "cos"):
        for K_val in range(1, max_order + 1):
            for period_val in periods:
                df = df.with_columns(calc_fourier(x = df['radians'], period = period_val, type = type_val, K = K_val).rename(f'{date_column}_{type_val}_{K_val}_{period_val}'))
        
    return df.to_pandas().drop(columns=['radians'])




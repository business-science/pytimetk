import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Tuple
from typing import Union, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from pytimetk.core.ts_summary import ts_summary, ts_summary_polars
from pytimetk.utils.memory_helpers import reduce_memory_usage


def calc_fourier(x, period, type: str, K = 1): 
    term = K / period
    return np.sin(2 * np.pi * term * x) if type == "sin" else np.cos(2 * np.pi * term * x)


def date_to_seq_scale_factor(data: pd.DataFrame, date_var: str) -> pd.DataFrame:
    return ts_summary(data, date_column=date_var)['diff_median']


def date_to_seq_scale_factor_polars(data: pl.DataFrame, date_var: str) -> pl.DataFrame:
    return ts_summary_polars(data, date_column=date_var)['diff_median']

def _augment_fourier_pandas(
    data: pd.DataFrame, 
    date_column: str,
    periods: Union[int, Tuple[int, int], List[int]],
    max_order: int
) -> pd.DataFrame:

    df = data.copy()
    df.sort_values(by=[date_column], inplace=True)

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
    df = pl.from_pandas(data) \
        .sort(by=[date_column], descending=False, nulls_last=True) 

    # Compute scale factor
    scale_factor = date_to_seq_scale_factor_polars(df, date_column)[0].total_seconds()
    if scale_factor == 0:
        raise ValueError("Time difference between observations is zero. Try arranging data to have a positive time difference between observations. If working with time series groups, arrange by groups first, then date.")
    
    # Convert dates to numeric representation
    min_date = df[date_column].min()
    df = df.with_columns((2 * np.pi * (df[date_column] - min_date).dt.seconds() / scale_factor).rename('radians'))
    
    # Compute Fourier series
    for type_val in ("sin", "cos"):
        for K_val in range(1, max_order + 1):
            for period_val in periods:
                df = df.with_columns(calc_fourier(x = df['radians'], period = period_val, type = type_val, K = K_val).rename(f'{date_column}_{type_val}_{K_val}_{period_val}'))
        
    return df.drop(columns=['radians']).to_pandas()


@pf.register_dataframe_method
def augment_fourier_v2(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    max_order: int = 1,
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
    
    # Add Fourier transforms for a single column
    fourier_df = (
        df
            .query("id == 'D10'")
            .augment_fourier_v2(
                date_column='date',
                periods=[1, 7],
                max_order=1
            )
    )
    fourier_df.head()
    
    fourier_df.plot_timeseries("date", "date_sin_1_7", x_axis_date_labels = "%B %d, %Y",)
    ```

    """

    if not engine in ['pandas', 'polars']: 
        raise ValueError(f"Supported engines are 'pandas' or 'polars'. Found {engine}. Please select an authorized engine.")
    
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    if isinstance(periods, int):
        periods = [periods]
    elif isinstance(periods, tuple):
        periods = list(range(periods[0], periods[1] + 1))
    elif not isinstance(periods, list):
        raise TypeError(f"Invalid periods specification: type: {type(periods)}. Please use int, tuple, or list.")

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):  
        data = data.apply(lambda x: x).reset_index(drop=True)  

    if engine == "pandas":
        return _augment_fourier_pandas(data, date_column, periods, max_order)
    else:
        return _augment_fourier_polars(data, date_column, periods, max_order)



@pf.register_dataframe_method
def augment_fourier(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    num_periods: int = 1,
    max_order: int = 1
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
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you want to apply Fourier transforms to. It can be either a single column name (string) or a list of column names.
    num_periods : int, optional
        The `num_periods` parameter specifies the number of periods for the Fourier series. Default is 1.
    max_order : int, optional
        The `max_order` parameter specifies the maximum Fourier order to calculate. Default is 1.

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
    
    # Add Fourier transforms for a single column
    fourier_df = (
        df
            .groupby('id')
            .augment_fourier(
                date_column='date',
                value_column='value',
                num_periods=7,
                max_order=1
            )
    )
    fourier_df.head()
    ```

    """

    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)

    if isinstance(value_column, str):
        value_column = [value_column]

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with Fourier transforms
    if isinstance(data, pd.DataFrame):

        # Calculate radians for the date values
        min_date = data[date_column].min()
        data['radians'] = 2 * np.pi * (data[date_column] - min_date).dt.total_seconds() / (24 * 3600)

        df = reduce_memory_usage(data.copy())

        df.sort_values(by=[date_column], inplace=True)

        for col in value_column:
            for order in range(1, max_order + 1):
                for period in range(1, num_periods + 1):
                    freq = 2 * np.pi * period
                    df[f'{col}_fourier_{order}_{period}'] = (
                        np.cos(freq * df['radians']) if order % 2 == 0 else np.sin(freq * df['radians'])
                    )

    # GROUPED EXTENSION - If data is a GroupBy object, add Fourier transforms by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj
        
        # Calculate radians for the date values
        min_date = data[date_column].min()
        data['radians'] = 2 * np.pi * (data[date_column] - min_date).dt.total_seconds() / (24 * 3600)

        df = data.copy()

        df.sort_values(by=[*group_names, date_column], inplace=True)

        for col in value_column:
            for order in range(1, max_order + 1):
                for period in range(1, num_periods + 1):
                    freq = 2 * np.pi * period
                    df[f'{col}_fourier_{order}_{period}'] = (
                        np.cos(freq * df['radians']) if order % 2 == 0 else np.sin(freq * df['radians'])
                    )

    # Drop the temporary 'radians' column
    df.drop(columns=['radians'], inplace=True)

    return reduce_memory_usage(df)

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_fourier = augment_fourier
pd.core.groupby.generic.DataFrameGroupBy.augment_fourier_v2 = augment_fourier_v2

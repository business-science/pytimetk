import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import Callable, Dict, Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column
from pytimetk.utils.pandas_helpers import flatten_multiindex_column_names
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.dataframe_ops import (
    convert_to_engine,
    normalize_engine,
    restore_output_type,
    FrameConversion,
    resolve_pandas_groupby_frame,
)


@pf.register_groupby_method
@pf.register_dataframe_method
def apply_by_time(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    freq: str = "D",
    wide_format: bool = False,
    fillna: int = 0,
    reduce_memory: bool = False,
    engine: str = "pandas",
    **named_funcs,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Apply for time series.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        Tabular data on which the operation is performed. Supports both pandas
        and polars DataFrames / GroupBy objects.
    date_column : str
        The name of the column in the DataFrame that contains the dates.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the data should be
        resampled. It accepts a string representing a time frequency, such as "D"
        for daily, "W" for weekly, "M" for monthly, etc. The default value is "D",
        which means the data will be resampled on a daily basis. Some common
        frequency aliases include:

        - S: secondly frequency
        - min: minute frequency
        - H: hourly frequency
        - D: daily frequency
        - W: weekly frequency
        - M: month end frequency
        - MS: month start frequency
        - Q: quarter end frequency
        - QS: quarter start frequency
        - Y: year end frequency
        - YS: year start frequency

    wide_format : bool, optional
        The `wide_format` parameter is a boolean flag that determines whether the
        output should be in wide format or not. If `wide_format` is set to `True`,
        the output will have a multi-index column structure, where the first level
        represents the original columns and the second level represents the group
        names.
    fillna : int, optional
        The `fillna` parameter is used to specify the value that will be used to
        fill missing values in the resulting DataFrame. By default, it is set to 0.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : {"pandas", "polars", "cudf", "auto"}, optional
        Execution engine. ``"pandas"`` (default) performs the computation using pandas.
        When "polars" the data is converted to pandas for evaluation and converted
        back to polars on return. ``"cudf"`` inputs currently reuse the pandas
        implementation. ``"auto"`` infers the engine from the input data.
    **named_funcs
        The `**named_funcs` parameter is used to specify one or more custom
        aggregation functions to apply to the data. It accepts named functions
        in the format:

        ``` python
            name = lambda df: df['column1'].corr(df['column2']])
        ```

        Where `name` is the name of the function and `df` is the DataFrame that will
        be passed to the function. The function must return a single value.



    Returns
    -------
    DataFrame
        The resulting data after applying the functions. The concrete type matches
        the engine used to process the data.

    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd

    df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])

    df.glimpse()
    ```

    ```{python}
    # Apply by time with a DataFrame object
    # Allows access to multiple columns at once
    (
        df[['order_date', 'price', 'quantity']]
            .apply_by_time(

                # Named apply functions
                price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),

                # Parameters
                date_column  = 'order_date',
                freq         = "MS",

            )
    )
    ```

    ```{python}
    # Apply by time with a GroupBy object
    (
        df[['category_1', 'order_date', 'price', 'quantity']]
            .groupby('category_1')
            .apply_by_time(

                # Named functions
                price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),

                # Parameters
                date_column  = 'order_date',
                freq         = "MS",

            )
    )
    ```

    ```{python}
    # Return complex objects
    (
        df[['order_date', 'price', 'quantity']]
            .apply_by_time(

                # Named apply functions
                complex_object = lambda df: [df],

                # Parameters
                date_column  = 'order_date',
                freq         = "MS",

            )
    )
    ```

    ```{python}
    # Polars DataFrame using the tk accessor
    import polars as pl


    pl_df = pl.from_pandas(df[['order_date', 'price', 'quantity']])

    (
        pl_df
            .tk.apply_by_time(
                date_column='order_date',
                freq='MS',
                total = lambda frame: (frame['price'] * frame['quantity']).sum(),
            )
    )
    ```
    """

    # Run common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    engine_resolved = normalize_engine(engine, data)

    if engine_resolved == "cudf":
        warnings.warn(
            "apply_by_time currently falls back to the pandas implementation when used with cudf data.",
            RuntimeWarning,
            stacklevel=2,
        )
        engine_resolved = "pandas"

    if engine_resolved == "pandas":
        conversion = convert_to_engine(data, "pandas")
        prepared = conversion.data
        result = _apply_by_time_pandas(
            prepared,
            date_column=date_column,
            freq=freq,
            wide_format=wide_format,
            fillna=fillna,
            reduce_memory=reduce_memory,
            named_funcs=named_funcs,
        )
        return restore_output_type(result, conversion)

    # polars engine - operate via pandas fallback
    conversion = convert_to_engine(data, "polars")
    prepared = conversion.data

    pandas_prepared = _polars_to_pandas(prepared, date_column)
    result_pd = _apply_by_time_pandas(
        pandas_prepared,
        date_column=date_column,
        freq=freq,
        wide_format=wide_format,
        fillna=fillna,
        reduce_memory=reduce_memory,
        named_funcs=named_funcs,
    )

    return pl.from_pandas(result_pd)


def _apply_by_time_pandas(
    prepared: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    *,
    date_column: str,
    freq: str,
    wide_format: bool,
    fillna: int,
    reduce_memory: bool,
    named_funcs: Dict[str, callable],
) -> pd.DataFrame:
    data = prepared

    if reduce_memory:
        if isinstance(data, pd.DataFrame):
            data = reduce_memory_usage(data)
        else:
            data = resolve_pandas_groupby_frame(data).copy()
            data = reduce_memory_usage(data)
            data = data.groupby(prepared.grouper.names)

    group_names = None
    if isinstance(data, pd.DataFrame):
        data = data.set_index(date_column)
    else:
        group_names = list(data.grouper.names)
        if date_column not in group_names:
            data = resolve_pandas_groupby_frame(data).set_index(date_column).groupby(group_names)

    grouped = data.resample(rule=freq, kind="timestamp")

    def custom_agg(group):
        agg_values = {}
        for name, func in named_funcs.items():
            agg_values[name] = func(group)
        return pd.Series(agg_values)

    result = grouped.apply(custom_agg)

    if wide_format and group_names is not None:
        result = result.unstack(group_names)

    result = result.fillna(fillna)
    result = flatten_multiindex_column_names(result)
    result.reset_index(inplace=True)

    if reduce_memory:
        result = reduce_memory_usage(result)

    return result


def _polars_to_pandas(
    prepared: Union[pl.DataFrame, pl.dataframe.group_by.GroupBy],
    date_column: str,
) -> Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
    if isinstance(prepared, pl.DataFrame):
        return prepared.to_pandas()

    raw = prepared.by
    if isinstance(raw, list):
        group_cols = []
        for item in raw:
            if isinstance(item, tuple):
                group_cols.extend(item)
            else:
                group_cols.append(item)
    elif isinstance(raw, tuple):
        group_cols = list(raw)
    else:
        group_cols = [raw]

    group_cols = [str(col) for col in group_cols]
    pandas_df = prepared.df.to_pandas()
    if date_column not in pandas_df.columns:
        raise KeyError(f"{date_column} not found in DataFrame")
    return pandas_df.groupby(group_cols, sort=False)

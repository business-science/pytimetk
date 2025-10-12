import pandas as pd
import polars as pl
import pandas_flavor as pf
import warnings

from typing import Optional, Union, List

from pytimetk.utils.checks import (
    check_dataframe_or_groupby,
    check_date_column,
    check_value_column,
)
from pytimetk.utils.dataframe_ops import (
    FrameConversion,
    convert_to_engine,
    normalize_engine,
    restore_output_type,
)
from pytimetk.utils.memory_helpers import reduce_memory_usage


@pf.register_groupby_method
@pf.register_dataframe_method
def augment_ewm(
    data: Union[
        pd.DataFrame,
        pd.core.groupby.generic.DataFrameGroupBy,
        pl.DataFrame,
        pl.dataframe.group_by.GroupBy,
    ],
    date_column: str,
    value_column: Union[str, list],
    window_func: Union[str, list] = "mean",
    alpha: float = None,
    **kwargs,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Add Exponential Weighted Moving (EWM) window functions to a DataFrame or
    GroupBy object.

    The `augment_ewm` function applies Exponential Weighted Moving (EWM) window
    functions to specified value columns of a DataFrame and adds the results as
    new columns.

    Parameters
    ----------
    data : DataFrame or GroupBy (pandas or polars)
        The input data to augment. Grouped inputs are processed per group before
        the EWM columns are appended.
    date_column : str
        The name of the column containing date information in the input
        DataFrame or GroupBy object.
    value_column : Union[str, list]
        The `value_column` parameter is used to specify the column(s) on which
        the Exponential Weighted Moving (EWM) calculations will be performed. It
        can be either a string or a list of strings, representing the name(s) of
        the column(s) in the input DataFrame or GroupBy
    window_func : Union[str, list], optional
        The `window_func` parameter is used to specify the Exponential Weighted
        Moving (EWM) window function(s) to apply. It can be a string or a list
        of strings. The possible values are:

        - 'mean': Calculate the exponentially weighted mean.
        - 'median': Calculate the exponentially weighted median.
        - 'std': Calculate the exponentially weighted standard deviation.
        - 'var': Calculate the exponentially weighted variance.

    alpha : float
        The `alpha` parameter is a float that represents the smoothing factor
        for the Exponential Weighted Moving (EWM) window function. It controls
        the rate at which the weights decrease exponentially as the data points
        move further away from the current point.
    engine : {"auto", "pandas", "polars"}, optional
        Execution engine. ``"auto"`` (default) infers the backend from the input
        data while allowing explicit overrides.
    reduce_memory : bool, optional
        Attempt to reduce memory usage before/after computation when operating
        on pandas data. If a polars input is supplied a warning is emitted and
        no conversion occurs.
    **kwargs:
        Additional arguments that are directly passed to the pandas EWM method.
        For more details, refer to the "Notes" section below.

    Returns
    -------
    DataFrame
        The function `augment_ewm` returns a DataFrame augmented with the
        results of the Exponential Weighted Moving (EWM) calculations.

    Notes
    ------
    Any additional arguments provided through **kwargs are directly passed
    to the pandas EWM method. These arguments can include parameters like
    'com', 'span', 'halflife', 'ignore_na', 'adjust' and more.

    For a comprehensive list and detailed description of these parameters:

    - Refer to the official pandas documentation:
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html

    - Or, within an interactive Python environment, use:
        `?pandas.DataFrame.ewm` to display the method's docstring.

    Examples
    --------
    ```{python}
    import pandas as pd
    import polars as pl
    import pytimetk as tk


    df = tk.load_dataset("m4_daily", parse_dates = ['date'])

    # Pandas example (engine inferred)
    ewm_df = (
        df
            .groupby('id')
            .augment_ewm(
                date_column = 'date',
                value_column = 'value',
                window_func = [
                    'mean',
                    'std',
                ],
                alpha = 0.1,
            )
    )

    # Polars example using the tk accessor
    ewm_pl = (
        pl.from_pandas(df)
        .group_by('id')
        .tk.augment_ewm(
            date_column='date',
            value_column='value',
            window_func='mean',
            alpha=0.1,
        )
    )
    ```
    """
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)

    engine_resolved = normalize_engine(engine, data)

    if reduce_memory and engine_resolved == "polars":
        warnings.warn(
            "`reduce_memory=True` is only supported for pandas data.",
            RuntimeWarning,
            stacklevel=2,
        )

    conversion: FrameConversion = convert_to_engine(data, "pandas")
    prepared_data = conversion.data

    result = _augment_ewm_pandas(
        data=prepared_data,
        date_column=date_column,
        value_column=value_column,
        window_func=window_func,
        alpha=alpha,
        reduce_memory=reduce_memory,
        **kwargs,
    )

    restored = restore_output_type(result, conversion)

    if isinstance(restored, pd.DataFrame):
        return restored.sort_index()

    return restored


def _augment_ewm_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    window_func: Union[str, List[str]],
    alpha: float,
    reduce_memory: bool,
    **kwargs,
) -> pd.DataFrame:
    value_columns = (
        [value_column] if isinstance(value_column, str) else list(value_column)
    )
    window_funcs = [window_func] if isinstance(window_func, str) else list(window_func)

    data_copy = data.copy() if isinstance(data, pd.DataFrame) else data.obj.copy()

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        grouped = data_copy.sort_values(by=[*group_names, date_column]).groupby(
            group_names
        )
    else:
        group_names = None
        grouped = [([], data_copy.sort_values(by=[date_column]))]

    def determine_decay_parameter(alpha, **kwargs):
        if alpha is not None:
            return "alpha", alpha
        for param in ["com", "span", "halflife"]:
            if param in kwargs:
                return param, kwargs[param]
        return None, None

    decay_param, value = determine_decay_parameter(alpha, **kwargs)

    if decay_param is None:
        raise ValueError(
            "No valid decay parameter provided. Specify 'alpha' through function arguments, or one of 'com', 'span', or 'halflife' through **kwargs."
        )

    result_dfs = []
    for _, group_df in grouped:
        for col in value_columns:
            for func in window_funcs:
                result_col_name = f"{col}_ewm_{func}_{decay_param}_{value}"
                ewm_obj = group_df[col].ewm(alpha=alpha, **kwargs)
                result_series = _apply_ewm_function(ewm_obj, func)
                group_df[result_col_name] = result_series
        result_dfs.append(group_df)

    result = pd.concat(result_dfs)
    if group_names is not None:
        result = result.sort_index(level=group_names)
    else:
        result = result.sort_index()

    if reduce_memory:
        result = reduce_memory_usage(result)

    return result


def _apply_ewm_function(ewm_obj, func: Union[str, callable]) -> pd.Series:
    if isinstance(func, str):
        method = getattr(ewm_obj, func, None)
        if method is None:
            raise ValueError(f"Invalid function name: {func}")
        return method()
    raise TypeError(f"Invalid function type: {type(func)}")

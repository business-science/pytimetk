import pandas as pd
import numpy as np
import pandas_flavor as pf

@pf.register_dataframe_method
def augment_lags(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy, 
    date_column: str,
    value_column: str or list, 
    lags: int or tuple or list = 1
) -> pd.DataFrame:
    """
    Adds lags to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_lags` function takes a Pandas DataFrame or GroupBy object, a date column, a value column or list of value columns, and a lag or list of lags, and adds lagged versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to add lagged columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to sort the data before adding the lagged values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you want to add lagged values for. It can be either a single column name (string) or a list of column names.
    lags : int or tuple or list, optional
        The `lags` parameter is an integer, tuple, or list that specifies the number of lagged values to add to the DataFrame. If it is an integer, the function will add that number of lagged values for each column specified in the `value_column` parameter. If it is a tuple, it will generate lags from the first to the second value (inclusive). If it is a list, it will generate lags based on the values in the list.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with lagged columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    
    # Add a lagged value of 2 for each grouped time series
    lagged_df = (
        df 
            .groupby('id')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=2
            )
    )

    # Add 7 lagged values for a single time series
    lagged_df_single = (
        df 
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=(1, 7)
            )
    )

    # Add 2 lagged values, 2 and 4, for a single time series
    lagged_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_lags(
                date_column='date',
                value_column='value',
                lags=[2, 4]
            )
    )
    ```

    """

    # Check if data is a Pandas DataFrame or GroupBy object
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame or GroupBy object.")

    if isinstance(value_column, str):
        value_column = [value_column]

    if isinstance(lags, int):
        lags = [lags]
    elif isinstance(lags, tuple):
        lags = list(range(lags[0], lags[1] + 1))
    elif not isinstance(lags, list):
        raise ValueError("Invalid lags specification. Please use int, tuple, or list.")

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    if isinstance(data, pd.DataFrame):

        df = data.copy()

        df.sort_values(by=[date_column], inplace=True)

        for col in value_column:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # GROUPED EXTENSION - If data is a GroupBy object, add lags by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        df.sort_values(by=[*group_names, date_column], inplace=True)

        for col in value_column:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby(group_names)[col].shift(lag)

    return df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_lags = augment_lags

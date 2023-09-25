import pandas as pd
import numpy as np
import pandas_flavor as pf

@pf.register_dataframe_method
def augment_leads(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy, 
    date_column: str,
    value_column: str or list, 
    leads: int or tuple or list = 1
) -> pd.DataFrame:
    """
    Adds leads to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_leads` function takes a Pandas DataFrame or GroupBy object, a date column, a value column or list of value columns, and a lead or list of leads, and adds leaded versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to add leaded columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to sort the data before adding the leaded values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you want to add leaded values for. It can be either a single column name (string) or a list of column names.
    leads : int or tuple or list, optional
        The `leads` parameter is an integer, tuple, or list that specifies the number of leaded values to add to the DataFrame. If it is an integer, the function will add that number of leaded values for each column specified in the `value_column` parameter. If it is a tuple, it will generate leads from the first to the second value (inclusive). If it is a list, it will generate leads based on the values in the list.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with leaded columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    
    # Add a leaded value of 2 for each grouped time series
    leaded_df = (
        df 
            .groupby('id')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=2
            )
    )

    # Add 7 leaded values for a single time series
    leaded_df_single = (
        df 
            .query('id == "D10"')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=(1, 7)
            )
    )

    # Add 2 leaded values, 2 and 4, for a single time series
    leaded_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=[2, 4]
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

    if isinstance(leads, int):
        leads = [leads]
    elif isinstance(leads, tuple):
        leads = list(range(leads[0], leads[1] + 1))
    elif not isinstance(leads, list):
        raise ValueError("Invalid leads specification. Please use int, tuple, or list.")

    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    if isinstance(data, pd.DataFrame):

        df = data.copy()

        df.sort_values(by=[date_column], inplace=True)

        for col in value_column:
            for lead in leads:
                df[f'{col}_lead_{lead}'] = df[col].shift(-lead)

    # GROUPED EXTENSION - If data is a GroupBy object, add leads by group
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj

        df = data.copy()

        df.sort_values(by=[*group_names, date_column], inplace=True)

        for col in value_column:
            for lead in leads:
                df[f'{col}_lead_{lead}'] = df.groupby(group_names)[col].shift(-lead)

    return df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_leads = augment_leads

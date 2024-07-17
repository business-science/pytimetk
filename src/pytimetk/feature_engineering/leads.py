import pandas as pd
import polars as pl
import numpy as np
import pandas_flavor as pf
from typing import Union, List, Tuple

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe

@pf.register_dataframe_method
def augment_leads(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    leads: Union[int, Tuple[int, int], List[int]] = 1,
    reduce_memory: bool = False,
    engine: str = 'pandas',
) -> pd.DataFrame:
    """
    Adds leads to a Pandas DataFrame or DataFrameGroupBy object.

    The `augment_leads` function takes a Pandas DataFrame or GroupBy object, a 
    date column, a value column or list of value columns, and a lag or list of 
    lags, and adds lagged versions of the value columns to the DataFrame.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to add lagged columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to sort the data before adding the lagged values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you 
        want to add lagged values for. It can be either a single column name 
        (string) or a list of column names.
    leads : int or tuple or list, optional
        The `leads` parameter is an integer, tuple, or list that specifies the 
        number of lead values to add to the DataFrame. 
        
        - If it is an integer, the function will add that number of lead values 
          for each column specified in the `value_column` parameter. 
        
        - If it is a tuple, it will generate leads from the first to the second 
          value (inclusive). 
        
        - If it is a list, it will generate leads based on the values in the list.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting lags. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
          for augmenting lags. This can be faster than using "pandas" for large datasets. 

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with lead columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```
    
    ```{python}
    # Example 1 - Add 7 lead values for a single DataFrame object, pandas engine
    lead_df_single = (
        df 
            .query('id == "D10"')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=(1, 7),
                engine='pandas'
            )
    )
    lead_df_single
    ```
    ```{python}
    # Example 2 - Add a single lead value of 2 for each GroupBy object, polars engine
    lead_df = (
        df 
            .groupby('id')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=2,
                engine='polars'
            )
    )
    lead_df
    ```

    ```{python}
    # Example 3 add 2 lead values, 2 and 4, for a single DataFrame object, pandas engine
    lead_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_leads(
                date_column='date',
                value_column='value',
                leads=[2, 4],
                engine='pandas'
            )
    )
    lead_df_single_two
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column, require_numeric_dtype = False)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)
    
    if engine == 'pandas':
        ret = _augment_leads_pandas(data, date_column, value_column, leads)
    elif engine == 'polars':
        ret = _augment_leads_polars(data, date_column, value_column, leads)
        # Polars Index to Match Pandas
        ret.index = idx_unsorted
    else:
        raise ValueError("Invalid engine. Use 'pandas' or 'polars'.")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_leads = augment_leads


def _augment_leads_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]],
    leads: Union[int, Tuple[int, int], List[int]] = 1
) -> pd.DataFrame:
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

def _augment_leads_polars(
    data: Union[pl.DataFrame, pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_columns: Union[str, List[str]],
    leads: Union[int, Tuple[int, int], List[int]] = 1
) -> pl.DataFrame:
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        # Data is a GroupBy object, use apply to get a DataFrame
        pandas_df = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        # Data is already a DataFrame
        pandas_df = data
    elif isinstance(data, pl.DataFrame):
        # Data is already a Polars DataFrame
        pandas_df = data.to_pandas()
    else:
        raise ValueError("data must be a pandas DataFrame, pandas GroupBy object, or a Polars DataFrame")

    if isinstance(value_columns, str):
        value_columns = [value_columns]

    lead_foo = pl.col(date_column).shift(-1).alias("_lead_1")

    if isinstance(leads, int):
        leads = [leads]  # Convert to a list with a single value
    elif isinstance(leads, tuple):
        leads = list(range(leads[0], leads[1] + 1))
    elif not isinstance(leads, list):
        raise TypeError(f"Invalid leads specification: type: {type(leads)}. Please use int, tuple, or list.")

    lead_exprs = []

    for col in value_columns:
        for lead in leads:
            lead_expr = pl.col(col).shift(-lead).alias(f"{col}_lead_{lead}")
            lead_exprs.append(lead_expr)

    # Select columns
    selected_columns = [lead_foo] + lead_exprs

    # Drop the first column by position (index)
    selected_columns = selected_columns[1:]

    # Select the columns
    df = pl.DataFrame(pandas_df)
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        out_df = df.group_by(data.grouper.names, maintain_order=True).agg(selected_columns)
        out_df = out_df.explode(out_df.columns[len(data.grouper.names):])
        out_df = out_df.drop(data.grouper.names)
    else: # a dataframe
        out_df = df.select(selected_columns)

    # Concatenate the DataFrames horizontally
    df = pl.concat([df, out_df], how="horizontal").to_pandas()

    return df

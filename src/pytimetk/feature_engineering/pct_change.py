import pandas as pd
import pandas_flavor as pf
from typing import Union, List, Tuple
from pytimetk.feature_engineering import augment_diffs


@pf.register_dataframe_method
def augment_pct_change(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy], 
    date_column: str,
    value_column: Union[str, List[str]], 
    periods: Union[int, Tuple[int, int], List[int]] = 1,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    """
    Adds percentage difference (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to add percentage differenced columns to.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to sort the data before adding the percentage differenced values.
    value_column : str or list
        The `value_column` parameter is the column(s) in the DataFrame that you 
        want to add percentage differences values for. It can be either a single column name 
        (string) or a list of column names.
    periods : int or tuple or list, optional
        The `periods` parameter is an integer, tuple, or list that specifies the 
        periods to shift values when percentage differencing. 
        
        - If it is an integer, the function will add that number of percentage differences 
          values for each column specified in the `value_column` parameter. 
        
        - If it is a tuple, it will generate percentage differences from the first to the second 
          value (inclusive). 
        
        - If it is a list, it will generate percentage differences based on the values in the list.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for 
        augmenting percentage differences. It can be either "pandas" or "polars". 
        
        - The default value is "pandas".
        
        - When "polars", the function will internally use the `polars` library 
        for augmenting percentage diffs. This can be faster than using "pandas" for large 
        datasets. 

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with percentage differenced columns added to it.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk

    df = tk.load_dataset('m4_daily', parse_dates=['date'])
    df
    ```
    
    ```{python}
    # Example 1 - Add 7 pctdiff values for a single DataFrame object, pandas engine
    pctdiff_df_single = (
        df 
            .query('id == "D10"')
            .augment_pct_change(
                date_column='date',
                value_column='value',
                periods=(1, 7),
                engine='pandas'
            )
    )
    pctdiff_df_single.glimpse()
    ```
    
    ```{python}
    # Example 2 - Add a single percent differenced value of 2 for each GroupBy object, polars engine
    pctdiff_df = (
        df 
            .groupby('id')
            .augment_pct_change(
                date_column='date',
                value_column='value',
                periods=2,
                engine='polars'
            )
    )
    pctdiff_df
    ```

    ```{python}
    # Example 3 add 2 percent differenced values, 2 and 4, for a single DataFrame object, pandas engine
    pctdiff_df_single_two = (
        df 
            .query('id == "D10"')
            .augment_diffs(
                date_column='date',
                value_column='value',
                periods=[2, 4],
                engine='pandas'
            )
    )
    pctdiff_df_single_two
    ```
    """
    
    # Use augment_diffs
    ret = augment_diffs(
        data = data, 
        date_column = date_column, 
        value_column = value_column, 
        periods = periods, 
        normalize = True, 
        reduce_memory = reduce_memory, 
        engine = engine
    )
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.augment_pct_change = augment_pct_change


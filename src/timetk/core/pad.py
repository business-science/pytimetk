

import pandas as pd
import pandas_flavor as pf

from itertools import product

from timetk.utils.datetime_helpers import get_pandas_frequency

@pf.register_dataframe_method
def pad_by_time(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str,  
    freq: str = 'D',
) -> pd.DataFrame:
    '''
    Make irregular time series regular by padding with missing dates.
    
    
    The `pad_by_time` function inserts missing dates into a Pandas DataFrame or DataFrameGroupBy object, through the process making an irregularly spaced time series regularly spaced.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter can be either a Pandas DataFrame or a Pandas DataFrameGroupBy object. It represents the data that you want to pad with missing dates.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to determine the minimum and maximum dates in theDataFrame, and to generate the regular date range for padding.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the missing timestamps should be generated. It accepts a string representing a pandas frequency alias. Some common frequency aliases include:
        
        - S: secondly frequency
        - min: minute frequency
        - H: hourly frequency
        - B: business day frequency
        - D: daily frequency
        - W: weekly frequency
        - M: month end frequency
        - MS: month start frequency
        - BMS: Business month start
        - Q: quarter end frequency
        - QS: quarter start frequency
        - Y: year end frequency
        - YS: year start frequency
        
    
    
    Returns
    -------
    pd.DataFrame
        The function `pad_by_time` returns a Pandas DataFrame that has been extended with future dates.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk
    
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
    df
    ```
       
    ```{python}
    # Pad Single Time Series: Fill missing dates
    padded_df = (
        df
            .query('symbol == "AAPL"')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D'
            )
            .assign(id = lambda x: x['symbol'].ffill())
    )
    padded_df 
    ```
    
    ```{python}
    # Pad by Group: Pad each group with missing dates
    padded_df = (
        df
            .groupby('symbol')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D'
            )
    )
    padded_df
    ```
    '''
    
    # Check if data is a Pandas DataFrame or GroupBy object
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    # DATAFRAME EXTENSION - If data is a Pandas DataFrame, extend with future dates
    
    if isinstance(data, pd.DataFrame):
        
        df = data.copy()
        
        df.sort_values(by=[date_column], inplace=True)
        
        # if freq == 'auto':
        #     freq = get_pandas_frequency(df[date_column], force_regular=force_regular)

        min_date = df[date_column].min()
        max_date = df[date_column].max()

        # Generate regular date range for the entire DataFrame
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)

        # Create new DataFrame from the date range
        padded_df = pd.DataFrame({date_column: date_range})
        
        # Merge with the original DataFrame to include other columns
        padded_df = padded_df.merge(df, on=[date_column], how='left')
        
        padded_df.sort_values(by=[date_column], inplace=True)
        
        padded_df.reset_index(drop=True, inplace=True)
        
    
    # GROUPED EXTENSION - If data is a GroupBy object, extend with future dates by group
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        # Get the group names and original ungrouped data
        group_names = data.grouper.names
        data = data.obj
        
        df = data.copy()
        df.sort_values(by=[*group_names, date_column], inplace=True)

        # Get unique group combinations
        groups = df[group_names].drop_duplicates()
        padded_dfs = []
        
        for idx, group in groups.iterrows():
            mask = (df[group_names] == group).all(axis=1)
            group_df = df[mask]
            
            # if freq == 'auto':
            #     freq = get_pandas_frequency(group_df[date_column], force_regular=force_regular)
            
            min_date = group_df[date_column].min()
            max_date = group_df[date_column].max()

            # Generate regular date range for each group
            date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            
            # Create new DataFrame from all combinations
            new_df = pd.DataFrame(date_range, columns=[date_column])
            
            for col, value in group.items():  # Changed iteritems to items
                new_df[col] = value
                
            new_df = new_df.merge(group_df, on=[*group_names, date_column], how='left')
            padded_dfs.append(new_df)

        padded_df = pd.concat(padded_dfs, ignore_index=True)
        
        padded_df.sort_values(by=[*group_names, date_column], inplace=True)
        
        padded_df.reset_index(drop=True, inplace=True)

    return padded_df

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.pad_by_time = pad_by_time

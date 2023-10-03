

import pandas as pd
import pandas_flavor as pf
from typing import Union

@pf.register_dataframe_method
def pad_by_time(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,  
    freq: str = 'D',
    start_date: str = None,
    end_date: str = None,
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
    start_date : str, optional
        Specifies the start of the padded series.  If NULL, it will use the lowest value of the input variable. In the case of groups, it will use the lowest value by group.
        
    end_date  : str, optional;
        Specifies the end of the padded series.  If NULL, it will use the highest value of the input variable.  In the case of groups, it will use the highest value by group.
    
    
    Returns
    -------
    pd.DataFrame
        The function `pad_by_time` returns a Pandas DataFrame that has been extended with future dates.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
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
    
    ```{python}
    # Pad with end dates specified
    padded_df = (
        df
            .groupby('symbol')
            .pad_by_time(
                date_column = 'date',
                freq        = 'D',
                start_date  = '2013-01-01',
                end_date    = '2023-09-21'
            )
    )
    padded_df
    '''
        
    if not isinstance(data, (pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
        raise TypeError("`data` must be a Pandas DataFrame or DataFrameGroupBy object.")

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    if start_date and end_date:
        if start_date > end_date:
            raise ValueError("Start date cannot be greater than end date.")

    # Handling DataFrame
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=[date_column], inplace=True)
        
        min_date = start_date if start_date else df[date_column].min()
        max_date = end_date if end_date else df[date_column].max()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
        padded_df = pd.DataFrame({date_column: date_range})
        padded_df = padded_df.merge(df, on=[date_column], how='left')
        
        padded_df.sort_values(by=[date_column], inplace=True)
        padded_df.reset_index(drop=True, inplace=True)
        
        return padded_df
    
    # Handling DataFrameGroupBy
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=[*group_names, date_column], inplace=True)
        
        groups = df[group_names].drop_duplicates()
        padded_dfs = []
        
        for _, group in groups.iterrows():
            mask = (df[group_names] == group).all(axis=1)
            group_df = df[mask]
            
            min_date = start_date if start_date else group_df[date_column].min()
            max_date = end_date if end_date else group_df[date_column].max()
            
            date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            new_df = pd.DataFrame({date_column: date_range})
            
            for col, value in group.items():
                new_df[col] = value
                
            new_df = new_df.merge(group_df, on=[*group_names, date_column], how='left')
            padded_dfs.append(new_df)
            
        padded_df = pd.concat(padded_dfs, ignore_index=True)
        padded_df.sort_values(by=[*group_names, date_column], inplace=True)
        padded_df.reset_index(drop=True, inplace=True)
        
        return padded_df


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.pad_by_time = pad_by_time

import pandas as pd
import pandas_flavor as pf
from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

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
    
    The `pad_by_time` function inserts missing dates into a Pandas DataFrame or 
    DataFrameGroupBy object, through the process making an irregularly spaced 
    time series regularly spaced.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter can be either a Pandas DataFrame or a Pandas 
        DataFrameGroupBy object. It represents the data that you want to pad 
        with missing dates.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to determine the minimum and maximum dates in theDataFrame, and to 
        generate the regular date range for padding.
    freq : str, optional
        The `freq` parameter specifies the frequency at which the missing 
        timestamps should be generated. It accepts a string representing a 
        pandas frequency alias. Some common frequency aliases include:
        
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
        Specifies the start of the padded series.  If NULL, it will use the 
        lowest value of the input variable. In the case of groups, it will use 
        the lowest value by group.
        
    end_date  : str, optional;
        Specifies the end of the padded series.  If NULL, it will use the highest 
        value of the input variable.  In the case of groups, it will use the 
        highest value by group.
    
    
    Returns
    -------
    pd.DataFrame
        The function `pad_by_time` returns a Pandas DataFrame that has been 
        extended with future dates.
        
    Notes
    -----
    
    ## Performance
    
    This function uses a number of techniques to speed up computation for large 
    datasets with many time series groups. 
    
    - We use a vectorized approach to generate the Cartesian product of all 
      unique group values and all dates in the date range. 
    - We then merge this Cartesian product with the original data to introduce 
      NaN values for missing rows. This approach is much faster than looping 
      through each group and applying a function to each group.
    
    Note: There is no parallel processing since the vectorized approach is 
          almost always faster.
    
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
                end_date    = '2023-09-22'
            )
    )
    padded_df.query('symbol == "AAPL"')
    ```
    '''
    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    # Prep Inputs
    if start_date is not None:
        start_date = pd.Timestamp(start_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    # Check if start_date is greater than end_date
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

        col_name = padded_df.columns[padded_df.nunique() == 1]
        if not col_name.empty:
            col_name = col_name[0]
        else:
            col_name = None

        if col_name is not None:
            padded_df = padded_df.assign(**{f'{col_name}': padded_df[col_name].ffill()})
        
        return padded_df
    
    # Handling DataFrameGroupBy
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj
        df = data.copy()
        
        df[date_column] = pd.to_datetime(df[date_column])
        df.sort_values(by=[*group_names, date_column], inplace=True)
        
        padded_df = _pad_by_time_vectorized(
            data=df,
            date_column=date_column,
            groupby_columns=group_names,
            freq=freq,
            start_date=start_date,
            end_date=end_date
        )
        
        return padded_df[df.columns]

def _pad_by_time_vectorized(
    data: pd.DataFrame, 
    date_column: str, 
    groupby_columns: list,
    freq: str = 'D',
    start_date: Union[str, None] = None,
    end_date: Union[str, None] = None
) -> pd.DataFrame:
    
    # Calculate the overall min and max dates across the entire dataset if not provided
    if not start_date:
        start_date = data[date_column].min()
    if not end_date:
        end_date = data[date_column].max()
        
    # Create a full date range
    all_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate the Cartesian product of all_dates and unique group values
    idx = pd.MultiIndex.from_product(
        [data[col].unique() for col in groupby_columns] + [all_dates], 
        names=groupby_columns + [date_column]
    )
    cartesian_df = pd.DataFrame(index=idx).reset_index()
    
    # Merge to introduce NaN values for missing rows
    padded_data = pd.merge(cartesian_df, data, on=groupby_columns + [date_column], how='left')
    
    return padded_data

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.pad_by_time = pad_by_time

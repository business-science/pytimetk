import pandas as pd
import pandas_flavor as pf


@pf.register_dataframe_method
def glimpse(
    data: pd.DataFrame, max_width: int = 76
):
    '''Takes a pandas DataFrame and prints a summary of
    its dimensions, column names, data types, and the first few values of each column.
    
    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a pandas DataFrame that contains the data you want to glimpse at. It is the main input to the `glimpse` function.
    max_width : int, optional
        The `max_width` parameter is an optional parameter that specifies the maximum width of each line when printing the glimpse of the DataFrame. If not provided, the default value is set to 76.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
    
    df.glimpse()
    ```
    
    '''
    
    df = data.copy()

    # find the max string lengths of the column names and dtypes for formatting
    _max_len = max([len(col) for col in df])
    _max_dtype_label_len = max([len(str(df[col].dtype)) for col in df])

    # print the dimensions of the dataframe
    print(f"{type(df)}: {df.shape[0]} rows of {df.shape[1]} columns")

    # print the name, dtype and first few values of each column
    for _column in df:
        
        _col_vals = df[_column].head(max_width).to_list()
        _col_type = str(df[_column].dtype)
        
        output_col = f"{_column}:".ljust(_max_len+1, ' ')
        output_dtype = f" {_col_type}".ljust(_max_dtype_label_len+3, ' ')

        output_combined = f"{output_col} {output_dtype} {_col_vals}"
    
        # trim the output if too long
        if len(output_combined) > max_width:
            output_combined = output_combined[0:(max_width-4)] + " ..."
        
        print(output_combined)



@pf.register_dataframe_method
def flatten_multiindex_column_names(data: pd.DataFrame, sep = '_'):
    '''Takes a DataFrame as input and flattens the column
    names if they are in a multi-index format.
    
    Parameters
    ----------
    data : pd.DataFrame
        The parameter "data" is expected to be a pandas DataFrame object.
    
    Returns
    -------
    pd.DataFrame
        The input data with flattened multiindex column names.
        
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    date_rng = pd.date_range(start='2023-01-01', end='2023-01-03', freq='D')

    data = {
        'date': date_rng,
        ('values', 'value1'): [1, 4, 7],
        ('values', 'value2'): [2, 5, 8],
        ('metrics', 'metric1'): [3, 6, 9],
        ('metrics', 'metric2'): [3, 6, 9],
    }
    df = pd.DataFrame(data)
    
    df.flatten_multiindex_column_names()
    
    ```
    '''
    
    # Check if data is a Pandas MultiIndex
    data.columns = [sep.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
                
    return data


import pandas as pd

def flatten_multiindex_column_names(data):
    """
    The function `flatten_multiindex_column_names` takes a DataFrame as input and flattens the column
    names if they are in a multi-index format.
    
    :param data: The parameter "data" is expected to be a pandas DataFrame object
    :return: the input data with flattened multiindex column names.
    """
    # Check if data is a Pandas MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
                
    return data
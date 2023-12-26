
import pandas as pd
import numpy as np
import pandas_flavor as pf

from typing import Union


@pf.register_dataframe_method
def binarize(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    n_bins: int = 4, 
    thresh_infreq: float=0.01, 
    name_infreq: str="-OTHER", 
    one_hot: bool=True
):
    '''The `binarize` function prepares data for `correlate`, which is used for analyzing correlationfunnel plots. 
    
    Binarization does the following:
    
    1. Takes in a pandas DataFrame or DataFrameGroupBy object, converts non-numeric
    columns to categorical, 
    2. Replaces boolean columns with integers, 
    3. Checks for data type and missing
    values, 
    4. fixes low cardinality numeric data, 
    5. fixes high skew numeric data, and 
    6. finally applies a
    transformation to create a new DataFrame with binarized data.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that you want to binarize. It can be either a pandas
        DataFrame or a DataFrameGroupBy object.
    n_bins : int
        The `n_bins` parameter specifies the number of bins to use when binarizing numeric data. It is used
        in the `create_recipe` function to determine the number of bins for each numeric column. 
        `pd.qcut()` is used to bin the numeric data.
    thresh_infreq : float
        The `thresh_infreq` parameter is a float that represents the threshold for infrequent categories.
        Categories that have a frequency below this threshold will be grouped together and labeled with the
        name specified in the `name_infreq` parameter. By default, the threshold is set to 0.01.
    name_infreq : str
        The `name_infreq` parameter is used to specify the name that will be assigned to the category
        representing infrequent values in a column. This is applicable when performing binarization on
        non-numeric columns. By default, the name assigned is "-OTHER".
    one_hot : bool
        The `one_hot` parameter is a boolean flag that determines whether or not to perform one-hot
        encoding on the categorical variables after binarization. If `one_hot` is set to `True`, the
        categorical variables will be one-hot encoded, creating binary columns for each unique category. 
    
    Returns
    -------
        The function `binarize` returns the transformed data after applying various data preprocessing
        steps such as converting non-numeric columns to categorical, replacing boolean columns with
        integers, fixing low cardinality numeric data, fixing high skew numeric data, and creating a recipe
        for binarization.
        
    See Also
    --------
    - `correlate()` : Calculates the correlation between a target variable and all other variables in a pandas DataFrame.
    
    Examples
    --------
    
    ``` {python}
    # NON-TIMESERIES EXAMPLE ----
    
    import pandas as pd
    import numpy as np
    import pytimetk as tk

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of rows for your DataFrame
    num_rows = 200

    # Create fake data for the columns
    data = {
        'Age': np.random.randint(18, 65, size=num_rows),
        'Gender': np.random.choice(['Male', 'Female'], size=num_rows),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], size=num_rows),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], size=num_rows),
        'Years_Playing': np.random.randint(0, 30, size=num_rows),
        'Average_Income': np.random.randint(20000, 100000, size=num_rows),
        'Member_Status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_rows),
        'Number_Children': np.random.randint(0, 5, size=num_rows),
        'Own_House_Flag': np.random.choice([True, False], size=num_rows),
        'Own_Car_Count': np.random.randint(0, 3, size=num_rows),
        'PersonId': range(1, num_rows + 1),  # Add a PersonId column as a row count
        'Client': np.random.choice(['A', 'B'], size=num_rows)  # Add a Client column with random values 'A' or 'B'
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Binarize the data
    df_binarized = df.binarize(n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)
    
    df_binarized.glimpse()    
    ```
    
    ``` {python}
    df_correlated = df_binarized.correlate(target='Member_Status__Platinum')
    df_correlated.head(10)
    ```
    
    ``` {python}
    # Interactive
    df_correlated.plot_correlation_funnel(
        engine='plotly', 
        height=600
    )
    ```
    
    ``` {python}
    # Static
    df_correlated.plot_correlation_funnel(
        engine ='plotnine', 
        height = 900
    )
    ```
    
    '''
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Error binarize(): Object is not of class `pd.DataFrame`.")

    # Get a list of columns with non-numeric and non-boolean data types
    non_numeric_columns = data.select_dtypes(exclude=['number', 'bool']).columns.tolist()
    
    # Convert non-numeric columns to categorical
    data[non_numeric_columns] = data[non_numeric_columns].astype('object')

    # The below part is me trying to fix the datatypes :(
    # Replace boolean columns with integers (0 and 1)
    for col in data.columns:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)

    # CHECKS ----
    # Check data types
    classes_not_allowed = ['datetime64', 'timedelta[ns]', 'complex64', 'complex128']
    check_data_type(data, classes_not_allowed, "binarize")

    # Check for missing values
    check_missing(data, "binarize")

    # FIXES ----
    data = logical_to_integer(data)

    # NON-BINARY DATA ----
    if len(data.select_dtypes(include=['number']).columns) > 0:
        data = fix_low_cardinality_numeric(data, thresh=n_bins + 3)

        # Check & fix skewed data
        data = fix_high_skew_numeric_data(data, unique_limit=2)

        # TRANSFORMATION STEPS ----
        data_transformed = create_recipe(data, n_bins, thresh_infreq, name_infreq, one_hot)
    
    # Ensure that the data is binary
    data_transformed = logical_to_integer(data_transformed)

    return data_transformed

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.binarize = binarize


@pf.register_dataframe_method
def correlate(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    target: str, 
    method: str ='pearson'
):
    '''The `correlate` function calculates the correlation between a target variable and all other
    variables in a pandas DataFrame, and returns the results sorted by absolute correlation in
    descending order.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The `data` parameter is the input data that you want to calculate correlations for. It can be
        either a pandas DataFrame or a grouped DataFrame obtained from a groupby operation.
    target : str
        The `target` parameter is a string that represents the column name in the DataFrame for which you
        want to calculate the correlation with other columns.
    method : str, default = 'pearson'
        The `method` parameter in the `correlate` function is used to specify the method for calculating
        the correlation coefficient. The available options for the `method` parameter are:
        
        * pearson : standard correlation coefficient
        * kendall : Kendall Tau correlation coefficient
        * spearman : Spearman rank correlation
        
    
    Returns
    -------
        The function `correlate` returns a DataFrame with two columns: 'feature' and 'correlation'. The
        'feature' column contains the names of the features in the input data, and the 'correlation' column
        contains the correlation coefficients between each feature and the target variable. The DataFrame is
        sorted in descending order based on the absolute correlation values.
        
    See Also
    --------
    - `binarize()` : Prepares data for `correlate`, which is used for analyzing correlationfunnel plots.
    
    Examples
    --------
    
    ``` {python}
    # NON-TIMESERIES EXAMPLE ----
    
    import pandas as pd
    import numpy as np
    import pytimetk as tk

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of rows for your DataFrame
    num_rows = 200

    # Create fake data for the columns
    data = {
        'Age': np.random.randint(18, 65, size=num_rows),
        'Gender': np.random.choice(['Male', 'Female'], size=num_rows),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], size=num_rows),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], size=num_rows),
        'Years_Playing': np.random.randint(0, 30, size=num_rows),
        'Average_Income': np.random.randint(20000, 100000, size=num_rows),
        'Member_Status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_rows),
        'Number_Children': np.random.randint(0, 5, size=num_rows),
        'Own_House_Flag': np.random.choice([True, False], size=num_rows),
        'Own_Car_Count': np.random.randint(0, 3, size=num_rows),
        'PersonId': range(1, num_rows + 1),  # Add a PersonId column as a row count
        'Client': np.random.choice(['A', 'B'], size=num_rows)  # Add a Client column with random values 'A' or 'B'
    }

    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Binarize the data
    df_binarized = df.binarize(n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)
    
    df_binarized.glimpse()    
    ```
    
    ``` {python}
    df_correlated = df_binarized.correlate(target='Member_Status__Platinum')
    df_correlated
    ```
    
    ``` {python}
    # Interactive
    df_correlated.plot_correlation_funnel(
        engine='plotly', 
        height=400
    )
    ```
    
    ``` {python}
    # Static
    fig = df_correlated.plot_correlation_funnel(
        engine ='plotnine', 
        height = 600
    )
    fig
    ```
    
    '''
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        data = data.obj
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Error correlate(): Object is not of class `pd.DataFrame`.")

    if target not in data.columns:
        raise ValueError(f"Error in correlate(): '{target}' not found in the DataFrame columns.")

    if method not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Invalid correlation method. Choose from 'pearson', 'kendall', or 'spearman'.")

    # Calculate the correlation
    correlations = data.corrwith(data[target], method=method)
    correlations = correlations.reset_index()
    correlations.columns = ['feature', 'correlation']

    # Sort by absolute correlation in descending order
    correlations = correlations.sort_values(by='correlation', key=abs, ascending=False)
    
    
    # Splitting the 'feature' column
    correlations[['feature', 'bin']] = correlations['feature'].str.split('__', expand=True)
    
    # Reorder the columns
    correlations = correlations[['feature', 'bin', 'correlation']]

    return correlations

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.correlate = correlate



# UTILITIES ----

def check_data_type(data, classes_not_allowed, fun_name=None):
    invalid_cols = [col for col in data.columns if str(data[col].dtype) in classes_not_allowed]
    # print(invalid_cols)
    if invalid_cols:
        msg = f"Error {fun_name}(): The following columns have invalid data types: {', '.join(invalid_cols)}"
        raise ValueError(msg)

def check_missing(data, fun_name=None):
    missing_cols = data.columns[data.isnull().any()]
    if not missing_cols.empty:
        msg = f"Error {fun_name}(): The following columns contain missing values: {', '.join(missing_cols)}"
        raise ValueError(msg)

def fix_low_cardinality_numeric(data, thresh):
    # Converts numeric columns with number of unique values <= thresh to categorical
    num_cols = data.select_dtypes(include=['number']).columns
    for col in num_cols:
        if len(data[col].unique()) <= thresh:
            data[col] = data[col].astype('category')
    return data

def fix_high_skew_numeric_data(data, unique_limit):
    # Converts numeric columns with number of unique quantile values <= limit to categorical 
    numeric_cols = data.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        quantiles = np.quantile(data[col], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if len(np.unique(quantiles)) <= unique_limit:
            data[col] = data[col].astype('category')
    return data

def create_recipe(data, n_bins, thresh_infreq, name_infreq, one_hot):
    
    # Recipe creation steps (similar to R code)
    num_count = len(data.select_dtypes(include=['number']).columns)
    cat_count = len(data.select_dtypes(include=['object', 'category']).columns)

    recipe = pd.DataFrame()

    if num_count > 0:
        # Convert continuous features to binned features
        for col in data.select_dtypes(include=['number']).columns:
            
            binned, bins = pd.qcut(data[col], q=n_bins, retbins=True, labels=False, duplicates='drop')
            bins = bins.tolist()
            one_hot_encoded = pd.get_dummies(binned)
            
            # Ensure the number of column names matches the number of columns
            col_names = [f"{col}__{round(a,1)}_{round(b,1)}" for a, b in zip(bins[:-1], bins[1:])]
            one_hot_encoded.columns = [col_names[i] for i in one_hot_encoded.columns]

            data = pd.concat([data, one_hot_encoded], axis=1)
            data.drop(col, axis=1, inplace=True)
    
    if cat_count > 0:
        # Resolves error on thresh_infreq = 0
        if thresh_infreq == 0:
            thresh_infreq = 1e-9

        # Reduce cardinality of infrequent categorical levels
        for col in data.select_dtypes(include=['object', 'category']).columns:
            value_counts = data[col].value_counts(normalize=True)
            infrequent_values = value_counts[value_counts < thresh_infreq].index
            data[col].replace(infrequent_values, name_infreq, inplace=True)

        # Convert categorical and binned features to binary features (one-hot encoding)
        recipe = pd.get_dummies(data, prefix_sep='__')
    
    return recipe

def logical_to_integer(data):
    # Convert logical columns to integer
    logical_cols = data.select_dtypes(include=['bool']).columns
    data[logical_cols] = data[logical_cols].astype(int)
    return data

import pandas as pd
import numpy as np
import pandas_flavor as pf

@pf.register_dataframe_method
def correlate(data, target, method='pearson'):
    
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

    return correlations

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.correlate = correlate


@pf.register_dataframe_method
def binarize(data, n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True):
    
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

    return data_transformed

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.binarize = binarize


# UTILITIES ----

def check_data_type(data, classes_not_allowed, fun_name=None):
    invalid_cols = [col for col in data.columns if str(data[col].dtype) in classes_not_allowed]
    print(invalid_cols)
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
            binned, bins = pd.cut(data[col], bins=n_bins, retbins=True, labels=False, right=False)
            bins=bins.tolist()
            one_hot_encoded = pd.get_dummies(binned)
            one_hot_encoded.columns = [f"{col}__{round(a,1)}_{round(b,1)}" for a, b in zip(bins[:-1], bins[1:])]
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

import pandas as pd
import polars as pl
from importlib.resources import files
    
def load_dataset(
    name: str = "m4_daily", 
    verbose: bool = False, 
    engine: str = 'pandas',
    **kwargs
) -> pd.DataFrame:
    '''
    Load one of 12 Time Series Datasets.
    
    The `load_dataset` function is used to load various time series datasets by 
    name, with options to print the available datasets and pass additional 
    arguments to `pandas.read_csv`. The available datasets are:
    
    - `m4_hourly`: The M4 hourly dataset
    - `m4_daily`: The M4 daily dataset
    - `m4_weekly`: The M4 weekly dataset
    - `m4_monthly`: The M4 monthly dataset
    - `m4_quarterly`: The M4 quarterly dataset
    - `m4_yearly`: The M4 yearly dataset
    - `bike_sharing_daily`: The bike sharing daily dataset
    - `bike_sales_sample`: The bike sales sample dataset
    - `taylor_30_min`: The Taylor 30 minute dataset
    - `walmart_sales_weekly`: The Walmart sales weekly dataset
    - `wikipedia_traffic_daily`: The Wikipedia traffic daily dataset
    - `stocks_daily`: The MAANNG stocks dataset
    - `expedia`: Expedia Hotel Time Series Dataset
    
    The datasets can be loaded with `load_dataset(name)`, where `name` is the 
    name of the dataset that you want to load. The default value is set to 
    "m4_daily", which is the M4 daily dataset. However, you can choose from a 
    list of available datasets mentioned above.
    
    Parameters
    ----------
    name : str, optional
        The `name` parameter is used to specify the name of the dataset that 
        you want to load. The default value is set to "m4_daily", which is the 
        M4 daily dataset. However, you can choose from a list of available 
        datasets mentioned in the function's docstring.
    verbose : bool, optional
        The `verbose` parameter is a boolean flag that determines whether or not 
        to print the names of the available datasets. If `verbose` is set to 
        `True`, the function will print the names of the available datasets. 
        If `verbose` is set to `False`, the function will not print anything.
    engine : str, optional
        The `engine` parameter is used to specify the engine to use for reading 
        the csv file. The default value is set to "pandas", which uses pandas to 
        read the csv file. If `engine` is set to "polars", the function will use 
        polars to read the csv file and convert it to a pandas DataFrame.
    **kwargs
        The `**kwargs` parameter is used to pass additional arguments to 
        `pandas.read_csv`.
    
    Returns
    -------
    pd.DataFrame
        The `load_dataset` function returns the requested dataset as a pandas 
        DataFrame.
        
        
    Examples
    --------
    ```python
    # Load the M4 daily dataset using pandas
    df = load_dataset('m4_daily')
    
    df
    ```
    
    ```python
    # Load the M4 daily dataset using polars
    df = load_dataset('m4_daily', engine='polars')
    
    df
    ```    
    '''
    # Return the list of available datasets
    dataset_list = get_available_datasets()
    
    if verbose:
        print("Available Datasets:")
        print(dataset_list)
        
    if name not in dataset_list:
        raise ValueError(f"Dataset {name} not found. Please choose from the following: \n{dataset_list}")
    
    # Load the dataset
    package_path = files('pytimetk')
    # Reference to the a file within the package
    text_path = f"{package_path}/datasets/{name}.csv"
    
    if engine == 'pandas':
        with open(text_path, 'r', encoding='utf-8') as f:
            df = pd.read_csv(f, **kwargs)
    elif engine == 'polars':
        df = pl.read_csv(text_path).to_pandas()

    return df

    
def get_available_datasets():
    '''Get a list of 12 datasets that can be loaded with `pytimetk.load_dataset`.
    
    The `get_available_datasets` function returns a sorted list of available 
    dataset names from the `pytimetk.datasets` module. The available datasets are:
    
    
    
    Returns
    -------
    list
        The function `get_available_datasets` returns a sorted list of available 
        dataset names from the `pytimetk.datasets` module.
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    
    tk.get_available_datasets()
    ```
    
    '''
    
    pathlist   = list(files("pytimetk.datasets").iterdir())
    file_names = [path.name for path in pathlist]
    dataset_list = [item for item in file_names if item.endswith(".csv")]
    dataset_list = [name.rstrip('.csv') for name in dataset_list]
    dataset_list = sorted(dataset_list)
    
    return dataset_list


import pandas as pd
from importlib.resources import open_text
from importlib.resources import contents

def load_dataset(name = "m4_daily", verbose=False, **kwargs):
    """The `pytimetk` package comes with various time series datasets that can be loaded by name using this function. These include:
    
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

    Args:
        name (str, optional): Name of the dataset. Defaults to "m4_daily".
        verbose (bool, optional): Prints the names of the available datasets. Defaults to False.
        **kwargs: Additional arguments passed to `pandas.read_csv`
        

    Returns:
        pandas.DataFrame: The requested dataset as a pandas DataFrame
        
    Example:
    >>> import timetk
    >>> timetk.load_dataset("m4_daily", verbose=True)
    
    """
    
    # Return the list of available datasets
    file_names   = list(contents("timetk.datasets"))
    dataset_list = [name.rstrip('.csv') for name in file_names]
    dataset_list = [x for x in dataset_list if x not in ['__init__.py', '__pycache__']]
    dataset_list = sorted(dataset_list)

    
    if verbose:
        print("Available Datasets:")
        print(dataset_list)
        
    if name not in dataset_list:
        raise ValueError(f"Dataset {name} not found. Please choose from the following: \n{dataset_list}")
    
    # Load the dataset
    with open_text("timetk.datasets", f"{name}.csv") as f:
        df = pd.read_csv(f, **kwargs)
        
    return df


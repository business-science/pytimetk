
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
        name (str, optional): Name of the dataset. Defaults to "m4_daily". See timetk.get_available_datasets() for a list of available datasets.
        verbose (bool, optional): Prints the names of the available datasets. Defaults to False.
        **kwargs: Additional arguments passed to `pandas.read_csv`
        

    Returns:
        pandas.DataFrame: The requested dataset as a pandas DataFrame
        
    Example:
    import timetk
    
    timetk.get_available_datasets()
    
    timetk.load_dataset("m4_daily", verbose=True)
    
    """
    
    # Return the list of available datasets
    dataset_list = get_available_datasets()
    
    if verbose:
        print("Available Datasets:")
        print(dataset_list)
        
    if name not in dataset_list:
        raise ValueError(f"Dataset {name} not found. Please choose from the following: \n{dataset_list}")
    
    # Load the dataset
    with open_text("timetk.datasets", f"{name}.csv") as f:
        df = pd.read_csv(f, **kwargs)
        
    return df

def get_available_datasets():
    """
    The function `get_available_datasets` returns a sorted list of available dataset names from the
    `timetk.datasets` module.
    :return: a list of available datasets.
    """
    
    file_names   = list(contents("timetk.datasets"))
    dataset_list = [item for item in file_names if item.endswith(".csv")]
    dataset_list = [name.rstrip('.csv') for name in dataset_list]
    dataset_list = sorted(dataset_list)
    
    return dataset_list
import pkg_resources
import pandas as pd

def load_dataset(name = "m4_daily", **kwargs):
    """[summary]

    Args:
        name (str, optional): [description]. Defaults to "m4_daily".

    Returns:
        [type]: [description]
    """

    stream = pkg_resources.resource_stream(__name__, f'datasets/{name}.csv')
    return pd.read_csv(stream, **kwargs)


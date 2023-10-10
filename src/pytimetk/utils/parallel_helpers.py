
import pandas as pd

from functools import partial

from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Iterable

def conditional_tqdm(iterable: Iterable, display: bool =True, **kwargs):
    '''Conditional tqdm progress bar
    
    Parameters
    ----------
    iterable : Iterable
        The `iterable` parameter is any object that can be iterated over, such as a list, tuple, or range. It is the collection of items that you want to iterate through and potentially display a progress bar for.
    display : bool, optional
        The `display` parameter is a boolean flag that determines whether or not to display the progress bar. If `display` is set to `True`, the progress bar will be displayed using the `tqdm` library. If `display` is set to `False`, the progress bar will not be displayed and the original `iterable` will be returned. The default value is `True`.
    **kwargs 
        The `**kwargs` parameter is a dictionary of keyword arguments that are passed to the `tqdm` function when `display` is `True`.
    
    Returns
    -------
    Iterable
        The `iterable` parameter is any object that can be iterated over, such as a list, tuple, or range. It is the collection of items that you want to iterate through and potentially display a progress bar for.
    
    '''
    if display:
        return tqdm(iterable, **kwargs)
    else:
        return iterable
    

def parallel_apply(data : pd.core.groupby.generic.DataFrameGroupBy, func, show_progress=True, threads=None, **kwargs):
    """
    Parallelize apply on grouped dataframes using concurrent.futures.

    Parameters:
        data (pd.core.groupby.generic.DataFrameGroupBy): Grouped dataframe.
        func (function): Function to apply on each group.
        show_progress (bool): Whether to display progress with tqdm.
        threads (int): Number of threads. -1 means using all processors. None means all processors. Defaults to None.

    Returns:
        pd.DataFrame: Combined result after applying the function on all groups.
        
    Examples:
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd
    
    # Example 1 - Single argument returns Series
    
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 3, 4, 5, 6]
    })

    grouped = df.groupby('A')
    result = grouped.apply(lambda df: df['B'].sum())
    result
    
    result = tk.parallel_apply(grouped, lambda df: df['B'].sum(), show_progress=True)
    result
    ```
    
    ``` {python}
    # Example 2 - Multiple arguments returns MultiIndex DataFrame
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'two', 'two', 'one', 'two'],
        'C': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        # This function calculates the sum and mean of column C for each group
        # and returns a DataFrame with two columns: sum and mean.
        return pd.DataFrame({
            'sum': [group['C'].sum()],
            'mean': [group['C'].mean()]
        })

    grouped = df.groupby(['A', 'B'])
    
    result = grouped.apply(calculate)
    result
    
    result = tk.parallel_apply(grouped, calculate, show_progress=True)
    result
    
    ```
    """
    
    if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        raise TypeError("`data` is not a Pandas DataFrameGroupBy object.")
    
    grouped_df = data
    
    if threads is None: 
        threads = cpu_count()
    if threads == -1: 
        threads = cpu_count()
        
    # func = partial(func, **kwargs)

    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit both group name and group data to the executor
        futures = {executor.submit(func, group): name for name, group in grouped_df}
        
        for future in conditional_tqdm(as_completed(futures), total=len(futures), display=show_progress):
            group_name = futures[future]
            result = future.result()
            
            # If the function returns a scalar or Series, convert it to DataFrame format
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame([result])
            
            # Set the index based on the group
            if isinstance(group_name, tuple):
                result.index = pd.MultiIndex.from_tuples([group_name] * len(result), names=grouped_df.keys)
            else:
                result.index = [group_name] * len(result)
            results.append(result)

    # Concatenate the results
    return pd.concat(results)

    # Concatenate the results
    return pd.concat(results)




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
    '''The `parallel_apply` function parallelizes the application of a function on grouped dataframes using
    concurrent.futures.
    
    Parameters
    ----------
    data : pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is a Pandas DataFrameGroupBy object, which is the result of grouping a DataFrame by one or more columns.
    func
        The `func` parameter is the function that you want to apply to each group in the grouped dataframe. This function should take a single argument, which is a dataframe representing a group, and return a result. The result can be a scalar value, a pandas Series, or a pandas DataFrame.
    show_progress, optional
        A boolean parameter that determines whether to display progress using tqdm. If set to True, progress will be displayed. If set to False, progress will not be displayed.
    threads
        The `threads` parameter specifies the number of threads to use for parallel processing. If `threads` is set to `None`, it will use all available processors. If `threads` is set to `-1`, it will use all available processors as well.
    
    Returns
    -------
    pd.DataFrame
        The function `parallel_apply` returns a combined result after applying the specified function on all groups in the grouped dataframe. The result can be a pandas DataFrame or a pandas Series, depending on the function applied.
        
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
    
    # Won't match exactly because of the decision to return a Named series instead of an unnamed series with named index
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
    
    # One difference is that the multi-index does not include the level 2 index containing 0's.
    result = tk.parallel_apply(grouped, calculate, show_progress=True)
    result
    
    ```
    '''
    
    if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        raise TypeError("`data` is not a Pandas DataFrameGroupBy object.")
    
    grouped_df = data
    
    if threads is None: 
        threads = cpu_count()
    if threads == -1: 
        threads = cpu_count()
        
    func = partial(func, **kwargs)

    results_dict = {}
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit both group name and group data to the executor
        futures = {executor.submit(func, group): name for name, group in grouped_df}
        
        for future in conditional_tqdm(as_completed(futures), total=len(futures), display=show_progress):
            result = future.result()
            group_name = futures[future]
            
            # If the result is scalar or a Series
            if not isinstance(result, pd.DataFrame):
                result = pd.Series([result])
                
            # Set the index based on the group
            if isinstance(group_name, tuple):
                result.index = pd.MultiIndex.from_tuples([group_name] * len(result), names=grouped_df.keys)
            else:
                result.index = [group_name] * len(result)
                result.name = grouped_df.keys[0]
            
            results_dict[group_name] = result

    # To maintain the order, concatenate the results based on the order in the group names
    ordered_results = [results_dict[name] for name in grouped_df.groups.keys()]

    # Concatenate the results
    if isinstance(ordered_results[0], pd.Series) and len(grouped_df.keys) == 1:
        return pd.concat(ordered_results, axis=0)
    return pd.concat(ordered_results, axis=0)

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.parallel_apply = parallel_apply


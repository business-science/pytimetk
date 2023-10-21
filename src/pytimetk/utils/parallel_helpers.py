
import pandas as pd
from functools import partial
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool

from typing import Iterable, Callable



def progress_apply(data : pd.core.groupby.generic.DataFrameGroupBy, func : Callable, show_progress: bool=True, desc: str="Processing...", **kwargs):
    '''
    Adds a progress bar to pandas apply().
    
    Parameters
    ----------
    data : pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is a pandas DataFrameGroupBy object. It represents 
        a grouped DataFrame, where the data is grouped based on one or more 
        columns.
    func : Callable
        The `func` parameter is a callable function that will be applied to each 
        group in the `data` DataFrameGroupBy object. This function will be 
        applied to each group separately.
    show_progress : bool
        A boolean value indicating whether to show the progress bar or not. If 
        set to True, a progress bar will be displayed while the function is 
        being applied. If set to False, no progress bar will be displayed.
    desc : str
        The `desc` parameter is used to provide a description for the progress 
        bar. It is displayed as a prefix to the progress bar.
    **kwargs
        The `**kwargs` parameter is a dictionary of keyword arguments that are 
        passed to the `func` function.
    
    Returns
    -------
    pd.DataFrame
        The result of applying the given function to the grouped data.
        
    Examples:
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd   
    
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 3, 4, 5, 6]
    })

    grouped = df.groupby('A')
    
    result = grouped.progress_apply(lambda df: df['B'].sum())
    result
    
    ```
    
    '''
    
    tqdm = get_tqdm()
    
    tqdm.pandas(desc = desc,)
    
    # BugFix: prior to pandas 2.0.0 the group_keys parameter was set to _NoDefault.no_default. After 2.0.0 group_keys is set to True by default. This causes an error when trying to apply a function to a grouped dataframe. The following code fixes this issue. 
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        if data.group_keys is not True:
            data.group_keys = True
    
    if show_progress:
        ret = data.progress_apply(func,  **kwargs)
    else:
        ret = data.apply(func, **kwargs)
    
    return ret

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.progress_apply = progress_apply


def parallel_apply(data : pd.core.groupby.generic.DataFrameGroupBy, func : Callable, show_progress: bool=True, threads: int=None, desc="Processing...", **kwargs):
    '''
    The `parallel_apply` function parallelizes the application of a function on 
    grouped dataframes using
    concurrent.futures.
    
    Parameters
    ----------
    data : pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is a Pandas DataFrameGroupBy object, which is the 
        result of grouping a DataFrame by one or more columns. It represents the 
        grouped data that you want to apply the function to.
    func : Callable
        The `func` parameter is the function that you want to apply to each 
        group in the grouped dataframe. This function should take a single 
        argument, which is a dataframe representing a group, and return a result. 
        The result can be a scalar value, a pandas Series, or a pandas DataFrame.
    show_progress : bool, optional
        A boolean parameter that determines whether to display progress using 
        tqdm. If set to True, progress will be displayed. If set to False, 
        progress will not be displayed.
    threads : int
        The `threads` parameter specifies the number of threads to use for 
        parallel processing. If `threads` is set to `None`, it will use all 
        available processors. If `threads` is set to `-1`, it will use all 
        available processors as well.
    **kwargs
        The `**kwargs` parameter is a dictionary of keyword arguments that are 
        passed to the `func` function.
    
    Returns
    -------
    pd.DataFrame
        The `parallel_apply` function returns a combined result after applying 
        the specified function on all groups in the grouped dataframe. The 
        result can be a pandas DataFrame or a pandas Series, depending on the 
        function applied.
    
        
    Examples:
    --------
    ``` {python}
    # Example 1 - Single argument returns Series
    
    import pytimetk as tk
    import pandas as pd   
    
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar'],
        'B': [1, 2, 3, 4, 5, 6]
    })

    grouped = df.groupby('A')
    
    result = grouped.apply(lambda df: df['B'].sum())
    result
    
    result = tk.parallel_apply(grouped, lambda df: df['B'].sum(), show_progress=True, threads=2)
    result
    ```
    
    ``` {python}
    # Example 2 - Multiple arguments returns MultiIndex DataFrame
    
    import pytimetk as tk
    import pandas as pd
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'two', 'two', 'one', 'two'],
        'C': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
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
    
    ``` {python}
    # Example 3 - Multiple arguments returns MultiIndex DataFrame
    
    import pytimetk as tk
    import pandas as pd
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': ['one', 'one', 'one', 'two', 'two', 'two', 'one', 'two'],
        'C': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        return group.head(2)

    grouped = df.groupby(['A', 'B'])
    
    result = grouped.apply(calculate)
    result
    
    result = tk.parallel_apply(grouped, calculate, show_progress=True)
    result
    
    ```
    
    ``` {python}
    # Example 4 - Single Grouping Column Returns DataFrame
    
    import pytimetk as tk
    import pandas as pd
    
    df = pd.DataFrame({
        'A': ['foo', 'foo', 'bar', 'bar', 'foo', 'bar', 'foo', 'foo'],
        'B': [1, 3, 5, 7, 9, 2, 4, 6]
    })

    def calculate(group):
        return pd.DataFrame({
            'sum': [group['B'].sum()],
            'mean': [group['B'].mean()]
        })

    grouped = df.groupby(['A'])
    
    result = grouped.apply(calculate)
    result
    
    result = tk.parallel_apply(grouped, calculate, show_progress=True)
    result
    
    ```
    '''
    
    if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        raise TypeError("`data` is not a Pandas DataFrameGroupBy object.")

    if threads is None:
        threads = cpu_count()
    if threads == -1:
        threads = cpu_count()

    pool = ProcessingPool(threads)

    groups = list(data)
    func = partial(func, **kwargs)
    results = list(conditional_tqdm(pool.map(func, (group for _, group in groups)), total=len(groups), display=show_progress, desc=desc))

    # Begin post-processing to format results properly
    results_dict = {}
    for (name, _), result in zip(groups, results):
        if isinstance(result, pd.DataFrame):
            if not isinstance(name, tuple):
                name = (name, )
            if isinstance(result.index, pd.RangeIndex) and result.index.start == 0:
                new_idx = range(len(result))
                multiindex_tuples = [name + (i,) for i in new_idx]
            else:
                multiindex_tuples = [name + (idx,) for idx in result.index]
            group_keys = data.keys
            if not isinstance(group_keys, list):
                group_keys = [group_keys]
            names = list(group_keys) + [result.index.name if result.index.name else None]
            result.index = pd.MultiIndex.from_tuples(multiindex_tuples, names=names)
        else:
            result = pd.Series([result])
            result.index = [name]
            result.index.name = data.keys[0]
            result.name = None
        results_dict[name] = result

    # Convert the keys to tuples if necessary
    grouped_df_groups_keys = data.groups.keys()
    first_key_groups = next(iter(data.groups))
    if isinstance(name, tuple) and not isinstance(first_key_groups, tuple):
        grouped_df_groups_keys = [tuple([key]) for key in grouped_df_groups_keys]
    ordered_results = [results_dict[key] for key in grouped_df_groups_keys]

    return pd.concat(ordered_results, axis=0)

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.parallel_apply = parallel_apply


# Utility functions
# -----------------

def get_threads(threads: int=None):
    if threads is None: 
        threads = cpu_count()
    if threads == -1: 
        threads = cpu_count()
    return threads

def conditional_tqdm(iterable: Iterable, display: bool =True, **kwargs):
    tqdm = get_tqdm()
    if display:
        return tqdm(iterable, **kwargs)
    else:
        return iterable

def get_tqdm():
    try:
        # Check if we are in a Jupyter environment
        ipy_instance = get_ipython().__class__.__name__
        if "ZMQ" in ipy_instance or "Shell" in ipy_instance:  # Jupyter Notebook or Jupyter Lab
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
    except (NameError, ImportError):  # Not in an IPython environment
        from tqdm import tqdm
    return tqdm   

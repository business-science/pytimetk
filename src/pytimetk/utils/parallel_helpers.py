
import pandas as pd
from functools import partial

from tqdm import tqdm
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

def conditional_tqdm(iterable, display=True, **kwargs):
    if display:
        return tqdm(iterable, **kwargs)
    else:
        return iterable
    

def parallel_apply(grouped_df, func, show_progress=True, threads=None, **kwargs):
    """
    Parallelize apply on grouped dataframes using concurrent.futures.

    Parameters:
        grouped_df (pd.core.groupby.generic.DataFrameGroupBy): Grouped dataframe.
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
    """
    
    if threads is None: 
        threads = cpu_count()
    if threads == -1: 
        threads = cpu_count()

    results = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Submit both group name and group data to the executor
        futures = {executor.submit(func, group): name for name, group in grouped_df}
        
        for future in conditional_tqdm(futures, display=show_progress):
            group_name = futures[future]
            result = future.result()
            
            # Convert scalar results or other objects to a Series
            if not isinstance(result, (pd.Series, pd.DataFrame)):
                result = pd.Series(result, index=[group_name])
                
            results.append(result)

    # Concatenate the results
    combined = pd.concat(results)
    
    # Set the name of the index to match the name of the grouping column
    combined.index.name = grouped_df.grouper.names[0]

    return combined


    # if threads is None: threads = cpu_count()
    # if threads == -1: threads = cpu_count()
        
    # func = partial(func, **kwargs)  # Preload kwargs
    
    # # Data Construction: Used to map group names to unique ids and preserve order during concatenation
    # group_names = grouped_df.grouper.names
    # df = grouped_df.obj.copy()
    
    # construct_df = df  
    # for col in group_names:
    #     construct_df[col] = df[col].astype(str)
    # construct_df['unique_id'] = construct_df[group_names].apply(lambda row: '_'.join(row), axis=1)
    
    # group_names_lookup_df = construct_df[[*group_names, 'unique_id']].drop_duplicates().reset_index(drop=True)

    # construct_df.drop(columns=group_names, inplace=True)
    
    # # Parallel Processing   
    
    # with ThreadPoolExecutor(max_workers=threads) as executor:
    #     # Using a dictionary to preserve the order of results
    #     futures = {executor.submit(func, group.drop('unique_id', axis = 1)): name for name, group in construct_df.groupby('unique_id')}

    #     results = {}
    #     for future in conditional_tqdm(as_completed(futures), total=len(futures), display=show_progress):
            
    #         print(future.result())
    #         print(futures[future])
            
    #         group_name = futures[future]
    #         results[group_name] = future.result()
    
    # ret = pd.concat(results, axis=0)
    
    # # Combine results into a dataframe
    # ret = pd.DataFrame(ret)
    
    # return ret

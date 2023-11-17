import pandas as pd
import numpy as np
import pandas_flavor as pf
from typing import Union, Optional, List

from pytimetk.core.frequency import get_frequency
from pytimetk.core.make_future_timeseries import make_future_timeseries

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column

from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads

from concurrent.futures import ProcessPoolExecutor

from pytimetk.utils.memory_helpers import reduce_memory_usage

@pf.register_dataframe_method
def future_frame(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str, 
    length_out: int,
    freq: Optional[str] = None, 
    force_regular: bool = False,
    bind_data: bool = True,
    threads: int = 1,
    show_progress: bool = True,
    reduce_memory: bool = False,
    engine: str = 'pandas'
) -> pd.DataFrame:
    '''
    Extend a DataFrame or GroupBy object with future dates.
    
    The `future_frame` function extends a given DataFrame or GroupBy object with 
    future dates based on a specified length, optionally binding the original data.
    
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input DataFrame or DataFrameGroupBy object 
        that you want to extend with future dates.
    date_column : str
        The `date_column` parameter is a string that specifies the name of the 
        column in the DataFrame that contains the dates. This column will be 
        used to generate future dates.
    freq : str, optional
    length_out : int
        The `length_out` parameter specifies the number of future dates to be 
        added to the DataFrame.
    force_regular : bool, optional
        The `force_regular` parameter is a boolean flag that determines whether 
        the frequency of the future dates should be forced to be regular. If 
        `force_regular` is set to `True`, the frequency of the future dates will 
        be forced to be regular. If `force_regular` is set to `False`, the 
        frequency of the future dates will be inferred from the input data (e.g. 
        business calendars might be used). The default value is `False`.
    bind_data : bool, optional
        The `bind_data` parameter is a boolean flag that determines whether the 
        extended data should be concatenated with the original data or returned 
        separately. If `bind_data` is set to `True`, the extended data will be 
        concatenated with the original data using `pd.concat`. If `bind_data` is 
        set to `False`, the extended data will be returned separately. The 
        default value is `True`.
    threads : int
        The `threads` parameter specifies the number of threads to use for 
        parallel processing. If `threads` is set to `None`, it will use all 
        available processors. If `threads` is set to `-1`, it will use all 
        available processors as well.
    show_progress : bool, optional
        A boolean parameter that determines whether to display progress using tqdm. 
        If set to True, progress will be displayed. If set to False, progress 
        will not be displayed.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
    engine : str, optional
        The `engine` parameter specifies the engine to use for computation. 
        - Currently only `pandas` is supported.
        - `polars` will be supported in the future.
    
    Returns
    -------
    pd.DataFrame
        An extended DataFrame with future dates.
    
    Notes
    -----
    
    ## Performance
    
    This function uses a number of techniques to speed up computation for large 
    datasets with many time series groups: 
    
    - We vectorize where possible and use parallel processing to speed up. 
    - The `threads` parameter controls the number of threads to use for parallel 
      processing.
    
        - Set threads = -1 to use all available processors. 
        - Set threads = 1 to disable parallel processing.
    
    
    See Also
    --------
    make_future_timeseries: Generate future dates for a time series.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    df = tk.load_dataset('m4_hourly', parse_dates = ['date'])
    df

    # Example 1 - Extend the data for a single time series group by 12 hours
    extended_df = (
        df
            .query('id == "H10"')
            .future_frame(
                date_column = 'date', 
                length_out  = 12
            )
    )
    extended_df
    ```
    
    ```{python}
    # Example 2 - Extend the data for each group by 12 hours
    extended_df = (
        df
            .groupby('id', sort = False) # Use sort = False to preserve the original order of the data
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                threads     = 1 # Use 2 threads for parallel processing
            )
    )    
    extended_df
    ```
    
    ```{python}
    # Example 3 - Same as above, but just return the extended data with bind_data=False
    extended_df = (
        df
            .groupby('id', sort = False)
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                bind_data   = False # Returns just future data
            )
    )    
    extended_df
    ```
    
    ```{python}
    # Example 4 - Working with irregular dates: Business Days (Stocks Data)
    
    import pytimetk as tk
    import pandas as pd
    
    # Stock data
    df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
    df
    
    # Allow irregular future dates (i.e. business days)
    extended_df = (
        df
            .groupby('symbol', sort = False)
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                force_regular = False, # Allow irregular future dates (i.e. business days)),
                bind_data   = True,
                threads     = 1
            )
    )    
    extended_df
    ```
    
    ```{python}
    # Force regular: Include Weekends
    extended_df = (
        df
            .groupby('symbol', sort = False)
            .future_frame(
                date_column = 'date', 
                length_out  = 12,
                force_regular = True, # Force regular future dates (i.e. include weekends)),
                bind_data   = True
            )
    )    
    extended_df
    ```
    '''
    
    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    if reduce_memory:
        data = reduce_memory_usage(data)
    
    if engine == 'pandas':
        ret = _future_frame_pandas(
            data=data,
            date_column=date_column, 
            length_out=length_out,
            freq=freq, 
            force_regular=force_regular,
            bind_data=bind_data,
            threads=threads,
            show_progress=show_progress
        )
    elif engine == 'polars':
        raise NotImplementedError("Polars engine is not yet supported.")
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    return ret
    
    
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.future_frame = future_frame

def _future_frame_pandas(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str, 
    length_out: int,
    freq: Optional[str] = None, 
    force_regular: bool = False,
    bind_data: bool = True,
    threads: int = 1,
    show_progress: bool = True
):
    
    if isinstance(data, pd.DataFrame):
        ts_series = data[date_column]
            
        new_dates = make_future_timeseries(
            idx=ts_series, 
            length_out=length_out, 
            freq = freq,
            force_regular=force_regular
        )
        
        new_rows = pd.DataFrame({date_column: new_dates})
        
        if bind_data:
            extended_df = pd.concat([data, new_rows], axis=0, ignore_index=True)
        else:
            extended_df = new_rows

        col_name = extended_df.columns[extended_df.nunique() == 1]
        if not col_name.empty:
            col_name = col_name[0]
        else:
            col_name = None

        if col_name is not None:
            extended_df = extended_df.assign(**{f'{col_name}': extended_df[col_name].ffill()})

        return extended_df  
    
    # If the data is grouped
    elif isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        
        # If freq is None, infer the frequency from the first series in the data
        if freq is None:
            label_of_first_group = list(data.groups.keys())[0]
            
            first_group = data.get_group(label_of_first_group)
            
            freq = get_frequency(first_group[date_column].sort_values(), force_regular=force_regular)
        
        last_dates_df = data.agg({date_column: 'max'}).reset_index()

        # Use parallel processing if threads is greater than 1
        if threads != 1:
            
            threads = get_threads(threads)
            
            chunk_size = int(len(last_dates_df) / threads)
            subsets = [last_dates_df.iloc[i:i + chunk_size] for i in range(0, len(last_dates_df), chunk_size)]
            
            future_dates_list = []
            with ProcessPoolExecutor(max_workers=threads) as executor:
                results = list(conditional_tqdm(executor.map(_process_future_frame_subset, subsets, 
                                                [date_column] * len(subsets),
                                                [group_names] * len(subsets),
                                                [length_out] * len(subsets),
                                                [freq] * len(subsets), 
                                                [force_regular] * len(subsets)),
                                    total=len(subsets), display= show_progress,
                                    desc = "Future framing..."))
                for future_dates_subset in results:
                    future_dates_list.extend(future_dates_subset)
        
        # Use non-parallel processing if threads is 1
        else:
            future_dates_list = []
            for _, row in conditional_tqdm(last_dates_df.iterrows(), total=len(last_dates_df), disable=not show_progress, display=show_progress,desc="Future framing..."):
                future_dates_subset = _process_future_frame_rows(
                    row, date_column, group_names, length_out, freq, force_regular
                )
                future_dates_list.append(future_dates_subset)
        
        future_dates_df = pd.concat(future_dates_list, axis=0).reset_index(drop=True)
        
        if bind_data:
            extended_df = pd.concat([data.obj, future_dates_df], axis=0).reset_index(drop=True)
        else:
            extended_df = future_dates_df
            
        return extended_df



# UTILITIES ------------------------------------------------------------------


def _process_future_frame_subset(subset, date_column, group_names, length_out, freq, force_regular):
    future_dates_list = []
    for _, row in subset.iterrows():
        future_dates = make_future_timeseries(
            idx=pd.Series(row[date_column]), 
            length_out=length_out,
            freq=freq, 
            force_regular=force_regular
        )
        
        future_dates_df = pd.DataFrame({date_column: future_dates})
        for group_name in group_names:
            future_dates_df[group_name] = row[group_name]
        
        future_dates_list.append(future_dates_df)
    return future_dates_list

def _process_future_frame_rows(row, date_column, group_names, length_out, freq, force_regular):
    future_dates = make_future_timeseries(
        idx=pd.Series(row[date_column]),
        length_out=length_out,
        freq=freq, 
        force_regular=force_regular
    )
    
    future_dates_df = pd.DataFrame({date_column: future_dates})
    for group_name in group_names:
        future_dates_df[group_name] = row[group_name]
    
    return future_dates_df

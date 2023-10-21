import pandas as pd
import pandas_flavor as pf

from functools import partial

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from pytimetk.utils.parallel_helpers import conditional_tqdm, get_threads

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column, check_installed

from typing import Optional, Union

try:
    import tsfeatures as tsf
    from tsfeatures import (
        acf_features, arch_stat, crossing_points,
        entropy, flat_spots, heterogeneity,
        holt_parameters, lumpiness, nonlinearity,
        pacf_features, stl_features, stability,
        hw_parameters, unitroot_kpss, unitroot_pp,
        series_length, hurst
    )
    from tsfeatures.tsfeatures import _get_feats
except ImportError:
    pass 

dict_freqs = {
    'H': 24, 'D': 1,
    'M': 12, 'Q': 4,
    'W': 1, 'Y': 1
}

@pf.register_dataframe_method
def ts_features(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: str,
    features: Optional[list] = None,
    freq: Optional[str] = None,
    scale: bool = True,
    threads: Optional[int] = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    '''
    Extracts aggregated time series features from a DataFrame or DataFrameGroupBy object using the `tsfeatures` package.
    
    Note: Requires the `tsfeatures` package to be installed.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input data that can be either a Pandas 
        DataFrame or a grouped DataFrame. It contains the time series data that 
        you want to extract features from.
    date_column : str
        The `date_column` parameter is the name of the column in the input data 
        that contains the dates or timestamps of the time series data.
    value_column : str
        The `value_column` parameter is the name of the column in the DataFrame 
        that contains the time series values.
    features : list
        The `features` parameter is a list of functions that represent the time 
        series features to be extracted. Each function should take a time series 
        as input and return a scalar value as output. 
        
        When `None`, uses the default list of features:
        - acf_features
        - arch_stat 
        - crossing_points
        - entropy
        - flat_spots
        - heterogeneity
        - holt_parameters
        - lumpiness
        - nonlinearity
        - pacf_features
        - stl_features
        - stability
        - hw_parameters
        - unitroot_kpss
        - unitroot_pp
        - series_length
        - hurst
        
    freq : str
        The `freq` parameter specifies the frequency of the time series data. 
        It is used to calculate features that are dependent on the frequency, 
        such as seasonal features. 
        
        - The frequency can be specified as a string, such as 'D' for daily, 'W' 
          for weekly, 'M' for monthly.
        
        - The frequency can be a numeric value representing the number of 
          observations per year, such as 365 for daily, 52 for weekly, 12 for 
          monthly.
    scale : bool, optional
        The `scale` parameter in the `ts_features` function determines whether 
        or not to scale the extracted features. 
        - If `scale` is set to `True`, the features will be scaled using z-score 
          normalization. 
        - If `scale` is set to `False`, the features will not be scaled.
    threads : Optional[int]
        The `threads` parameter is an optional parameter that specifies the 
        number of threads to use for parallel processing. 
        - If is `None`, tthe function will use all available threads on the system.
        - If is -1, the function will use all available threads on the system.
    show_progress : bool
        The `show_progress` parameter is a boolean parameter that determines 
        whether or not to show a progress bar when extracting features.
    
    Returns
    -------
    pd.DataFrame
        The function `ts_features` returns a pandas DataFrame containing the 
        extracted time series features. If grouped data is provided, the DataFrame 
        will contain the grouping columns as well.
    
    Notes
    -----
    ## Performance
    
    This function uses parallel processing to speed up computation for large 
    datasets with many time series groups: 
    
    Parallel processing has overhead and may not be faster on small datasets.
    
    To use parallel processing, set `threads = -1` to use all available processors.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import pytimetk as tk
    
    # tsfeatures comes with these features:
    from tsfeatures import (
        acf_features, arch_stat, crossing_points,
        entropy, flat_spots, heterogeneity,
        holt_parameters, lumpiness, nonlinearity,
        pacf_features, stl_features, stability,
        hw_parameters, unitroot_kpss, unitroot_pp,
        series_length, hurst
    )
    
    df = tk.load_dataset('m4_daily', parse_dates = ['date'])
    
    # Example 1 - Grouped DataFrame
    # Feature Extraction
    feature_df = (
        df
            .groupby('id')
            .ts_features(    
                date_column   = 'date', 
                value_column  = 'value',
                features      = [acf_features, hurst],
                freq          = 7,
                threads       = 1,
                show_progress = True
            )
    ) 
    feature_df
    ```
    '''
    # Checks 
    check_installed('tsfeatures')
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
        
    # Get threads
    threads = get_threads(threads)

    group_names = None  
    if isinstance(data, pd.DataFrame):
        df = data.copy()        
        df.sort_values(by=[date_column], inplace=True)
        df['unique_id'] = "X1"
        df = df[['unique_id', date_column, value_column]]
        group_names = ['unique_id']

    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        df = data.obj.copy()
        df.sort_values(by=[*group_names, date_column], inplace=True)
        df = df[[*group_names, date_column, value_column]]
    
    if features is None:
        features = [
            acf_features,
            arch_stat,
            crossing_points,
            entropy,
            flat_spots,
            heterogeneity,
            holt_parameters,
            lumpiness,
            nonlinearity,
            pacf_features,
            stl_features,
            stability,
            hw_parameters,
            unitroot_kpss,
            unitroot_pp,
            series_length,
            hurst
        ]
    
    # Construct the DataFrame for tsfeatures
    if isinstance(data, pd.DataFrame):
        construct_df = df[group_names]
        construct_df['ds'] = df[date_column]
        construct_df['y'] = df[value_column]  
    else: # grouped dataframe
        construct_df = df[group_names]    
        for col in group_names:
            construct_df[col] = df[col].astype(str)
        construct_df['unique_id'] = construct_df[group_names].apply(lambda row: '_'.join(row), axis=1)
        
        group_names_lookup_df = construct_df[[*group_names, 'unique_id']].drop_duplicates().reset_index(drop=True)
        
        construct_df.drop(columns=group_names, inplace=True)
        construct_df['ds'] = df[date_column]
        construct_df['y'] = df[value_column]
 
    # Run tsfeatures
    # features_df = tsf.tsfeatures(construct_df, features=features, freq=freq, scale=scale, threads=threads)
    
        
    partial_get_feats = partial(
        _get_feats, 
        freq=freq, 
        scale=scale,
        features=features, 
        dict_freqs=dict_freqs
    )  
    
    if isinstance(data, pd.DataFrame):
        
        name = "X1"
        group = construct_df
        result = partial_get_feats(name, group, features = features)
        
        # RETURN SINGLE DATA FRAME HERE
        return result
        
    else: # grouped dataframe 
        
        # Replicate tsfeatures without threads
        # https://github.com/Nixtla/tsfeatures/blob/fe4f6e63b8883f84922354b7a57056cf534aa4ae/tsfeatures/tsfeatures.py#L967

    
        if threads != 1:
            
            if threads is None: threads = cpu_count()
            if threads == -1: threads = cpu_count()
            
            # Switch to concurrent.futures for better performance
            # multiprocessing.Pool is slower than concurrent.futures.ProcessPoolExecutor
            with ProcessPoolExecutor(threads) as executor:
                
                futures = [executor.submit(partial_get_feats, *args) for args in construct_df.groupby('unique_id')]
                
                ts_features = []
                for future in conditional_tqdm(as_completed(futures), total=len(futures), desc="TS Featurizing...", display=show_progress):
                    ts_features.append(future.result())
        
        else:
            # Don't parallel process
            ts_features = []
            for name, group in conditional_tqdm(construct_df.groupby('unique_id'), total=len(construct_df.unique_id.unique()), desc="TS Featurizing...", display=show_progress):
                result = partial_get_feats(name, group, features = features)
                ts_features.append(result)    
    
        # Combine the list to a DataFrame      
        ts_features = pd.concat(ts_features).rename_axis('unique_id')
        ts_features = ts_features.reset_index() 

        # Finalize id or grouping columns
        ts_features = group_names_lookup_df.merge(ts_features, on='unique_id', how='left')

        # drop unique_id column
        ts_features.drop(columns=['unique_id'], inplace=True)
        
        return ts_features
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.ts_features = ts_features
    

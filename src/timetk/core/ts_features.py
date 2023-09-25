import pandas as pd
import numpy as np

from typing import List, Callable, Optional

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
except ImportError:
    pass 



def ts_features(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column: str,
    value_column: str,
    features: list = None,
    freq: str = None,
    scale: bool = True,
    threads: Optional[int] = None,
) -> pd.DataFrame:
    '''Extracts aggregated time series features from a DataFrame or DataFrameGroupBy object using the `tsfeatures` package.
    
    Note: Requires the `tsfeatures` package to be installed.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input data that can be either a Pandas DataFrame or a grouped DataFrame. It contains the time series data that you want to extract features from.
    date_column : str
        The `date_column` parameter is the name of the column in the input data that contains the dates or timestamps of the time series data.
    value_column : str
        The `value_column` parameter is the name of the column in the DataFrame that contains the time series values.
    features : list
        The `features` parameter is a list of functions that represent the time series features to be extracted. Each function should take a time series as input and return a scalar value as output. 
        
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
        The `freq` parameter specifies the frequency of the time series data. It is used to calculate features that are dependent on the frequency, such as seasonal features. 
        
        - The frequency can be specified as a string, such as 'D' for daily, 'W' for weekly, 'M' for monthly.
        
        - The frequency can be a numeric value representing the number of observations per year, such as 365 for daily, 52 for weekly, 12 for monthly.
    scale : bool, optional
        The `scale` parameter in the `ts_features` function determines whether or not to scale the extracted features. 
        - If `scale` is set to `True`, the features will be scaled using z-score normalization. 
        - If `scale` is set to `False`, the features will not be scaled.
    threads : Optional[int]
        The `threads` parameter is an optional parameter that specifies the number of threads to use for parallel processing. If not specified, the function will use the default number of threads available on the system.
    
    Returns
    -------
        The function `ts_features` returns a pandas DataFrame containing the extracted time series features.
        
    Examples
    --------
    ```python
    # REQUIRES tsfeatures: pip install tsfeatures
    import pandas as pd
    import timetk as tk
    
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
    df
    ```
    
    ```python
    # Feature Extraction
    feature_df = (
        df
            .groupby('id')
            .ts_features(    
                date_column  = 'date', 
                value_column = 'value',
                features     = [acf_features, hurst],
                freq         = 7
            )
    ) 
    feature_df
    ```
    '''
    
    # This function requires the holidays package to be installed
    try:
        import tsfeatures
    except ImportError:
        raise ImportError("The 'tsfeatures' package is not installed. Please install it by running 'pip install tsfeatures'.")
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    if isinstance(data, pd.DataFrame):
        df = data.copy()        
        df.sort_values(by=[date_column], inplace=True)
        df['unique_id'] = "X1"
        group_names = ['unique_id']
    
    group_names = None
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj
        
        df = data.copy()
        df.sort_values(by=[*group_names, date_column], inplace=True)
        
    
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
    construct_df = pd.DataFrame()    
    construct_df['unique_id'] = df[group_names].apply(lambda row: '_'.join(row), axis=1)
    construct_df['ds'] = df[date_column]
    construct_df['y'] = df[value_column]
    
    # Run tsfeatures
    features_df = tsf.tsfeatures(construct_df, features=features, freq=freq, scale=scale, threads=threads)
    
    return features_df
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.ts_features = ts_features
    
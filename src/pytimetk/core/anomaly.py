import pandas as pd
import pandas_flavor as pf
import numpy as np

from typing import Union

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose

@pf.register_dataframe_method
def anomalize(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: str,
    period: int = None,
    method: str = 'twitter',
    iqr_alpha: float = 0.05,
    max_anomalies: float = 0.2,
    
    
) -> pd.DataFrame:
    """
    
    Examples
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np

    # Create a date range
    date_rng = pd.date_range(start='2021-01-01', end='2024-01-01', freq='MS')

    # Generate some random data with a few outliers
    np.random.seed(42)
    data = np.random.randn(len(date_rng)) * 10 + 25  
    data[3] = 100  # outlier

    # Create a DataFrame
    df = pd.DataFrame(date_rng, columns=['date'])
    df['value'] = data
    df    
    
    df.plot_timeseries("date", "value")
    
    # Anomalize the data
    anomalize_df = tk.anomalize(df, "date", "value", method = "seasonal", iqr_alpha = 0.05)
    
    anomalize_df.glimpse()
    
    anomalize_df[["date", "observed", "recomposed_l1", "recomposed_l2"]] \
        .melt(id_vars = "date", value_name='val') \
        .plot_timeseries("date", "val", color_column = "variable", smooth = False)
    ```
    """
    
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
    
    
    # STEP 1: Decompose the time series
    if method == 'twitter':
        
        median_span = len(data[value_column]) // 4
        
        result = _twitter_decompose(
            data = data, 
            date_column=date_column, 
            value_column=value_column, 
            period = period,
            median_span=median_span,
        )
    else:
        result = _seasonal_decompose(
            data = data, 
            date_column=date_column, 
            value_column=value_column, 
            period = period,
            model='additive',
            filt=None,
            two_sided=True, 
            extrapolate_trend = 'freq'
        )
    
    result = pd.concat([data, result.drop(date_column, axis=1)], axis=1)
    
    
    # STEP 2: Identify the outliers
    
    outlier_dict = _iqr(
        data = result, 
        target = 'remainder', 
        alpha = iqr_alpha, 
        max_anoms = max_anomalies
    )
    
    limits = outlier_dict['critical_limits']
    outliers = outlier_dict['outlier'].sort_index()
    
    result['remainder_l1'] = limits[0]
    result['remainder_l2'] = limits[1]
    
    result['anomaly'] = outliers
    
    # STEP 3: Recompose the time series
    
    result['recomposed_l1'] = result['seasonal'] + result['trend'] + result['remainder_l1']
    
    result['recomposed_l2'] = result['seasonal'] + result['trend'] + result['remainder_l2']
    
   
    return result
    

def _seasonal_decompose(
    data, 
    date_column, 
    value_column, 
    model='additive',
    period = None, 
    filt=None,
    two_sided=True, 
    extrapolate_trend = 'freq'
):
    '''
    This is an internal function that is not meant to be called directly by the user.
    
    Examples 
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=365),
        'value': np.sin(np.linspace(0, 365, 365)) + np.random.normal(scale=0.5, size=365) + np.linspace(0, 5, 365)
    })
    
    
    ```
    '''
    
    orig_index = data.index
        
    series = data.set_index(date_column)[value_column]
     
    
    # Need to add freq, trend, and kwargs
    result = seasonal_decompose(
        series, 
        model=model,
        period = period,
        filt=filt,
        two_sided=two_sided,
        extrapolate_trend = extrapolate_trend,
    )
    
    # Construct TS Decomposition DataFrame
    observed = series
    
    seasadj = series - result.seasonal
    
    trend = result.trend
    
    resid = seasadj - trend
    
    
    result_df = pd.concat([observed, result.seasonal, seasadj, trend, resid], axis=1)
    
    result_df.columns = ['observed', 'seasonal', 'seasadj', 'trend', 'remainder']
    
    result_df.reset_index(inplace=True)
    
    result_df.index = orig_index

    
    return result_df 

def _stl_decompose(data, date_column, value_column, period = None, seasonal_span = None, trend_span = None, robust = True, **kwargs):
    '''
    This is an internal function that is not meant to be called directly by the user.
    
    Examples 
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=365),
        'value': np.sin(np.linspace(0, 365, 365)) + np.random.normal(scale=0.5, size=365) + np.linspace(0, 5, 365)
    })
    
    tk.stl_decompose(df, "date", "value")
    ```
    '''
    
    orig_index = data.index
        
    series = data.set_index(date_column)[value_column]
    
    def make_odd(n):
        return n + 1 if n % 2 == 0 else n   
    
    # Need to add freq, trend, and kwargs
    result = STL(
        series, 
        period = period, 
        seasonal = make_odd(seasonal_span), 
        trend = make_odd(trend_span),
        robust = robust,
    ).fit()
    
    # Construct TS Decomposition DataFrame
    observed = series
    
    seasadj = series - result.seasonal
    
    result_df = pd.concat([observed, result.seasonal, seasadj, result.trend, result.resid], axis=1)
    
    result_df.columns = ['observed', 'seasonal', 'seasadj', 'trend', 'remainder']
    
    result_df.reset_index(inplace=True)
    
    result_df.index = orig_index

    
    return result_df 
    
def _twitter_decompose(
    data, 
    date_column, 
    value_column, 
    period = None, 
    median_span = None,
):
    orig_index = data.index
        
    series = data.set_index(date_column)[value_column]
     
    
    # Need to add freq, trend, and kwargs
    result = seasonal_decompose(
        series, 
        period = period,
    )
    
    # Construct TS Decomposition DataFrame
    observed = series
    
    seasadj = series - result.seasonal
    seasadj.name = 'seasadj'
    
    # Calculate median trend
    
    median_span = len(series) // 6
    
    def repeat_sequence(seq, length_out):
        quotient, remainder = divmod(length_out, len(seq))
        return seq * quotient + seq[:remainder]
    
    df = pd.DataFrame(seasadj)
    
    df['median_index'] = sorted(repeat_sequence(list(range(median_span)), len(seasadj)))
    
    trend = df.groupby('median_index')['seasadj'].transform('median')
    
    resid = seasadj - trend
    
    
    result_df = pd.concat([observed, result.seasonal, seasadj, trend, resid], axis=1)
    
    result_df.columns = ['observed', 'seasonal', 'seasadj', 'trend', 'remainder']
    
    result_df.reset_index(inplace=True)
    
    result_df.index = orig_index

    
    return result_df 
    
    
    
    


def _iqr(data, target, alpha=0.05, max_anoms=0.2):
    """
    This function is not intended for general use. It is used internally by the anomaly detection functions.
    
    Examples
    --------
    ``` {python}
    import pytimetk as tk
    import pandas as pd
    import numpy as np
    
    # Example Usage:
    df = pd.DataFrame({
        'x': list(range(100)),
        'y': np.random.randn(100)
    })

    outlier_dict = tk.iqr(df, 'y')
    ```
    """
    
    # Compute the interquartile range
    q1, q3 = np.percentile(data[target], [25, 75])
    iq_range = q3 - q1
    limits = [-1*(q1 + (0.15 / alpha) * iq_range), q3 + (0.15 / alpha) * iq_range]

    # Identify the outliers
    outlier_idx = (data[target] < limits[0]) | (data[target] > limits[1])
    outlier_vals = data.loc[outlier_idx, target]

    centerline = sum(limits) / 2
    data['distance'] = abs(data[target] - centerline)
    data_sorted = data.sort_values(by='distance', ascending=False)

    n_outliers = int(max_anoms * len(data_sorted))
    outliers_reported = ['No'] * len(data_sorted)
    outliers_reported[:n_outliers] = ['Yes'] * n_outliers

    data_sorted['outlier_reported'] = outliers_reported

    # Direction of the outlier
    data_sorted['direction'] = np.where(data_sorted[target] > limits[1], 'Up', np.where(data_sorted[target] < limits[0], 'Down', "Not Outlier"))

    
    return {
        'outlier': data_sorted['outlier_reported'],
        
        'outlier_idx': data_sorted.index[data_sorted['outlier_reported'] == 'Yes'],
        
        'outlier_vals': data_sorted[data_sorted['outlier_reported'] == 'Yes'][target],
        
        'outlier_direction': data_sorted[data_sorted['outlier_reported'] == 'Yes']['direction'],
        
        'critical_limits': limits,
        
        'outlier_report': data_sorted[['outlier_reported', 'direction']]
    }
    



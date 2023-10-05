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
    seasonal_span: int = None,
    trend_span: int = None,
) -> None:
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
    data = np.random.randn(len(date_rng)) * 36 + 50  
    data[3] = 200  # outlier

    # Create a DataFrame
    df = pd.DataFrame(date_rng, columns=['date'])
    df['value'] = data
    df    
    
    df.plot_timeseries("date", "value")
    
    # Anomalize the data
    anomalize_df = tk.anomalize(df, "date", "value", period = 12, seasonal_span=12, trend_span=12)
    
    anomalize_df \
        .melt(id_vars = "date", value_name='val') \
        .groupby('variable') \
        .plot_timeseries("date", "val", smooth = False)
    ```
    """
    
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_value_column(data, value_column)
    
    
    # STEP 1: Decompose the time series
    data_stl = stl_decompose(
        data = data, 
        date_column=date_column, 
        value_column=value_column, 
        period = period,
        seasonal_span=seasonal_span, 
        trend_span=trend_span,
        robust = True,
    )
    data_stl = pd.concat([data, data_stl.drop(date_column, axis=1)], axis=1)
    
    
    # STEP 2: Identify the outliers
    
    
    # STEP 3: Recompose the time series
    
    ret = data_stl
   
    return ret
    


@pf.register_dataframe_method
def stl_decompose(data, date_column, value_column, period = None, seasonal_span = None, trend_span = None, robust = True, **kwargs):
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
        
    series = data.set_index(date_column, **kwargs)[value_column]
    
    def make_odd(n):
        return n + 1 if n % 2 == 0 else n   
    
    # Need to add freq, trend, and kwargs
    result_stl = STL(
        series, 
        period = period, 
        seasonal = make_odd(seasonal_span), 
        trend = make_odd(trend_span),
        robust = robust,
    ).fit()
    
    # Construct TS Decomposition DataFrame
    observed = series
    
    seasadj = series - result_stl.seasonal
    
    stl_df = pd.concat([observed, result_stl.seasonal, seasadj, result_stl.trend, result_stl.resid], axis=1)
    
    stl_df.columns = ['observed', 'seasonal', 'seasadj', 'trend', 'remainder']
    
    stl_df.reset_index(inplace=True)
    
    stl_df.index = orig_index

    
    return stl_df 
    


@pf.register_dataframe_method
def iqr(data, target, alpha=0.05, max_anoms=0.2):
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
    limits = [q1 - (0.15 / alpha) * iq_range, q3 + (0.15 / alpha) * iq_range]

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
    data_sorted['direction'] = np.where(data_sorted[target] > limits[1], 'Up', np.where(data_sorted[target] < limits[0], 'Down', 'NA'))

    
    return {
        'outlier': data_sorted['outlier_reported'],
        
        'outlier_idx': data_sorted.index[data_sorted['outlier_reported'] == 'Yes'],
        
        'outlier_vals': data_sorted[data_sorted['outlier_reported'] == 'Yes'][target],
        
        'outlier_direction': data_sorted[data_sorted['outlier_reported'] == 'Yes']['direction'],
        
        'critical_limits': limits,
        
        'outlier_report': data_sorted[['outlier_reported', 'direction']]
    }
    



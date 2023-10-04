import pandas as pd

from pytimetk import ts_summary

from statsmodels.tsa.seasonal import STL

def stl_decompose(data, date_column, value_column, freq, trend, **kwargs):
        
    summary_df = ts_summary(data, date_column)
    
    # Decompose the time series
    stl = STL(data[value_column], freq=freq, **kwargs)
    res = stl.fit()
    
    # Build the dataframe containing the decomposed components
    df = pd.DataFrame({
        'date': data[date_column],
        'value': data[value_column],
        'trend': res.trend,
        'seasonal': res.seasonal,
        'residual': res.resid
    })
    
    return df 
    
    
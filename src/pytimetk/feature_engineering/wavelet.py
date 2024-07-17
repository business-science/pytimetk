import numpy as np
import pandas as pd
import pandas_flavor as pf
import polars as pl

from typing import Union, List
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column
from pytimetk.utils.polars_helpers import pandas_to_polars_frequency, pandas_to_polars_aggregation_mapping
from pytimetk.utils.memory_helpers import reduce_memory_usage
from pytimetk.utils.pandas_helpers import sort_dataframe



#@pf.register_dataframe_method
def augment_wavelet(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str, 
    value_column: str, 
    method: str, 
    sample_rate: str,
    scales: Union[str, List[str]],
    reduce_memory: bool = False,
) -> pd.DataFrame:
    """
    Apply the Wavely transform to specified columns of a DataFrame or 
    DataFrameGroupBy object.
    
    A wavelet transform is a mathematical tool used to decompose a signal or function into different frequency components and then study each component with a resolution matched to its scale. The wavelet transform uses wavelets, which are functions that are localized in both time and frequency. 
    
    Uses:
    
    1. Noise Reduction: Wavelet transform can be used to filter out noise from signals. By transforming a noisy signal and then zeroing out the wavelet coefficients that correspond to noise, the inverse wavelet transform can produce a denoised version of the original signal.
    
    2. Feature Extraction: In pattern recognition and machine learning, wavelet transforms can be used to extract features from signals which can be fed to forecasting algorithms.

    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        Input DataFrame or DataFrameGroupBy object with one or more columns of 
        real-valued signals.
    value_column : str or list
        List of column names in 'data' to which the Hilbert transform will be 
        applied.
    sample_rate : 
        Sampling rate of the input data.
        For time-series data, the sample rate (sample_rate) typically refers 
        to the frequency at which data points are collected.

        For example, if your data has a 30-minute interval, if you think of the 
        data in terms of "samples per hour", the sample rate would be:

        sample_rate = samples / hour = 1 / 0.5 = 2
    scales : str or list 
        Array of scales to use in the transform.
        The choice of scales in wavelet analysis determines which frequencies 
        (or periodicities) in the data you want to analyze. In other words, the 
        scales determine the "window size" or the "look-back period" the wavelet 
        uses to analyze the data.

        Smaller scales: Correspond to analyzing high-frequency changes 
        (short-term fluctuations) in the data.
    
        Larger scales: Correspond to analyzing low-frequency changes 
        (long-term fluctuations) in the data.

        The specific values for scales depend on what frequencies or 
        periodicities you expect in your data and wish to study.

        For instance, if you believe there are daily, weekly, and monthly 
        patterns in your data, you'd choose scales that correspond to these 
        periodicities given your sampling rate.

        For a daily pattern with data at 30-minute intervals:
        scales = 2 * 24 = 48 because there are 48 half hour intervals in a day

        For a weekly pattern with data at 30-minute intervals:
        scales = 48 * 7 = 336 because there are 336 half hour intervals in a 
        week

        Recommendation, use a range of values to cover both short term and long
        term patterns, then adjust accordingly.
    reduce_memory : bool, optional
        The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.


    Returns
    -------
    df_wavelet : pd.DataFrame
        DataFrame with added columns for CWT coefficients for each scale, with 
        a real and imaginary column added.

    Notes
    -----
    For a detailed introduction to wavelet transforms, you can visit this 
    website.
    https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

    The Bump wavelet is a real-valued wavelet function, so its imaginary 
    part is inherently zero.

    In the continuous wavelet transform (CWT), the Morlet and Analytic 
    Morlet wavelets are complex-valued, so their convolutions with the signal 
    yield complex results (with both real and imaginary parts). 

    Wavelets, in general, are mathematical functions that can decompose a 
    signal into its constituent parts at different scales. Different wavelet 
    functions are suitable for different types of signals and analytical goals. 
    Let's look at the three wavelet methods:

    1. Morlet Wavelet:
        
        Characteristics:
        Essentially a complex sinusoid modulated by a Gaussian window.
        It provides a good balance between time localization and frequency 
        localization.
    
        When to use:
        When you want a good compromise between time and frequency localization.
        Particularly useful when you're interested in sinusoidal components or 
        oscillatory patterns of your data. Commonly used in time-frequency analysis 
        because of its simplicity and effectiveness.

    2. Bump Wavelet:
        
        Characteristics:
        Has an oscillating behavior similar to the Morlet but has sharper time 
        localization. Its frequency localization isn't as sharp as its time 
        localization.

        When to use:
        When you are more interested in precisely identifying when certain events or 
        anomalies occur in your data. It can be especially useful for detecting 
        sharp spikes or short-lived events in your signal.

    3. Analytic Morlet Wavelet:
        
        Characteristics:
        A variation of the Morlet wavelet that is designed to have no negative 
        frequencies when transformed. This means it's "analytic." Offers slightly 
        better frequency localization than the standard Morlet wavelet.

        When to use:
        When you're interested in phase properties of your signal.
        Can be used when you need to avoid negative frequencies in your analysis, 
        making it useful for certain types of signals, like analytic signals.
        Offers a cleaner spectrum in the frequency domain than the standard Morlet.

    Examples
    --------
    ```{python}
    # Example 1: Using Pandas Engine on a pandas groupby object
    import pytimetk as tk
    import pandas as pd
    
    df = tk.datasets.load_dataset('walmart_sales_weekly', parse_dates = ['Date'])

    wavelet_df = (
        df
            .groupby('id')
            .augment_wavelet(
                date_column = 'Date',
                value_column ='Weekly_Sales', 
                scales = [15],
                sample_rate =1,
                method = 'bump'
            )
        )
    wavelet_df.head()

    ```

    ```{python}
    # Example 2: Using Pandas Engine on a pandas dataframe
    import pytimetk as tk
    import pandas as pd

    df = tk.load_dataset('taylor_30_min', parse_dates = ['date'])

    result_df = (
        tk.augment_wavelet(
            df, 
            date_column = 'date',
            value_column ='value', 
            scales = [15],
            sample_rate =1000,
            method = 'morlet'
        )
    )
    
    result_df
    ```
    """
    # Run common checks
    check_dataframe_or_groupby(data)
    check_value_column(data, value_column)
    check_date_column(data, date_column)

    if reduce_memory:
        data = reduce_memory_usage(data)
        
    data, idx_unsorted = sort_dataframe(data, date_column, keep_grouped_df = True)

    wavelet_functions = {
        'morlet': morlet_wavelet,
        'bump': bump_wavelet,
        'analytic_morlet': analytic_morlet_wavelet
    }

    # Sort the DataFrame by the date column before applying the CWT
    if isinstance(data, pd.DataFrame):
        data = data.sort_values(by=date_column)
    
    if method not in wavelet_functions:
        raise ValueError(f"Invalid method '{method}'. Available methods are {list(wavelet_functions.keys())}")

    # Select the wavelet function
    wavelet_function = wavelet_functions[method]

    # Compute the CWT
    def compute_cwt(signal, wavelet_function, scales, sampling_rate):
        coefficients = []

        for scale in scales:
            # Adjust the wavelet time vector based on the sample rate
            wavelet_data = wavelet_function(np.arange(-len(signal) // 2, len(signal) // 2) / sampling_rate / scale)
            convolution = np.convolve(signal, np.conj(wavelet_data), mode='same')
            coefficients.append(convolution)
        return np.array(coefficients)

    # Define helper function
    def _apply_cwt(df):
        values = df[value_column].values
        coeffs = compute_cwt(values, wavelet_function, scales, sample_rate)
        for idx, scale in enumerate(scales):
            df[f'{method}_scale_{scale}_real'] = coeffs[idx].real
            df[f'{method}_scale_{scale}_imag'] = coeffs[idx].imag
        return df

    # Check if data is a groupby object
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        ret = pd.concat([_apply_cwt(group.sort_values(by=date_column)) for _, group in data]).reset_index(drop=True)
    else:
        ret = _apply_cwt(data)

    if reduce_memory:
        ret = reduce_memory_usage(ret)
        
    ret = ret.sort_index()
    
    return ret

# Monkey-patch the method to the DataFrameGroupBy class
pd.core.groupby.DataFrameGroupBy.augment_wavelet = augment_wavelet
    
def morlet_wavelet(t, fc=1.0):
    """Compute the Complex Morlet wavelet"""
    return np.exp(1j * np.pi * fc * t) * np.exp(-t**2 / 2)

def bump_wavelet(t, w=1.0):
    """Compute the Bump wavelet."""
    s1 = np.exp(-1 / (1 - t**2))
    s2 = np.exp(-w**2 / (w**2 - t**2))
    condition = np.logical_and(t > -1, t < 1)
    return np.where(condition, s1 * s2, 0)

def analytic_morlet_wavelet(t, w=5.0):
    """Compute the Analytic Morlet wavelet."""
    s1 = np.exp(2j * np.pi * w * t)
    s2 = np.exp(-(t**2) / 2)
    return s1 * s2



from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy as _BaseGroupBy

class DataFrameGroupBy(
    _BaseGroupBy,
):

    def anomalize(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Detects anomalies in time series data, either for a single time
        series or for multiple time series grouped by a specific column.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data, which can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object.
        date_column : str
            The name of the column in the data that contains the dates or timestamps.
        value_column : str
            The name of the column in the data that contains the values to be analyzed
            for anomalies.
        period : Optional[int]
            The `period` parameter specifies the length of the seasonal component in the
            time series. It is used in the decomposition process to separate the time
            series into its seasonal, trend, and remainder components. If not specified,
            the function will automatically determine the period based on the data.
        trend : Optional[int]
            The `trend` parameter is an optional integer that specifies the length of
            the moving average window used for trend estimation. If `trend` is set to
            `None`, no trend estimation will be performed.
        method : str
            The `method` parameter determines the method used for anomaly detection.
            The only available method is `twitter`, which is the default value.
            More anomaly detection methods will be added in upcoming releases.
        decomp : str
            The `decomp` parameter specifies the type of decomposition to use for time
            series decomposition. It can take two values:
            1. 'additive' - This is the default value. It specifies that the time series
                will be decomposed using an additive model.
            2. 'multiplicative' - This specifies that the time series will be decomposed
                using a multiplicative model.
        clean : str
            The `clean` parameter specifies the method used to clean the anomalies.
            It can take two values:
        
            1. 'min_max' - This specifies that the anomalies will be cleaned using the
                min-max method. This method replaces the anomalies with the 0.75 * lower
                or upper bound of the recomposed time series, depending on the direction
                of the anomaly. The 0.75 multiplier can be adjusted using the
                `clean_alpha` parameter.
            2. 'linear' - This specifies that the anomalies will be cleaned using
                        linear interpolation.
        iqr_alpha : float
            The `iqr_alpha` parameter is used to determine the threshold for detecting
            outliers. It is the significance level used in the interquartile range (IQR)
            method for outlier detection.
            - The default value is 0.05, which corresponds to a 5% significance level.
            - A lower significance level will result in a higher threshold, which means
            fewer outliers will be detected.
            - A higher significance level will result in a lower threshold, which means
            more outliers will be detected.
        clean_alpha : float
            The `clean_alpha` parameter is used to determine the threshold for cleaning
            the outliers. The default is 0.75, which means that the anomalies will be
            cleaned using the 0.75 * lower or upper bound of the recomposed time series,
            depending on the direction of the anomaly.
        max_anomalies : float
            The `max_anomalies` parameter is used to specify the maximum percentage of
            anomalies allowed in the data. It is a float value between 0 and 1. For
            example, if `max_anomalies` is set to 0.2, it means that the function will
            identify and remove outliers until the percentage of outliers in the data is
            less than or equal to 20%. The default value is 0.2.
        bind_data : bool
            The `bind_data` parameter determines whether the original data will be
            included in the output. If set to `True`, the original data will be included
            in the output dataframe. If set to `False`, only the anomalous data will be
            included.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        threads : int
            The `threads` parameter specifies the number of threads to use for parallel
            processing. By default, it is set to `1`, which means no parallel processing
            is used. If you set `threads` to `-1`, it will use all available processors
            for parallel processing.
        show_progress : bool
            A boolean parameter that determines whether to show a progress bar during
            the execution of the function. If set to True, a progress bar will be
            displayed. If set to False, no progress bar will be shown.
        verbose: bool
            The `verbose` parameter is a boolean flag that determines whether or not to
            display additional information and progress updates during the execution of
            the `anomalize` function. If `verbose` is set to `True`, you will see more
            detailed output.
        
        Returns
        -------
        pd.DataFrame
            Returns a pandas DataFrame containing the original data with additional columns.
        
        - observed: original data
        - seasonal: seasonal component
        - seasadaj: seasonal adjusted
        - trend: trend component
        - remainder: residual component
        - anomaly: Yes/No flag for outlier detection
        - anomaly score: distance from centerline
        - anomaly direction: -1, 0, 1 inidicator for direction of the anomaly
        - recomposed_l1: lower level bound of recomposed time series
        - recomposed_l2: upper level bound of recomposed time series
        - observed_clean: original data with anomalies interpolated
        
        
        Notes
        -----
        ## Performance
        
        This function uses parallel processing to speed up computation for large
        datasets with many time series groups:
        
        Parallel processing has overhead and may not be faster on small datasets.
        
        To use parallel processing, set `threads = -1` to use all available processors.
        
        
        Examples
        --------
        ``` {python}
        # EXAMPLE 1: SINGLE TIME SERIES
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
        
        # Anomalize the data
        anomalize_df = tk.anomalize(
            df, "date", "value",
            method = "twitter",
            iqr_alpha = 0.10,
            clean_alpha = 0.75,
            clean = "min_max",
            verbose = True,
        )
        
        anomalize_df.glimpse()
        ```
        
        ``` {python}
        # Visualize the results
        anomalize_df.plot_anomalies_decomp("date")
        ```
        
        ``` {python}
        # Visualize the anomaly bands
        (
             anomalize_df
                .plot_anomalies(
                    date_column = "date",
                    engine = "plotly",
                )
        )
        ```
        
        ``` {python}
        # Get the anomalies
        anomalize_df.query("anomaly=='Yes'")
        ```
        
        ``` {python}
        # Visualize observed vs cleaned
        anomalize_df.plot_anomalies_cleaned("date")
        ```
        
        ``` {python}
        # EXAMPLE 2: MULTIPLE TIME SERIES
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset("wikipedia_traffic_daily", parse_dates = ['date'])
        
        anomalize_df = (
            df
                .groupby('Page', sort = False)
                .anomalize(
                    date_column = "date",
                    value_column = "value",
                    method = "stl",
                    iqr_alpha = 0.025,
                    verbose = False,
                )
        )
        
        # Visualize the decomposition results
        
        (
            anomalize_df
                .groupby("Page")
                .plot_anomalies_decomp(
                    date_column = "date",
                    width = 1800,
                    height = 1000,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly'
                )
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands
        (
            anomalize_df
                .groupby("Page")
                .plot_anomalies(
                    date_column = "date",
                    facet_ncol = 2,
                    width = 1000,
                    height = 1000,
                )
        )
        ```
        
        ``` {python}
        # Get the anomalies
        anomalize_df.query("anomaly=='Yes'")
        ```
        
        ``` {python}
        # Visualize observed vs cleaned
        (
            anomalize_df
                .groupby("Page")
                .plot_anomalies_cleaned(
                    "date",
                    facet_ncol = 2,
                    width = 1000,
                    height = 1000,
                )
        )
        ```
        """
        ...
    def apply_by_time(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply for time series.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object. It represents the data on which the apply operation
            will be performed.
        date_column : str
            The name of the column in the DataFrame that contains the dates.
        freq : str, optional
            The `freq` parameter specifies the frequency at which the data should be
            resampled. It accepts a string representing a time frequency, such as "D"
            for daily, "W" for weekly, "M" for monthly, etc. The default value is "D",
            which means the data will be resampled on a daily basis. Some common
            frequency aliases include:
        
            - S: secondly frequency
            - min: minute frequency
            - H: hourly frequency
            - D: daily frequency
            - W: weekly frequency
            - M: month end frequency
            - MS: month start frequency
            - Q: quarter end frequency
            - QS: quarter start frequency
            - Y: year end frequency
            - YS: year start frequency
        
        wide_format : bool, optional
            The `wide_format` parameter is a boolean flag that determines whether the
            output should be in wide format or not. If `wide_format` is set to `True`,
            the output will have a multi-index column structure, where the first level
            represents the original columns and the second level represents the group
            names.
        fillna : int, optional
            The `fillna` parameter is used to specify the value that will be used to
            fill missing values in the resulting DataFrame. By default, it is set to 0.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        **named_funcs
            The `**named_funcs` parameter is used to specify one or more custom
            aggregation functions to apply to the data. It accepts named functions
            in the format:
        
            ``` python
                name = lambda df: df['column1'].corr(df['column2']])
            ```
        
            Where `name` is the name of the function and `df` is the DataFrame that will
            be passed to the function. The function must return a single value.
        
        
        
        Returns
        -------
        pd.DataFrame
            The function `apply_by_time` returns a pandas DataFrame object.
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
        
        df.glimpse()
        ```
        
        ```{python}
        # Apply by time with a DataFrame object
        # Allows access to multiple columns at once
        (
            df[['order_date', 'price', 'quantity']]
                .apply_by_time(
        
                    # Named apply functions
                    price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                    price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),
        
                    # Parameters
                    date_column  = 'order_date',
                    freq         = "MS",
        
                )
        )
        ```
        
        ```{python}
        # Apply by time with a GroupBy object
        (
            df[['category_1', 'order_date', 'price', 'quantity']]
                .groupby('category_1')
                .apply_by_time(
        
                    # Named functions
                    price_quantity_sum = lambda df: (df['price'] * df['quantity']).sum(),
                    price_quantity_mean = lambda df: (df['price'] * df['quantity']).mean(),
        
                    # Parameters
                    date_column  = 'order_date',
                    freq         = "MS",
        
                )
        )
        ```
        
        ```{python}
        # Return complex objects
        (
            df[['order_date', 'price', 'quantity']]
                .apply_by_time(
        
                    # Named apply functions
                    complex_object = lambda df: [df],
        
                    # Parameters
                    date_column  = 'order_date',
                    freq         = "MS",
        
                )
        )
        ```
        """
        ...
    def augment_adx(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX), +DI, and -DI for a financial time series to determine strength of trend.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input pandas DataFrame or GroupBy object with time series data.
        date_column : str
            Column name containing dates or timestamps.
        high_column : str
            Column name with high prices.
        low_column : str
            Column name with low prices.
        close_column : str
            Column name with closing prices.
        periods : Union[int, Tuple[int, int], List[int]], optional
            Number of periods for ADX calculation. Accepts int, tuple (start, end), or list. Default is 14.
        reduce_memory : bool, optional
            If True, reduces memory usage before calculation. Default is False.
        engine : str, optional
            Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - {close_column}_plus_di_{period}: Positive Directional Indicator (+DI)
            - {close_column}_minus_di_{period}: Negative Directional Indicator (-DI)
            - {close_column}_adx_{period}: Average Directional Index (ADX)
        
        Notes
        -----
        - The ADX is a trend strength indicator that ranges from 0 to 100.
        - A high ADX value indicates a strong trend, while a low ADX value indicates a weak trend.
        - The +DI and -DI values range from 0 to 100.
        - The ADX is calculated as the average of the DX values over the specified period.
        - The DX value is calculated as 100 * |(+DI - -DI)| / (+DI + -DI).
        - The True Range (TR) is the maximum of the following:
            - High - Low
            - High - Previous Close
            - Low - Previous Close
        - The +DM is calculated as follows:
            - If High - Previous High > Previous Low - Low, then +DM = max(High - Previous High, 0)
            - Otherwise, +DM = 0
        - The -DM is calculated as follows:
            - If Previous Low - Low > High - Previous High, then -DM = max(Previous Low - Low, 0)
            - Otherwise, -DM = 0
        
        References:
        
        - https://www.investopedia.com/terms/a/adx.asp
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Example 1 - Single stock ADX with pandas engine
        adx_df = (
            df.query("symbol == 'AAPL'")
            .augment_adx(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                periods=[14, 28]
            )
        )
        adx_df.head()
        ```
        
        ```{python}
        # Example 2 - Multiple stocks with groupby using pandas engine
        adx_df = (
            df.groupby('symbol')
            .augment_adx(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                periods=14
            )
        )
        adx_df.groupby('symbol').tail(1)
        ```
        
        ```{python}
        # Example 3 - Single stock ADX with polars engine
        adx_df = (
            df.query("symbol == 'AAPL'")
            .augment_adx(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                periods=[14, 28],
                engine='polars'
            )
        )
        adx_df.head()
        ```
        
        ```{python}
        # Example 4 - Multiple stocks with groupby using polars engine
        adx_df = (
            df.groupby('symbol')
            .augment_adx(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                periods=14,
                engine='polars'
            )
        )
        adx_df.groupby('symbol').tail(1)
        ```
        """
        ...
    def augment_atr(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `augment_atr` function is used to calculate Average True Range (ATR) and
        Normalized Average True Range (NATR) for a given dataset and return
        the augmented dataset.
        Set the `normalize` parameter to `True` to calculate NATR.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object. It contains the data on which the Bollinger Bands will be calculated.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the dates.
        high_column : str
            The `high_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the high prices of the asset.
        low_column : str
            The `low_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the low prices of the asset.
        close_column : str
            The `close_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the closing prices of the asset.
        periods : Union[int, Tuple[int, int], List[int]], optional
            The `periods` parameter in the `augment_atr` function can be specified as an integer, a tuple,
            or a list. This parameter specifies the number of rolling periods to use when calculating the ATR.
        normalize : bool, optional
            The `normalize` parameter is a boolean flag that indicates whether or not to normalize the ATR
            values. If set to `True`, the function will normalize the ATR values to express this volatility as a percentage of
            the closing price.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
            usage of the input data before performing the calculation. If set to `True`, the function will
            attempt to reduce the memory usage of the input data using techniques such as downcasting numeric
            columns and converting object columns
        engine : str, optional
            The `engine` parameter specifies the computation engine to use for calculating the Bollinger Bands.
            It can take two values: 'pandas' or 'polars'. If 'pandas' is selected, the function will use the
            pandas library for computation. If 'polars' is selected,
        
        Returns
        -------
        pd.DataFrame
            The function `augment_atr` returns a pandas DataFrame.
        
        Notes
        -----
        
        ## ATR (Average True Range)
        
        The Average True Range (ATR) is a technical analysis indicator used to measure market volatility. It was introduced by J. Welles Wilder Jr. in his 1978 book "New Concepts in Technical Trading Systems."
        
        The ATR is calculated as follows:
        
        1. True Range: For each period (typically a day), the True Range is the greatest of the following:
        
            - The current high minus the current low.
            - The absolute value of the current high minus the previous close.
            - The absolute value of the current low minus the previous close.
        
        2. Average True Range: The ATR is an average of the True Range over a specified number of periods (commonly 14 days).
        
        ## NATR (Normalized Average True Range)
        
        The NATR (Normalized Average True Range) is a variation of the ATR that normalizes the ATR values to express this volatility as a percentage of the closing price.
        
        The NATR (`normalize = True`) is calculated as follows:
        NATR = (ATR / Close) * 100
        
        
        Examples
        --------
        
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df
        ```
        
        ``` {python}
        # ATR pandas engine
        df_atr = (
            df
                .groupby('symbol')
                .augment_atr(
                    date_column = 'date',
                    high_column='high',
                    low_column='low',
                    close_column='close',
                    periods = [14, 28],
                    normalize = False, # True for NATR
                    engine = "pandas"
                )
        )
        
        df_atr.glimpse()
        ```
        
        ``` {python}
        # ATR polars engine
        df_atr = (
            df
                .groupby('symbol')
                .augment_atr(
                    date_column = 'date',
                    high_column='high',
                    low_column='low',
                    close_column='close',
                    periods = [14, 28],
                    normalize = False, # True for NATR
                    engine = "polars"
                )
        )
        
        df_atr.glimpse()
        ```
        """
        ...
    def augment_bbands(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `augment_bbands` function is used to calculate Bollinger Bands for a given dataset and return
        the augmented dataset.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object. It contains the data on which the Bollinger Bands will be calculated.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the dates.
        close_column : str
            The `close_column` parameter is a string that specifies the name of the column in the `data`
            DataFrame that contains the closing prices of the asset.
        periods : Union[int, Tuple[int, int], List[int]], optional
            The `periods` parameter in the `augment_bbands` function can be specified as an integer, a tuple,
            or a list. This parameter specifies the number of rolling periods to use when calculating the Bollinger Bands.
        std_dev : float, optional
            The `std_dev` parameter is a float that represents the number of standard deviations to use
            when calculating the Bollinger Bands. Bollinger Bands are a technical analysis tool that consists of
            a middle band (usually a simple moving average) and an upper and lower band that are typically two
            standard deviations away from the middle band. The `std_dev` parameter specifies the number of standard deviations. `std_dev` can be a list of floats as well.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
            usage of the input data before performing the calculation. If set to `True`, the function will
            attempt to reduce the memory usage of the input data using techniques such as downcasting numeric
            columns and converting object columns
        engine : str, optional
            The `engine` parameter specifies the computation engine to use for calculating the Bollinger Bands.
            It can take two values: 'pandas' or 'polars'. If 'pandas' is selected, the function will use the
            pandas library for computation. If 'polars' is selected,
        
        Returns
        -------
        pd.DataFrame
            The function `augment_bbands` returns a pandas DataFrame.
        
        Notes
        -----
        
        Bollinger Bands are a technical analysis tool developed by John
        Bollinger in the 1980s. They are used to measure the
        'volatility' of a stock price or other financial instrument.
        This indicator consists of three lines which are plotted in
        relation to an asset's price:
        
        1. The Middle Band: This is typically a simple moving average
        (SMA) of the closing prices over a certain number of days
        (commonly 20 days).
        
        2. The Upper Band: This is set a specified number of standard
        deviations (usually two) above the middle band.
        
        3. The Lower Band: This is set the same number of standard
        deviations (again, usually two) below the middle band.
        
        Volatility Indicator: The width of the bands is a measure of
        volatility. When the bands widen, it indicates increased
        volatility, and when they contract, it suggests decreased
        volatility.
        
        Overbought and Oversold Conditions: Prices are considered
        overbought near the upper band and oversold near the lower
        band. However, these conditions do not necessarily signal a
        reversal; prices can remain overbought or oversold for extended
        periods during strong trends.
        
        
        Examples
        --------
        
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df
        ```
        
        ``` {python}
        # BBANDS pandas engine
        df_bbands = (
            df
                .groupby('symbol')
                .augment_bbands(
                    date_column = 'date',
                    close_column='close',
                    periods = [20, 40],
                    std_dev = 2,
                    engine = "pandas"
                )
        )
        
        df_bbands.glimpse()
        ```
        
        ``` {python}
        # BBANDS polars engine
        df_bbands = (
            df
                .groupby('symbol')
                .augment_bbands(
                    date_column = 'date',
                    close_column='close',
                    periods = [20, 40],
                    std_dev = 2,
                    engine = "polars"
                )
        )
        
        df_bbands.glimpse()
        ```
        """
        ...
    def augment_cmo(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `augment_cmo` function calculates the Chande Momentum Oscillator (CMO) for a given financial
        instrument using either pandas or polars engine, and returns the augmented DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object. It contains the data on which the Chande Momentum  Oscillator (CMO) will be
            calculated.
        date_column : str
            The name of the column in the data that contains the dates or timestamps.
        close_column : str
            The `close_column` parameter is used to specify the column in the input data that contain the
            values on which the CMO will be calculated.
        periods : Union[int, Tuple[int, int], List[int]], optional
            The `periods` parameter in the `augment_cmo` function specifies the number of rolling periods over which
            the Chande Momentum Oscillator (CMO) is calculated. It can be provided as an integer, a tuple of two
            integers (start and end periods), or a list of integers.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
            usage of the data before performing the CMO calculation. If set to `True`, the function will attempt
            to reduce the memory usage of the input data. If set to `False`, the function will not attempt to reduce the memory usage of the input data.
        engine : str, optional
            The `engine` parameter specifies the computation engine to use for calculating the Chande Momentum
            Oscillator (CMO). It can take two values: 'pandas' or 'polars'.
        
        Returns
        -------
        pd.DataFrame
            The function `augment_cmo` returns a pandas DataFrame that contains the augmented data with the
            Chande Momentum Oscillator (CMO) values added.
        
        Notes
        -----
        The Chande Momentum Oscillator (CMO), developed by Tushar Chande, is a technical analysis tool used to gauge the momentum of a financial instrument. It is similar to other momentum indicators like the Relative Strength Index (RSI), but with some distinct characteristics. Here's what the CMO tells us:
        
        Momentum of Price Movements:
        
        The CMO measures the strength of trends in price movements. It calculates the difference between the sum of gains and losses over a specified period, normalized to oscillate between -100 and +100.
        Overbought and Oversold Conditions:
        
        Values close to +100 suggest overbought conditions, indicating that the price might be too high and could reverse.
        Conversely, values near -100 suggest oversold conditions, implying that the price might be too low and could rebound.
        Trend Strength:
        
        High absolute values (either positive or negative) indicate strong trends, while values near zero suggest a lack of trend or a weak trend.
        Divergences:
        
        Divergences between the CMO and price movements can be significant. For example, if the price is making new highs but the CMO is declining, it may indicate weakening momentum and a potential trend reversal.
        Crossing the Zero Line:
        
        When the CMO crosses above zero, it can be seen as a bullish signal, whereas a cross below zero can be interpreted as bearish.
        Customization:
        
        The period over which the CMO is calculated can be adjusted. A shorter period makes the oscillator more sensitive to price changes, suitable for short-term trading. A longer period smooths out the oscillator for a longer-term perspective.
        It's important to note that while the CMO can provide valuable insights into market momentum and potential price reversals, it is most effective when used in conjunction with other indicators and analysis methods. Like all technical indicators, the CMO should not be used in isolation but rather as part of a comprehensive trading strategy.
        
        References:
        1. https://www.fmlabs.com/reference/default.htm?url=CMO.htm
        
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        df
        
        # Example 1 - Calculate CMO for a single column
        cmo_df = (
            df
                .query("symbol == 'AAPL'")
                .augment_cmo(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28]
                )
        )
        cmo_df
        ```
        
        ``` {python}
        # Example 2 - Calculate CMO for multiple groups
        cmo_df = (
            df
                .groupby('symbol')
                .augment_cmo(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28]
                )
        )
        cmo_df.groupby('symbol').tail(1)
        
        ```
        
        ```{python}
        # Example 3 - Calculate CMO for polars engine
        cmo_df = (
            df
                .query("symbol == 'AAPL'")
                .augment_cmo(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28],
                    engine='polars'
                )
        )
        cmo_df
        ```
        
        ```{python}
        # Example 4 - Calculate CMO for polars engine and groups
        cmo_df = (
            df
                .groupby('symbol')
                .augment_cmo(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28],
                    engine='polars'
                )
        )
        cmo_df.groupby('symbol').tail(1)
        ```
        """
        ...
    def augment_diffs(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds differences and percentage difference (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.
        
        The `augment_diffs` function takes a Pandas DataFrame or GroupBy object, a
        date column, a value column or list of value columns, and a period or list of
        periods, and adds differenced versions of the value columns to the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object
            that you want to add differenced columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to sort the data before adding the differenced values.
        value_column : str or list
            The `value_column` parameter is the column(s) in the DataFrame that you
            want to add differences values for. It can be either a single column name
            (string) or a list of column names.
        periods : int or tuple or list, optional
            The `periods` parameter is an integer, tuple, or list that specifies the
            periods to shift values when differencing.
        
            - If it is an integer, the function will add that number of differences
              values for each column specified in the `value_column` parameter.
        
            - If it is a tuple, it will generate differences from the first to the second
              value (inclusive).
        
            - If it is a list, it will generate differences based on the values in the list.
        normalize : bool, optional
            The `normalize` parameter is used to specify whether to normalize the
            differenced values as a percentage difference. Default is False.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting differences. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for augmenting diffs. This can be faster than using "pandas" for large
              datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with differenced columns added to it.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('m4_daily', parse_dates=['date'])
        df
        ```
        
        ```{python}
        # Example 1 - Add 7 differenced values for a single DataFrame object, pandas engine
        diffed_df_single = (
            df
                .query('id == "D10"')
                .augment_diffs(
                    date_column='date',
                    value_column='value',
                    periods=(1, 7),
                    engine='pandas'
                )
        )
        diffed_df_single.glimpse()
        ```
        ```{python}
        # Example 2 - Add a single differenced value of 2 for each GroupBy object, polars engine
        diffed_df = (
            df
                .groupby('id')
                .augment_diffs(
                    date_column='date',
                    value_column='value',
                    periods=2,
                    engine='polars'
                )
        )
        diffed_df
        ```
        
        ```{python}
        # Example 3 add 2 differenced values, 2 and 4, for a single DataFrame object, pandas engine
        diffed_df_single_two = (
            df
                .query('id == "D10"')
                .augment_diffs(
                    date_column='date',
                    value_column='value',
                    periods=[2, 4],
                    engine='pandas'
                )
        )
        diffed_df_single_two
        ```
        """
        ...
    def augment_drawdown(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The augment_drawdown function calculates the drawdown metrics for a financial time series
        using either pandas or polars engine, and returns the augmented DataFrame with peak value,
        drawdown, and drawdown percentage columns.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data can be either a pandas DataFrame or a pandas DataFrameGroupBy object
            containing the time series data for drawdown calculation.
        date_column : str
            The name of the column containing dates or timestamps.
        close_column : str
            The column containing the values (e.g., price) to calculate drawdowns from.
        reduce_memory : bool, optional
            If True, reduces memory usage of the DataFrame before calculation. Default is False.
        engine : str, optional
            The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame augmented with three columns:
            - {close_column}_peak: Running maximum value up to each point
            - {close_column}_drawdown: Absolute difference from peak to current value
            - {close_column}_drawdown_pct: Percentage decline from peak to current value
        
        Notes
        -----
        Drawdown is a measure of peak-to-trough decline in a time series, typically used to assess
        the risk of a financial instrument:
        
        - Peak Value: The highest value observed up to each point in time
        - Drawdown: The absolute difference between the peak and current value
        - Drawdown Percentage: The percentage decline from the peak value
        
        Examples
        --------
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Single stock drawdown
        dd_df = (
            df.query("symbol == 'AAPL'")
            .augment_drawdown(
                date_column='date',
                close_column='close',
            )
        )
        dd_df.head()
        ```
        
        ``` {python}
        dd_df.groupby('symbol').plot_timeseries('date', 'close_drawdown_pct')
        ```
        
        ``` {python}
        # Multiple stocks with groupby
        dd_df = (
            df.groupby('symbol')
            .augment_drawdown(
                date_column='date',
                close_column='close',
                engine='polars'
            )
        )
        dd_df.head()
        ```
        
        ``` {python}
        dd_df.groupby('symbol').plot_timeseries('date', 'close_drawdown_pct')
        ```
        """
        ...
    def augment_ewm(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Add Exponential Weighted Moving (EWM) window functions to a DataFrame or
        GroupBy object.
        
        The `augment_ewm` function applies Exponential Weighted Moving (EWM) window
        functions to specified value columns of a DataFrame and adds the results as
        new columns.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input DataFrame or GroupBy object.
        date_column : str
            The name of the column containing date information in the input
            DataFrame or GroupBy object.
        value_column : Union[str, list]
            The `value_column` parameter is used to specify the column(s) on which
            the Exponential Weighted Moving (EWM) calculations will be performed. It
            can be either a string or a list of strings, representing the name(s) of
            the column(s) in the input DataFrame or GroupBy
        window_func : Union[str, list], optional
            The `window_func` parameter is used to specify the Exponential Weighted
            Moving (EWM) window function(s) to apply. It can be a string or a list
            of strings. The possible values are:
        
            - 'mean': Calculate the exponentially weighted mean.
            - 'median': Calculate the exponentially weighted median.
            - 'std': Calculate the exponentially weighted standard deviation.
            - 'var': Calculate the exponentially weighted variance.
        
        alpha : float
            The `alpha` parameter is a float that represents the smoothing factor
            for the Exponential Weighted Moving (EWM) window function. It controls
            the rate at which the weights decrease exponentially as the data points
            move further away from the current point.
        **kwargs:
            Additional arguments that are directly passed to the pandas EWM method.
            For more details, refer to the "Notes" section below.
        
        Returns
        -------
        pd.DataFrame
            The function `augment_ewm` returns a DataFrame augmented with the
            results of the Exponential Weighted Moving (EWM) calculations.
        
        Notes
        ------
        Any additional arguments provided through **kwargs are directly passed
        to the pandas EWM method. These arguments can include parameters like
        'com', 'span', 'halflife', 'ignore_na', 'adjust' and more.
        
        For a comprehensive list and detailed description of these parameters:
        
        - Refer to the official pandas documentation:
            https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html
        
        - Or, within an interactive Python environment, use:
            `?pandas.DataFrame.ewm` to display the method's docstring.
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        from pytimetk import augment_ewm
        import pandas as pd
        import numpy as np
        
        df = tk.load_dataset("m4_daily", parse_dates = ['date'])
        ```
        
        ```{python}
        # This example demonstrates the use of string-named functions on an EWM.
        # The decay parameter used in this example is 'alpha', but other methods
        #  (e.g., 'com', 'span', 'halflife') can also be utilized.
        
        ewm_df = (
            df
                .groupby('id')
                .augment_ewm(
                    date_column = 'date',
                    value_column = 'value',
                    window_func = [
                        'mean',
                        'std',
                    ],
                    alpha = 0.1,
                )
        )
        display(ewm_df)
        ```
        """
        ...
    def augment_ewma_volatility(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility for a financial time series.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input pandas DataFrame or GroupBy object with time series data.
        date_column : str
            Column name containing dates or timestamps.
        close_column : str
            Column name with closing prices to calculate volatility.
        decay_factor : float, optional
            Smoothing factor (lambda) for EWMA, between 0 and 1. Higher values give more weight to past data. Default is 0.94 (RiskMetrics standard).
        window : Union[int, Tuple[int, int], List[int]], optional
            Size of the rolling window to initialize EWMA calculation. For each window value the EWMA volatility is only computed when at least that many observations are available.
            You may provide a single integer or multiple values (via tuple or list). Default is 20.
        reduce_memory : bool, optional
            If True, reduces memory usage before calculation. Default is False.
        engine : str, optional
            Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - {close_column}_ewma_vol_{window}_{decay_factor}: EWMA volatility calculated using a minimum number of periods equal to each specified window.
        
        Notes
        -----
        EWMA volatility emphasizes recent price movements and is computed recursively as:
        
            σ²_t = (1 - λ) * r²_t + λ * σ²_{t-1}
        
        where r_t is the log return. By using the `min_periods` (set to the provided window value) we ensure that the EWMA is only calculated after enough observations have accumulated.
        
        References:
        
        - https://www.investopedia.com/articles/07/ewma.asp
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        # Example 1 - Calculate EWMA volatility for a single stock
        
        df.query("symbol == 'AAPL'").augment_ewma_volatility(
            date_column='date',
            close_column='close',
            decay_factor=0.94,
            window=[20, 50]
        ).glimpse()
        ```
        
        ```{python}
        # Example 2 - Calculate EWMA volatility for multiple stocks
        df.groupby('symbol').augment_ewma_volatility(
            date_column='date',
            close_column='close',
            decay_factor=0.94,
            window=[20, 50]
        ).glimpse()
        ```
        
        ```{python}
        # Example 3 - Calculate EWMA volatility using Polars engine
        df.query("symbol == 'AAPL'").augment_ewma_volatility(
            date_column='date',
            close_column='close',
            decay_factor=0.94,
            window=[20, 50],
            engine='polars'
        ).glimpse()
        ```
        
        ```{python}
        # Example 4 - Calculate EWMA volatility for multiple stocks using Polars engine
        
        df.groupby('symbol').augment_ewma_volatility(
            date_column='date',
            close_column='close',
            decay_factor=0.94,
            window=[20, 50],
            engine='polars'
        ).glimpse()
        ```
        """
        ...
    def augment_expanding(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply one or more Series-based expanding functions to one or more columns of a DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
        date_column : str
            Name of the datetime column. Data is sorted by this column within each group.
        value_column : Union[str, list]
            Column(s) to which the expanding window functions should be applied. Can be
            a single column name or a list.
        window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
            The `window_func` parameter in the `augment_expanding` function specifies
            the function(s) to be applied to the expanding windows of the value column(s).
        
            1. It can be either:
                - A string representing the name of a standard function (e.g., 'mean', 'sum').
        
            2. For custom functions:
                - Provide a list of tuples. Each tuple should contain a custom name for
                  the function and the function itself.
                - Each custom function should accept a Pandas Series as its input and
                  operate on that series. Example: ("range", lambda x: x.max() - x.min())
        
            (See more Examples below.)
        
            Note: If your function needs to operate on multiple columns (i.e., it
                  requires access to a DataFrame rather than just a Series), consider
                  using the `augment_expanding_apply` function in this library.
        min_periods : int, optional, default None
            Minimum observations in the window to have a value. Defaults to the window
            size. If set, a value will be produced even if fewer observations are
            present than the window size.
        engine : str, optional, default 'pandas'
            Specifies the backend computation library for augmenting expanding window
            functions.
        
            The options are:
                - "pandas" (default): Uses the `pandas` library.
                - "polars": Uses the `polars` library, which may offer performance
                   benefits for larger datasets.
        threads : int, optional, default 1
            Number of threads to use for parallel processing. If `threads` is set to
            1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
        show_progress : bool, optional, default True
            If `True`, a progress bar will be displayed during parallel processing.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        **kwargs : additional keyword arguments
            Additional arguments passed to the `pandas.Series.expanding` method when
            using the Pandas engine.
        
        Returns
        -------
        pd.DataFrame
            The `augment_expanding` function returns a DataFrame with new columns for
            each applied function, window size, and value column.
        
        Notes
        -----
        
        ## Performance
        
        ### Polars Engine (3X faster than Pandas)
        
        In most cases, the `polars` engine will be faster than the `pandas` engine. Speed tests indicate 3X or more.
        
        ### Parallel Processing (Pandas Engine Only)
        
        This function uses parallel processing to speed up computation for large
        datasets with many time series groups:
        
        Parallel processing has overhead and may not be faster on small datasets.
        
        To use parallel processing, set `threads = -1` to use all available processors.
        
        Examples
        --------
        
        ```{python}
        # Example 1 - Pandas Backend for Expanding Window Functions
        # This example demonstrates the use of string-named functions
        # on an expanding window using the Pandas backend for computations.
        
        import pytimetk as tk
        import pandas as pd
        import numpy as np
        
        df = tk.load_dataset("m4_daily", parse_dates = ['date'])
        
        expanded_df = (
            df
                .groupby('id')
                .augment_expanding(
                    date_column = 'date',
                    value_column = 'value',
                    window_func = [
                        'mean',  # Built-in mean function
                        'std',   # Built-in standard deviation function,
                         ('quantile_75', lambda x: pd.Series(x).quantile(0.75)),  # Custom quantile function
        
                    ],
                    min_periods = 1,
                    engine = 'pandas',  # Utilize pandas for the underlying computations
                    threads = 1,  # Disable parallel processing
                    show_progress = True,  # Display a progress bar
                    )
        )
        display(expanded_df)
        ```
        
        
        ```{python}
        # Example 2 - Polars Backend for Expanding Window Functions using Built-Ins
        #             (538X Faster than Pandas)
        #  This example demonstrates the use of string-named functions and configurable
        #  functions using the Polars backend for computations. Configurable functions,
        #  like pl_quantile, allow the use of specific parameters associated with their
        #  corresponding polars.Expr.rolling_<function_name> method.
        #  For instance, pl_quantile corresponds to polars.Expr.rolling_quantile.
        
        import pytimetk as tk
        import pandas as pd
        import polars as pl
        import numpy as np
        from pytimetk.utils.polars_helpers import pl_quantile
        from pytimetk.utils.pandas_helpers import pd_quantile
        
        df = tk.load_dataset("m4_daily", parse_dates = ['date'])
        
        expanded_df = (
            df
                .groupby('id')
                .augment_expanding(
                    date_column = 'date',
                    value_column = 'value',
                    window_func = [
                        'mean',  # Built-in mean function
                        'std',   # Built-in std function
                        ('quantile_75', pl_quantile(quantile=0.75)),  # Configurable with all parameters found in polars.Expr.rolling_quantile
                    ],
                    min_periods = 1,
                    engine = 'polars',  # Utilize Polars for the underlying computations
                )
        )
        display(expanded_df)
        ```
        
        ```{python}
        # Example 3 - Lambda Functions for Expanding Window Functions are faster in Pandas than Polars
        # This example demonstrates the use of lambda functions of the form lambda x: x
        # Identity lambda functions, while convenient, have signficantly slower performance.
        # When using lambda functions the Pandas backend will likely be faster than Polars.
        
        import pytimetk as tk
        import pandas as pd
        import numpy as np
        
        df = tk.load_dataset("m4_daily", parse_dates = ['date'])
        
        expanded_df = (
            df
                .groupby('id')
                .augment_expanding(
                    date_column = 'date',
                    value_column = 'value',
                    window_func = [
        
                        ('range', lambda x: x.max() - x.min()),  # Identity lambda function: can be slower, especially in Polars
                    ],
                    min_periods = 1,
                    engine = 'pandas',  # Utilize pandas for the underlying computations
                )
        )
        display(expanded_df)
        ```
        """
        ...
    def augment_expanding_apply(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply one or more DataFrame-based expanding functions to one or more columns of a DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
        date_column : str
            Name of the datetime column. Data is sorted by this column within each group.
        window_func : Union[Tuple[str, Callable], List[Tuple[str, Callable]]]
            The `window_func` parameter in the `augment_expanding_apply` function
            specifies the function(s) that operate on a expanding window with the
            consideration of multiple columns.
        
            The specification can be:
            - A tuple where the first element is a string representing the function's name and the second element is the callable function itself.
            - A list of such tuples for multiple functions.
        
            Note: For functions targeting only a single value column without the need for
            contextual data from other columns, consider using the `augment_expanding`
            function in this library.
        min_periods : int, optional, default None
            Minimum observations in the window to have a value. Defaults to the window
            size. If set, a value will be produced even if fewer observations are
            present than the window size.
        threads : int, optional, default 1
            Number of threads to use for parallel processing. If `threads` is set to
            1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
        show_progress : bool, optional, default True
            If `True`, a progress bar will be displayed during parallel processing.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        
        
        Returns
        -------
        pd.DataFrame
            The `augment_expanding` function returns a DataFrame with new columns
            for each applied function, window size, and value column.
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        import pandas as pd
        import numpy as np
        ```
        
        ```{python}
        # Example showcasing the expanding correlation between two columns (`value1` and
        # `value2`).
        # The correlation requires both columns as input.
        
        # Sample DataFrame with id, date, value1, and value2 columns.
        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'value1': [10, 20, 29, 42, 53, 59],
            'value2': [2, 16, 20, 40, 41, 50],
        })
        
        # Compute the expanding correlation for each group of 'id'
        expanding_df = (
            df.groupby('id')
              .augment_expanding_apply(
                date_column='date',
                window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],  # Lambda function for correlation
                threads = 1,  # Disable parallel processing
            )
        )
        display(expanding_df)
        ```
        
        ```{python}
        # expanding Regression Example: Using `value1` as the dependent variable and
        # `value2` and `value3` as the independent variables.
        # This example demonstrates how to perform a expanding regression using two
        # independent variables.
        
        # Sample DataFrame with `id`, `date`, `value1`, `value2`, and `value3` columns.
        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'value1': [10, 20, 29, 42, 53, 59],
            'value2': [5, 16, 24, 35, 45, 58],
            'value3': [2, 3, 6, 9, 10, 13]
        })
        
        # Define Regression Function to be applied on the expanding window.
        def regression(df):
        
            # Required module (scikit-learn) for regression.
            from sklearn.linear_model import LinearRegression
        
            model = LinearRegression()
            X = df[['value2', 'value3']]  # Independent variables
            y = df['value1']  # Dependent variable
            model.fit(X, y)
            ret = pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])
        
            return ret # Return intercept and slope as a Series
        
        # Compute the expanding regression for each group of `id`
        result_df = (
            df.groupby('id')
            .augment_expanding_apply(
                date_column='date',
                window_func=[('regression', regression)],
                threads = 1
            )
            .dropna()
        )
        
        # Format the results to have each regression output (slope and intercept) in
        #  separate columns.
        regression_wide_df = pd.concat(result_df['expanding_regression'].to_list(), axis=1).T
        regression_wide_df = pd.concat([result_df.reset_index(drop = True), regression_wide_df], axis=1)
        display(regression_wide_df)
        ```
        """
        ...
    def augment_fip_momentum(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate the "Frog In The Pan" (FIP) momentum metric over one or more rolling windows
        using either the pandas or polars engine, augmenting the DataFrame with FIP columns.
        
        The FIP momentum is defined as:
        
        - For `fip_method = 'original'`: FIP = Total Return * (percent of negative returns - percent of positive returns)
        - For `fip_method = 'modified'`: FIP = sign(Total Return) * (percent of positive returns - percent of negative returns)
        
        An optional parameter, `skip_window`, allows you to skip the first n periods (e.g., one month)
        to mitigate the effects of mean reversion.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input pandas DataFrame or grouped DataFrame containing time series data.
        date_column : str
            Name of the column with dates or timestamps.
        close_column : str
            Name of the column with closing prices to calculate returns.
        window : Union[int, List[int]], optional
            Size of the rolling window(s) as an integer or list of integers (default is 252).
        reduce_memory : bool, optional
            If True, reduces memory usage of the DataFrame. Default is False.
        engine : str, optional
            Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
        fip_method : str, optional
            Type of FIP calculation:
            - 'original': Original FIP calculation (default) where negative FIP indicates greater momentum.
            - 'modified': Modified FIP where positive FIP indicates greater momentum.
        skip_window : int, optional
            Number of initial periods to skip (set to NA) for each rolling calculation. Default is 0.
        
        Returns
        -------
        pd.DataFrame
            DataFrame augmented with FIP momentum columns:
        
            - {close_column}_fip_momentum_{w}: Rolling FIP momentum for each window w
        
        
        Notes
        -----
        
        - For 'original', a positive FIP may indicate inconsistency in the trend.
        - For 'modified', a positive FIP indicates stronger momentum in the direction of the trend (upward or downward).
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Single window with original FIP
        fip_df = (
            df.query("symbol == 'AAPL'")
            .augment_fip_momentum(
                date_column='date',
                close_column='close',
                window=252
            )
        )
        fip_df.tail()
        ```
        
        ```{python}
        # Multiple windows, polars engine, modified FIP
        fip_df = (
            df.groupby('symbol')
            .augment_fip_momentum(
                date_column='date',
                close_column='close',
                window=[63, 252],
                fip_method='modified',
                engine='polars'
            )
        )
        fip_df.tail()
        ```
        """
        ...
    def augment_fourier(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds Fourier transforms to a Pandas DataFrame or DataFrameGroupBy object.
        
        The `augment_fourier` function takes a Pandas DataFrame or GroupBy object, a date column, a value column or list of value columns, the number of periods for the Fourier series, and the maximum Fourier order, and adds Fourier-transformed columns to the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object that you want to add Fourier-transformed columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the column in the DataFrame that contains the dates. This column will be used to compute the Fourier transforms.
        periods : int or list, optional
            The `periods` parameter specifies how many timesteps between each peak in the fourier series. Default is 1.
        max_order : int, optional
            The `max_order` parameter specifies the maximum Fourier order to calculate. Default is 1.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting lags. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library.
            This can be faster than using "pandas" for large datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with Fourier-transformed columns added to it.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('m4_daily', parse_dates=['date'])
        
        # Example 1 - Add Fourier transforms for a single column
        fourier_df = (
            df
                .query("id == 'D10'")
                .augment_fourier(
                    date_column='date',
                    periods=[1, 7],
                    max_order=1
                )
        )
        fourier_df.head()
        
        fourier_df.plot_timeseries("date", "date_sin_1_7", x_axis_date_labels = "%B %d, %Y",)
        ```
        
        ``` {python}
        # Example 2 - Add Fourier transforms for grouped data
        fourier_df = (
            df
                .groupby("id")
                .augment_fourier(
                    date_column='date',
                    periods=[1, 7],
                    max_order=1,
                    engine= "pandas"
                )
        )
        fourier_df
        ```
        
        ``` {python}
        # Example 3 - Add Fourier transforms for grouped data
        fourier_df = (
            df
                .groupby("id")
                .augment_fourier(
                    date_column='date',
                    periods=[1, 7],
                    max_order=1,
                    engine= "polars"
                )
        )
        fourier_df
        ```
        """
        ...
    def augment_hilbert(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply the Hilbert transform to specified columns of a DataFrame or
        DataFrameGroupBy object.
        
        Signal Processing: The Hilbert transform is used in various signal processing techniques, including phase and amplitude modulation and demodulation, and in the analysis of signals with time-varying amplitude and frequency.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            Input DataFrame or DataFrameGroupBy object with one or more columns of
            real-valued signals.
        value_column : str or list
            List of column names in 'data' to which the Hilbert transform will be
            applied.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            summarizing the data. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
            for summarizing the data. This can be faster than using "pandas" for
            large datasets.
        
        Returns
        -------
        df_hilbert : pd.DataFrame
            A new DataFrame with the 2 Hilbert-transformed columns added, 1 for the
            real and 1 for imaginary (original columns are preserved).
        
        Notes
        -----
        The Hilbert transform is used in time series analysis primarily for:
        
        1. Creating Analytic Signals: Forms a complex-valued signal whose
        properties (magnitude and phase) provide valuable insights into the
        original signal's structure.
        
        2. Determining Instantaneous Phase/Frequency: Offers real-time signal
        characteristics, crucial for non-stationary signals whose properties
        change over time.
        
        3. Extracting Amplitude Envelope: Helps in identifying signal's
        amplitude variations, useful in various analysis tasks.
        
        4. Enhancing Signal Analysis: Assists in tasks like demodulation, trend
        analysis, feature extraction for machine learning, and improving
        signal-to-noise ratio, providing a deeper understanding of underlying
        patterns and trends.
        
        
        Examples
        --------
        ```{python}
        # Example 1: Using Pandas Engine on a pandas groupby object
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
        
        
        df_hilbert = (
            df
                .groupby('id')
                .augment_hilbert(
                    date_column = 'Date',
                    value_column = ['Weekly_Sales'],
                    engine = 'pandas'
                )
        )
        
        df_hilbert.head()
        ```
        
        ```{python}
        # Example 2: Using Polars Engine on a pandas groupby object
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
        df_hilbert = (
            df
                .groupby('id')
                .augment_hilbert(
                    date_column = 'Date',
                    value_column = ['Weekly_Sales'],
                    engine = 'polars'
                )
        )
        
        df_hilbert.head()
        ```
        
        ```{python}
        # Example 3: Using Polars Engine on a pandas dataframe
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('taylor_30_min', parse_dates=['date'])
        df_hilbert = (
            df
                .augment_hilbert(
                    date_column = 'date',
                    value_column = ['value'],
                    engine = 'polars'
                )
        )
        
        df_hilbert.head()
        ```
        
        ```{python}
        # Example 4: Using Polars Engine on a groupby object
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('taylor_30_min', parse_dates=['date'])
        df_hilbert_pd = (
            df
                .augment_hilbert(
                    date_column = 'date',
                    value_column = ['value'],
                    engine = 'pandas'
                )
        )
        
        df_hilbert.head()
        ```
        """
        ...
    def augment_holiday_signature(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Engineers 4 different holiday features from a single datetime for 137 countries
        and 2 financial markets.
        
        Note: Requires the `holidays` package to be installed. See
              https://pypi.org/project/holidays/ for more information.
        
        Parameters
        ----------
        data (pd.DataFrame):
            The input DataFrame.
        date_column (str or pd.Series):
            The name of the datetime-like column in the DataFrame.
        country_name (str):
            The name of the country for which to generate holiday features. Defaults
            to United States holidays, but the following countries are currently
            available and accessible by the full name or ISO code: See NOTES.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting holidays. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for augmenting holidays. This can be faster than using "pandas" for
              large datasets.
        
        Returns
        -------
        pd.DataFrame:
            A pandas DataFrame with three holiday-specific features:
            - is_holiday: (0, 1) indicator for holiday
            - before_holiday: (0, 1) indicator for day before holiday
            - after_holiday: (0, 1) indicator for day after holiday
            - holiday_name: name of the holiday
        
        Notes
        -----
        
        Any of the following are acceptable keys for `country_name`:
        
        | Available Countries                       | Full Country                     | Code |
        |:-----------------------------------------:|:--------------------------------:|:----:|
        | Albania                                   | Albania                          | AL   |
        | Algeria                                   | Algeria                          | DZ   |
        | American Samoa                            | AmericanSamoa                    | AS   |
        | Andorra                                   | Andorra                          | AD   |
        | Angola                                    | Angola                           | AO   |
        | Argentina                                 | Argentina                        | AR   |
        | Armenia                                   | Armenia                          | AM   |
        | Aruba                                     | Aruba                            | AW   |
        | Australia                                 | Australia                        | AU   |
        | Austria                                   | Austria                          | AT   |
        | Azerbaijan                                | Azerbaijan                       | AZ   |
        | Bahrain                                   | Bahrain                          | BH   |
        | Bangladesh                                | Bangladesh                       | BD   |
        | Barbados                                  | Barbados                         | BB   |
        | Belarus                                   | Belarus                          | BY   |
        | Belgium                                   | Belgium                          | BE   |
        | Belize                                    | Belize                           | BZ   |
        | Bolivia                                   | Bolivia                          | BO   |
        | Bosnia and Herzegovina                    | BosniaandHerzegovina             | BA   |
        | Botswana                                  | Botswana                         | BW   |
        | Brazil                                    | Brazil                           | BR   |
        | Brunei                                    | Brunei                           | BN   |
        | Bulgaria                                  | Bulgaria                         | BG   |
        | Burkina Faso                              | BurkinaFaso                      | BF   |
        | Burundi                                   | Burundi                          | BI   |
        | Laos                                      | Laos                             | LA   |
        | Latvia                                    | Latvia                           | LV   |
        | Lesotho                                   | Lesotho                          | LS   |
        | Liechtenstein                             | Liechtenstein                    | LI   |
        | Lithuania                                 | Lithuania                        | LT   |
        | Luxembourg                                | Luxembourg                       | LU   |
        | Madagascar                                | Madagascar                       | MG   |
        | Malawi                                    | Malawi                           | MW   |
        | Malaysia                                  | Malaysia                         | MY   |
        | Maldives                                  | Maldives                         | MV   |
        | Malta                                     | Malta                            | MT   |
        | Marshall Islands                          | MarshallIslands                  | MH   |
        | Mexico                                    | Mexico                           | MX   |
        | Moldova                                   | Moldova                          | MD   |
        | Monaco                                    | Monaco                           | MC   |
        | Montenegro                                | Montenegro                       | ME   |
        | Morocco                                   | Morocco                          | MA   |
        | Mozambique                                | Mozambique                       | MZ   |
        | Namibia                                   | Namibia                          | NA   |
        | Netherlands                               | Netherlands                      | NL   |
        | New Zealand                               | NewZealand                       | NZ   |
        | Nicaragua                                 | Nicaragua                        | NI   |
        | Nigeria                                   | Nigeria                          | NG   |
        | Northern Mariana Islands                  | NorthernMarianaIslands           | MP   |
        | North Macedonia                           | NorthMacedonia                   | MK   |
        | Norway                                    | Norway                           | NO   |
        | Pakistan                                  | Pakistan                         | PK   |
        | Panama                                    | Panama                           | PA   |
        | Paraguay                                  | Paraguay                         | PY   |
        | Peru                                      | Peru                             | PE   |
        | Philippines                               | Philippines                      | PH   |
        | Poland                                    | Poland                           | PL   |
        | Portugal                                  | Portugal                         | PT   |
        | Puerto Rico                               | PuertoRico                       | PR   |
        | Romania                                   | Romania                          | RO   |
        | Russia                                    | Russia                           | RU   |
        | San Marino                                | SanMarino                        | SM   |
        | Saudi Arabia                              | SaudiArabia                      | SA   |
        | Serbia                                    | Serbia                           | RS   |
        | Singapore                                 | Singapore                        | SG   |
        | Slovakia                                  | Slovakia                         | SK   |
        | Slovenia                                  | Slovenia                         | SI   |
        | South Africa                              | SouthAfrica                      | ZA   |
        | South Korea                               | SouthKorea                       | KR   |
        | Spain                                     | Spain                            | ES   |
        | Sweden                                    | Sweden                           | SE   |
        | Switzerland                               | Switzerland                      | CH   |
        | Taiwan                                    | Taiwan                           | TW   |
        | Tanzania                                  | Tanzania                         | TZ   |
        | Thailand                                  | Thailand                         | TH   |
        | Tunisia                                   | Tunisia                          | TN   |
        | Turkey                                    | Turkey                           | TR   |
        | Ukraine                                   | Ukraine                          | UA   |
        | United Arab Emirates                      | UnitedArabEmirates               | AE   |
        | United Kingdom                            | UnitedKingdom                    | GB   |
        | United States Minor Outlying Islands      | UnitedStatesMinorOutlyingIslands | UM   |
        | United States of America                  | UnitedStatesofAmerica            | US   |
        | United States Virgin Islands              | UnitedStatesVirginIslands        | VI   |
        | Uruguay                                   | Uruguay                          | UY   |
        | Uzbekistan                                | Uzbekistan                       | UZ   |
        | Vanuatu                                   | Vanuatu                          | VU   |
        | Vatican City                              | VaticanCity                      | VA   |
        | Venezuela                                 | Venezuela                        | VE   |
        | Vietnam                                   | Vietnam                          | VN   |
        | Virgin Islands (U.S.)                     | VirginIslandsUS                  | VI   |
        | Zambia                                    | Zambia                           | ZM   |
        | Zimbabwe                                  | Zimbabwe                         | ZW   |
        
        
        These are the Available Financial Markets:
        
        | Available Financial Markets  | Full Country           | Code |
        |:----------------------------:|:----------------------:|:----:|
        | European Central Bank        | EuropeanCentralBank    | ECB  |
        | New York Stock Exchange      | NewYorkStockExchange   | XNYS |
        
        Example
        -------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        # Make a DataFrame with a date column
        start_date = '2023-01-01'
        end_date = '2023-01-10'
        df = pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
        
        # Add holiday features for US
        tk.augment_holiday_signature(df, 'date', 'UnitedStates')
        ```
        
        ```{python}
        # Add holiday features for France
        tk.augment_holiday_signature(df, 'date', 'France')
        ```
        
        ```{python}
        # Add holiday features for France
        tk.augment_holiday_signature(df, 'date', 'France', engine='polars')
        ```
        """
        ...
    def augment_hurst_exponent(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate the Hurst Exponent on a rolling window for a financial time series. Used for detecting trends and mean-reversion.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input pandas DataFrame or GroupBy object with time series data.
        date_column : str
            Column name containing dates or timestamps.
        close_column : str
            Column name with closing prices to calculate the Hurst Exponent.
        window : Union[int, Tuple[int, int], List[int]], optional
            Size of the rolling window for Hurst Exponent calculation. Accepts int, tuple (start, end), or list. Default is 100.
        reduce_memory : bool, optional
            If True, reduces memory usage before calculation. Default is False.
        engine : str, optional
            Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - {close_column}_hurst_{window}: Hurst Exponent for each window size
        
        Notes
        -----
        The Hurst Exponent measures the long-term memory of a time series:
        
        - H < 0.5: Mean-reverting behavior
        - H ≈ 0.5: Random walk (no persistence)
        - H > 0.5: Trending or persistent behavior
        Computed using a simplified R/S analysis over rolling windows.
        
        References:
        
        - https://en.wikipedia.org/wiki/Hurst_exponent
        
        Examples:
        ---------
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Example 1 - Single stock Hurst Exponent with pandas engine
        hurst_df = (
            df.query("symbol == 'AAPL'")
            .augment_hurst_exponent(
                date_column='date',
                close_column='close',
                window=[100, 200]
            )
        )
        hurst_df.glimpse()
        ```
        
        ``` {python}
        # Example 2 - Multiple stocks with groupby using pandas engine
        hurst_df = (
            df.groupby('symbol')
            .augment_hurst_exponent(
                date_column='date',
                close_column='close',
                window=100
            )
        )
        hurst_df.glimpse()
        ```
        
        ``` {python}
        # Example 3 - Single stock Hurst Exponent with polars engine
        hurst_df = (
            df.query("symbol == 'AAPL'")
            .augment_hurst_exponent(
                date_column='date',
                close_column='close',
                window=[100, 200],
                engine='polars'
            )
        )
        hurst_df.glimpse()
        ```
        
        ``` {python}
        # Example 4 - Multiple stocks with groupby using polars engine
        hurst_df = (
            df.groupby('symbol')
            .augment_hurst_exponent(
                date_column='date',
                close_column='close',
                window=100,
                engine='polars'
            )
        )
        hurst_df.glimpse()
        ```
        """
        ...
    def augment_lags(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds lags to a Pandas DataFrame or DataFrameGroupBy object.
        
        The `augment_lags` function takes a Pandas DataFrame or GroupBy object, a
        date column, a value column or list of value columns, and a lag or list of
        lags, and adds lagged versions of the value columns to the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object
            that you want to add lagged columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to sort the data before adding the lagged values.
        value_column : str or list
            The `value_column` parameter is the column(s) in the DataFrame that you
            want to add lagged values for. It can be either a single column name
            (string) or a list of column names.
        lags : int or tuple or list, optional
            The `lags` parameter is an integer, tuple, or list that specifies the
            number of lagged values to add to the DataFrame.
        
            - If it is an integer, the function will add that number of lagged
              values for each column specified in the `value_column` parameter.
        
            - If it is a tuple, it will generate lags from the first to the second
              value (inclusive).
        
            - If it is a list, it will generate lags based on the values in the list.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting lags. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for augmenting lags. This can be faster than using "pandas" for large
              datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with lagged columns added to it.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('m4_daily', parse_dates=['date'])
        df
        ```
        
        ```{python}
        # Example 1 - Add 7 lagged values for a single DataFrame object, pandas engine
        lagged_df_single = (
            df
                .query('id == "D10"')
                .augment_lags(
                    date_column='date',
                    value_column='value',
                    lags=(1, 7),
                    engine='pandas'
                )
        )
        lagged_df_single
        ```
        ```{python}
        # Example 2 - Add a single lagged value of 2 for each GroupBy object, polars engine
        lagged_df = (
            df
                .groupby('id')
                .augment_lags(
                    date_column='date',
                    value_column='value',
                    lags=(1, 3),
                    engine='polars'
                )
        )
        lagged_df
        ```
        
        ```{python}
        # Example 3 add 2 lagged values, 2 and 4, for a single DataFrame object, pandas engine
        lagged_df_single_two = (
            df
                .query('id == "D10"')
                .augment_lags(
                    date_column='date',
                    value_column='value',
                    lags=[2, 4],
                    engine='pandas'
                )
        )
        lagged_df_single_two
        ```
        """
        ...
    def augment_leads(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds leads to a Pandas DataFrame or DataFrameGroupBy object.
        
        The `augment_leads` function takes a Pandas DataFrame or GroupBy object, a
        date column, a value column or list of value columns, and a lag or list of
        lags, and adds lagged versions of the value columns to the DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object
            that you want to add lagged columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to sort the data before adding the lagged values.
        value_column : str or list
            The `value_column` parameter is the column(s) in the DataFrame that you
            want to add lagged values for. It can be either a single column name
            (string) or a list of column names.
        leads : int or tuple or list, optional
            The `leads` parameter is an integer, tuple, or list that specifies the
            number of lead values to add to the DataFrame.
        
            - If it is an integer, the function will add that number of lead values
              for each column specified in the `value_column` parameter.
        
            - If it is a tuple, it will generate leads from the first to the second
              value (inclusive).
        
            - If it is a list, it will generate leads based on the values in the list.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting lags. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for augmenting lags. This can be faster than using "pandas" for large datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with lead columns added to it.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('m4_daily', parse_dates=['date'])
        df
        ```
        
        ```{python}
        # Example 1 - Add 7 lead values for a single DataFrame object, pandas engine
        lead_df_single = (
            df
                .query('id == "D10"')
                .augment_leads(
                    date_column='date',
                    value_column='value',
                    leads=(1, 7),
                    engine='pandas'
                )
        )
        lead_df_single
        ```
        ```{python}
        # Example 2 - Add a single lead value of 2 for each GroupBy object, polars engine
        lead_df = (
            df
                .groupby('id')
                .augment_leads(
                    date_column='date',
                    value_column='value',
                    leads=2,
                    engine='polars'
                )
        )
        lead_df
        ```
        
        ```{python}
        # Example 3 add 2 lead values, 2 and 4, for a single DataFrame object, pandas engine
        lead_df_single_two = (
            df
                .query('id == "D10"')
                .augment_leads(
                    date_column='date',
                    value_column='value',
                    leads=[2, 4],
                    engine='pandas'
                )
        )
        lead_df_single_two
        ```
        """
        ...
    def augment_macd(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate MACD for a given financial instrument using either pandas or polars engine.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Pandas DataFrame or GroupBy object containing financial data.
        date_column : str
            Name of the column containing date information.
        close_column : str
            Name of the column containing closing price data.
        fast_period : int, optional
            Number of periods for the fast EMA in MACD calculation.
        slow_period : int, optional
            Number of periods for the slow EMA in MACD calculation.
        signal_period : int, optional
            Number of periods for the signal line EMA in MACD calculation.
        reduce_memory : bool, optional
            Whether to reduce memory usage of the data before performing the calculation.
        engine : str, optional
            Computation engine to use ('pandas' or 'polars').
        
        Returns
        -------
        pd.DataFrame
            DataFrame with MACD line, signal line, and MACD histogram added.
        
        Notes
        -----
        The MACD (Moving Average Convergence Divergence) is a
        trend-following momentum indicator that shows the relationship
        between two moving averages of a security’s price. Developed by
        Gerald Appel in the late 1970s, the MACD is one of the simplest
        and most effective momentum indicators available.
        
        MACD Line: The MACD line is the difference between two
        exponential moving averages (EMAs) of a security’s price,
        typically the 12-day and 26-day EMAs.
        
        Signal Line: This is usually a 9-day EMA of the MACD line. It
        acts as a trigger for buy and sell signals.
        
        Histogram: The MACD histogram plots the difference between the
        MACD line and the signal line. A histogram above zero indicates
        that the MACD line is above the signal line (bullish), and
        below zero indicates it is below the signal line (bearish).
        
        Crossovers: The most common MACD signals are when the MACD line
        crosses above or below the signal line. A crossover above the
        signal line is a bullish signal, indicating it might be time to
        buy, and a crossover below the signal line is bearish,
        suggesting it might be time to sell.
        
        
        Examples
        --------
        
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df
        ```
        
        ``` {python}
        # MACD pandas engine
        df_macd = (
            df
                .groupby('symbol')
                .augment_macd(
                    date_column = 'date',
                    close_column = 'close',
                    fast_period = 12,
                    slow_period = 26,
                    signal_period = 9,
                    engine = "pandas"
                )
        )
        
        df_macd.glimpse()
        ```
        
        ``` {python}
        # MACD polars engine
        df_macd = (
            df
                .groupby('symbol')
                .augment_macd(
                    date_column = 'date',
                    close_column = 'close',
                    fast_period = 12,
                    slow_period = 26,
                    signal_period = 9,
                    engine = "polars"
                )
        )
        
        df_macd.glimpse()
        ```
        """
        ...
    def augment_pct_change(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds percentage difference (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object
            that you want to add percentage differenced columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to sort the data before adding the percentage differenced values.
        value_column : str or list
            The `value_column` parameter is the column(s) in the DataFrame that you
            want to add percentage differences values for. It can be either a single column name
            (string) or a list of column names.
        periods : int or tuple or list, optional
            The `periods` parameter is an integer, tuple, or list that specifies the
            periods to shift values when percentage differencing.
        
            - If it is an integer, the function will add that number of percentage differences
              values for each column specified in the `value_column` parameter.
        
            - If it is a tuple, it will generate percentage differences from the first to the second
              value (inclusive).
        
            - If it is a list, it will generate percentage differences based on the values in the list.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting percentage differences. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
            for augmenting percentage diffs. This can be faster than using "pandas" for large
            datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with percentage differenced columns added to it.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('m4_daily', parse_dates=['date'])
        df
        ```
        
        ```{python}
        # Example 1 - Add 7 pctdiff values for a single DataFrame object, pandas engine
        pctdiff_df_single = (
            df
                .query('id == "D10"')
                .augment_pct_change(
                    date_column='date',
                    value_column='value',
                    periods=(1, 7),
                    engine='pandas'
                )
        )
        pctdiff_df_single.glimpse()
        ```
        
        ```{python}
        # Example 2 - Add a single percent differenced value of 2 for each GroupBy object, polars engine
        pctdiff_df = (
            df
                .groupby('id')
                .augment_pct_change(
                    date_column='date',
                    value_column='value',
                    periods=2,
                    engine='polars'
                )
        )
        pctdiff_df
        ```
        
        ```{python}
        # Example 3 add 2 percent differenced values, 2 and 4, for a single DataFrame object, pandas engine
        pctdiff_df_single_two = (
            df
                .query('id == "D10"')
                .augment_diffs(
                    date_column='date',
                    value_column='value',
                    periods=[2, 4],
                    engine='pandas'
                )
        )
        pctdiff_df_single_two
        ```
        """
        ...
    def augment_ppo(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate PPO for a given financial instrument using either pandas or polars engine.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Pandas DataFrame or GroupBy object containing financial data.
        date_column : str
            Name of the column containing date information.
        close_column : str
            Name of the column containing closing price data.
        fast_period : int, optional
            Number of periods for the fast EMA in PPO calculation.
        slow_period : int, optional
            Number of periods for the slow EMA in PPO calculation.
        reduce_memory : bool, optional
            Whether to reduce memory usage of the data before performing the calculation.
        engine : str, optional
            Computation engine to use ('pandas' or 'polars').
        
        
        Returns
        -------
        pd.DataFrame
            DataFrame with PPO values added.
        
        Notes
        -----
        
        The Percentage Price Oscillator (PPO) is a momentum oscillator
        that measures the difference between two moving averages as a
        percentage of the larger moving average. The PPO is best used
        to confirm the direction of the price trend and gauge its
        momentum.
        
        The PPO is calculated by subtracting a long-term EMA from a
        short-term EMA, then dividing the result by the long-term EMA,
        and finally multiplying by 100.
        
        Advantages Over MACD: The PPO's percentage-based calculation
        allows for easier comparisons between different securities,
        regardless of their price levels. This is a distinct advantage
        over the MACD, which provides absolute values and can be less
        meaningful when comparing stocks with significantly different
        prices.
        
        
        Examples
        --------
        
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df
        ```
        
        ``` {python}
        # PPO pandas engine
        df_ppo = (
            df
                .groupby('symbol')
                .augment_ppo(
                    date_column = 'date',
                    close_column = 'close',
                    fast_period = 12,
                    slow_period = 26,
                    engine = "pandas"
                )
        )
        
        df_ppo.glimpse()
        ```
        
        ``` {python}
        # PPO polars engine
        df_ppo = (
            df
                .groupby('symbol')
                .augment_ppo(
                    date_column = 'date',
                    close_column = 'close',
                    fast_period = 12,
                    slow_period = 26,
                    engine = "polars"
                )
        )
        
        df_ppo.glimpse()
        ```
        """
        ...
    def augment_qsmomentum(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The function `augment_qsmomentum` calculates Quant Science Momentum for financial data.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter in the `augment_qsmomentum` function is expected to be a pandas DataFrame or a
            pandas DataFrameGroupBy object. This parameter represents the input data on which the momentum
            calculations will be performed.
        date_column : str
            The `date_column` parameter in the `augment_qsmomentum` function refers to the column in your input
            data that contains the dates associated with the financial data. This column is used for time-based
            operations and calculations within the function.
        close_column : str
            The `close_column` parameter in the `augment_qsmomentum` function refers to the column in the input
            DataFrame that contains the closing prices of the financial instrument or asset for which you want
            to calculate the momentum.
        roc_fast_period : Union[int, Tuple[int, int], List[int]], optional
            The `roc_fast_period` parameter in the `augment_qsmomentum` function determines the period used for
            calculating the fast Rate of Change (ROC) momentum indicator.
        roc_slow_period : Union[int, Tuple[int, int], List[int]], optional
            The `roc_slow_period` parameter in the `augment_qsmomentum` function represents the period used for
            calculating the slow rate of change (ROC) in momentum analysis.
        returns_period : Union[int, Tuple[int, int], List[int]], optional
            The `returns_period` parameter in the `augment_qsmomentum` function determines the period over
            which the returns are calculated.
        reduce_memory : bool, optional
            The `reduce_memory` parameter in the `augment_qsmomentum` function is a boolean flag that indicates
            whether memory reduction techniques should be applied to the input data before and after the
            momentum calculation process. If set to `True`, memory reduction methods will be used to optimize
            memory usage, potentially reducing
        engine : str, optional
            The `engine` parameter in the `augment_qsmomentum` function specifies the computation engine to be
            sed for calculating momentum. It can have two possible values: "pandas" or "polars".
        
        Returns
        -------
            The function `augment_qsmomentum` returns a pandas DataFrame that has been augmented with columns
            representing the Quant Science Momentum (QSM) calculated based on the specified parameters
            such as roc_fast_period, roc_slow_period, and returns_period.
        
        Notes
        -----
        
        The Quant Science Momentum (QSM) is a momentum indicator that is calculated based on the Slow Rate of Change (ROC) usually over a 252-day period and the Fast Rate of Change (ROC) usually over a 21-day period.
        
        The QSM is calculated as the difference between the slow and fast ROCs divided by the standard deviation of the returns over a specified period.
        
        This provides a measure of momentum that is normalized by the rolling volatility of the returns.
        
        Examples
        --------
        ``` {python}
        import pandas as pd
        import polars as pl
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df.glimpse()
        ```
        
        ``` {python}
        # PANDAS QS MOMENTUM CALCULATION
        df_qsmom = (
            df
                .query('symbol == "GOOG"')
                .augment_qsmomentum(
                    date_column = 'date',
                    close_column = 'close',
                    roc_fast_period = [1, 5, 21],
                    roc_slow_period = 252,
                    returns_period = 126,
                    engine = "pandas"
                )
        )
        
        df_qsmom.dropna().glimpse()
        ```
        
        ``` {python}
        # POLARS QS MOMENTUM CALCULATION
        df_qsmom = (
            df
                .query('symbol == "GOOG"')
                .augment_qsmomentum(
                    date_column = 'date',
                    close_column = 'close',
                    roc_fast_period = [1, 5, 21],
                    roc_slow_period = 252,
                    returns_period = 126,
                    engine = "polars"
                )
        )
        
        df_qsmom.dropna().glimpse()
        ```
        """
        ...
    def augment_regime_detection(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Detect regimes in a financial time series using a specified method (e.g., HMM).
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input pandas DataFrame or GroupBy object with time series data.
        date_column : str
            Column name containing dates or timestamps.
        close_column : str
            Column name with closing prices for regime detection.
        window : Union[int, Tuple[int, int], List[int]], optional
            Size of the rolling window to fit the regime detection model. Default is 252.
        n_regimes : int, optional
            Number of regimes to detect (e.g., 2 for bull/bear). Default is 2.
        method : str, optional
            Method for regime detection. Currently supports 'hmm'. Default is 'hmm'.
        step_size : int, optional
            Step size between HMM fits (e.g., 10 fits every 10 rows). Default is 1.
        n_iter : int, optional
            Number of iterations for HMM fitting. Default is 100.
        n_jobs : int, optional
            Number of parallel jobs for group processing (-1 uses all cores). Default is -1.
        reduce_memory : bool, optional
            If True, reduces memory usage. Default is False.
        engine : str, optional
            Computation engine: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added columns:
            - {close_column}_regime_{window}: Integer labels for detected regimes (e.g., 0, 1).
        
        Notes
        -----
        - Uses Hidden Markov Model (HMM) to identify latent regimes based on log returns.
        - Regimes reflect distinct statistical states (e.g., high/low volatility, trending).
        - Requires 'hmmlearn' package. Install with `pip install hmmlearn`.
        
        Examples
        --------
        ```python
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Example 1 - Single stock regime detection with pandas engine
        # Requires hmmlearn: pip install hmmlearn
        regime_df = (
            df.query("symbol == 'AAPL'")
            .augment_regime_detection(
                date_column='date',
                close_column='close',
                window=252,
                n_regimes=2
            )
        )
        regime_df.head().glimpse()
        ```
        
        ```python
        # Example 2 - Multiple stocks with groupby using pandas engine
        # Requires hmmlearn: pip install hmmlearn
        regime_df = (
            df.groupby('symbol')
            .augment_regime_detection(
                date_column='date',
                close_column='close',
                window=[252, 504],  # One year and two years
                n_regimes=3
            )
        )
        regime_df.groupby('symbol').tail(1).glimpse()
        ```
        
        ```python
        # Example 3 - Single stock regime detection with polars engine
        # Requires hmmlearn: pip install hmmlearn
        regime_df = (
            df.query("symbol == 'AAPL'")
            .augment_regime_detection(
                date_column='date',
                close_column='close',
                window=252,
                n_regimes=2,
                engine='polars'
            )
        )
        regime_df.glimpse()
        ```
        
        ```python
        # Example 4 - Multiple stocks with groupby using polars engine
        # Requires hmmlearn: pip install hmmlearn
        regime_df = (
            df.groupby('symbol')
            .augment_regime_detection(
                date_column='date',
                close_column='close',
                window=504,
                n_regimes=3,
                engine='polars'
            )
        )
        regime_df.groupby('symbol').tail(1).glimpse()
        ```
        """
        ...
    def augment_roc(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Adds rate of change (percentage change) to a Pandas DataFrame or DataFrameGroupBy object.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter is the input DataFrame or DataFrameGroupBy object
            that you want to add percentage differenced columns to.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to sort the data before adding the percentage differenced values.
        close_column : str
            The `close_column` parameter in the `augment_qsmomentum` function refers to the column in the input
            DataFrame that contains the closing prices of the financial instrument or asset for which you want
            to calculate the momentum.
        periods : int or tuple or list, optional
            The `periods` parameter is an integer, tuple, or list that specifies the
            periods to shift values when percentage differencing.
        
            - If it is an integer, the function will add that number of percentage differences
              values for each column specified in the `value_column` parameter.
        
            - If it is a tuple, it will generate percentage differences from the first to the second
              value (inclusive).
        
            - If it is a list, it will generate percentage differences based on the values in the list.
        start_index : int, optional
            The `start_index` parameter is an integer that specifies the starting index for the percentage difference calculation.
            Default is 0 which is the last element in the group.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is True.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting percentage differences. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
            for augmenting percentage diffs. This can be faster than using "pandas" for large
            datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with percentage differenced columns added to it.
        
        Notes
        -----
        The rate of change (ROC) calculation is a momentum indicator that measures the percentage change in price between the current price and the price a certain number of periods ago. The ROC indicator is used to identify the speed and direction of price movements. It is calculated as follows:
        
        ROC = [(Close - Close n periods ago) / (Close n periods ago)]
        
        When `start_index` is used, the formula becomes:
        
        ROC = [(Close start_index periods ago - Close n periods ago) / (Close n periods ago)]
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset("stocks_daily", parse_dates = ['date'])
        
        df.glimpse()
        ```
        
        ```{python}
        # Example 1 - Add 7 roc values for a single DataFrame object, pandas engine
        roc_df = (
            df
                .query('symbol == "GOOG"')
                .augment_roc(
                    date_column='date',
                    close_column='close',
                    periods=(1, 7),
                    engine='pandas'
                )
        )
        roc_df.glimpse()
        ```
        
        ```{python}
        # Example 2 - Add 2 ROC with start index 21 using GroupBy object, polars engine
        roc_df = (
            df
                .groupby('symbol')
                .augment_roc(
                    date_column='date',
                    close_column='close',
                    periods=[63, 252],
                    start_index=21,
                    engine='polars'
                )
        )
        roc_df
        ```
        """
        ...
    def augment_rolling(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply one or more Series-based rolling functions and window sizes to one or more columns of a DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input data to be processed. Can be a Pandas DataFrame or a GroupBy
            object.
        date_column : str
            Name of the datetime column. Data is sorted by this column within each
            group.
        value_column : Union[str, list]
            Column(s) to which the rolling window functions should be applied. Can
            be a single column name or a list.
        window_func : Union[str, list, Tuple[str, Callable]], optional, default 'mean'
            The `window_func` parameter in the `augment_rolling` function specifies
            the function(s) to be applied to the rolling windows of the value
            column(s).
        
            1. It can be either:
                - A string representing the name of a standard function (e.g.,
                  'mean', 'sum').
        
            2. For custom functions:
                - Provide a list of tuples. Each tuple should contain a custom name
                  for the function and the function itself.
                - Each custom function should accept a Pandas Series as its input
                  and operate on that series.
                  Example: ("range", lambda x: x.max() - x.min())
        
            (See more Examples below.)
        
            Note: If your function needs to operate on multiple columns (i.e., it
                  requires access to a DataFrame rather than just a Series),
                  consider using the `augment_rolling_apply` function in this library.
        window : Union[int, tuple, list], optional, default 2
            Specifies the size of the rolling windows.
            - An integer applies the same window size to all columns in `value_column`.
            - A tuple generates windows from the first to the second value (inclusive).
            - A list of integers designates multiple window sizes for each respective
              column.
        min_periods : int, optional, default None
            Minimum observations in the window to have a value. Defaults to the
            window size. If set, a value will be produced even if fewer observations
            are present than the window size.
        center : bool, optional, default False
            If `True`, the rolling window will be centered on the current value. For
            even-sized windows, the window will be left-biased. Otherwise, it uses a trailing window.
        threads : int, optional, default 1
            Number of threads to use for parallel processing. If `threads` is set to
            1, parallel processing will be disabled. Set to -1 to use all available CPU cores.
        show_progress : bool, optional, default True
            If `True`, a progress bar will be displayed during parallel processing.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
        engine : str, optional, default 'pandas'
            Specifies the backend computation library for augmenting expanding window
            functions.
        
            The options are:
                - "pandas" (default): Uses the `pandas` library.
                - "polars": Uses the `polars` library, which may offer performance
                   benefits for larger datasets.
        
        Returns
        -------
        pd.DataFrame
            The `augment_rolling` function returns a DataFrame with new columns for
            each applied function, window size, and value column.
        
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
        import pytimetk as tk
        import pandas as pd
        import numpy as np
        
        df = tk.load_dataset("m4_daily", parse_dates = ['date'])
        ```
        
        ```{python}
        # Example 1 - Using a single window size and a single function name, pandas engine
        # This example demonstrates the use of both string-named functions and lambda
        # functions on a rolling window. We specify a list of window sizes: [2,7].
        # As a result, the output will have computations for both window sizes 2 and 7.
        # Note - It's preferred to use built-in or configurable functions instead of
        # lambda functions for performance reasons.
        
        rolled_df = (
            df
                .groupby('id')
                .augment_rolling(
                    date_column = 'date',
                    value_column = 'value',
                    window = [2,7],  # Specifying multiple window sizes
                    window_func = [
                        'mean',  # Built-in mean function
                        ('std', lambda x: x.std())  # Lambda function to compute standard deviation
                    ],
                    threads = 1,  # Disabling parallel processing
                    engine = 'pandas'  # Using pandas engine
                )
        )
        display(rolled_df)
        ```
        
        ```{python}
        # Example 2 - Multiple groups, pandas engine
        # Example showcasing the use of string function names and lambda functions
        # applied on rolling windows. The `window` tuple (1,3) will generate window
        # sizes of 1, 2, and 3.
        # Note - It's preferred to use built-in or configurable functions instead of
        # lambda functions for performance reasons.
        
        rolled_df = (
            df
                .groupby('id')
                .augment_rolling(
                    date_column = 'date',
                    value_column = 'value',
                    window = (1,3),  # Specifying a range of window sizes
                    window_func = [
                        'mean',  # Using built-in mean function
                        ('std', lambda x: x.std())  # Lambda function for standard deviation
                    ],
                    threads = 1,  # Disabling parallel processing
                    engine = 'pandas'  # Using pandas engine
                )
        )
        display(rolled_df)
        ```
        
        ```{python}
        # Example 3 - Multiple groups, polars engine
        
        rolled_df = (
            df
                .groupby('id')
                .augment_rolling(
                    date_column = 'date',
                    value_column = 'value',
                    window = (1,3),  # Specifying a range of window sizes
                    window_func = [
                        'mean',  # Using built-in mean function
                        'std',  # Using built-in standard deviation function
                    ],
                    engine = 'polars'  # Using polars engine
                )
        )
        display(rolled_df)
        ```
        """
        ...
    def augment_rolling_apply(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply one or more DataFrame-based rolling functions and window sizes to one
        or more columns of a DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            Input data to be processed. Can be a Pandas DataFrame or a GroupBy object.
        date_column : str
            Name of the datetime column. Data is sorted by this column within each
            group.
        window_func : Union[Tuple[str, Callable], List[Tuple[str, Callable]]]
            The `window_func` parameter in the `augment_rolling_apply` function
            specifies the function(s) that operate on a rolling window with the
            consideration of multiple columns.
        
            The specification can be:
            - A tuple where the first element is a string representing the function's
              name and the second element is the callable function itself.
            - A list of such tuples for multiple functions.
        
            (See more Examples below.)
        
            Note: For functions targeting only a single value column without the
            need for contextual data from other columns, consider using the
            `augment_rolling` function in this library.
        window : Union[int, tuple, list], optional
            Specifies the size of the rolling windows.
            - An integer applies the same window size to all columns in `value_column`.
            - A tuple generates windows from the first to the second value (inclusive).
            - A list of integers designates multiple window sizes for each respective
              column.
        min_periods : int, optional, default None
            Minimum observations in the window to have a value. Defaults to the
            window size. If set, a value will be produced even if fewer observations
            are present than the window size.
        center : bool, optional
            If `True`, the rolling window will be centered on the current value. For
            even-sized windows, the window will be left-biased. Otherwise, it uses a
            trailing window.
        threads : int, optional, default 1
            Number of threads to use for parallel processing. If `threads` is set to
            1, parallel processing will be disabled. Set to -1 to use all available
            CPU cores.
        show_progress : bool, optional, default True
            If `True`, a progress bar will be displayed during parallel processing.
        
        Returns
        -------
        pd.DataFrame
            The `augment_rolling` function returns a DataFrame with new columns for
            each applied function, window size, and value column.
        
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
        import pytimetk as tk
        import pandas as pd
        import numpy as np
        
        # Example 1 - showcasing the rolling correlation between two columns
        # (`value1` and `value2`).
        # The correlation requires both columns as input.
        
        # Sample DataFrame with id, date, value1, and value2 columns.
        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'value1': [10, 20, 29, 42, 53, 59],
            'value2': [2, 16, 20, 40, 41, 50],
        })
        
        # Compute the rolling correlation for each group of 'id'
        # Using a rolling window of size 3 and a lambda function to calculate the
        # correlation.
        
        rolled_df = (
            df.groupby('id')
            .augment_rolling_apply(
                date_column='date',
                window=3,
                window_func=[('corr', lambda x: x['value1'].corr(x['value2']))],  # Lambda function for correlation
                center = False,  # Not centering the rolling window
                threads = 1 # Increase threads for parallel processing (use -1 for all cores)
            )
        )
        display(rolled_df)
        ```
        
        ```{python}
        # Example 2 - Rolling Regression Example: Using `value1` as the dependent
        # variable and `value2` and `value3` as the independent variables. This
        # example demonstrates how to perform a rolling regression using two
        # independent variables.
        
        # Sample DataFrame with `id`, `date`, `value1`, `value2`, and `value3` columns.
        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06']),
            'value1': [10, 20, 29, 42, 53, 59],
            'value2': [5, 16, 24, 35, 45, 58],
            'value3': [2, 3, 6, 9, 10, 13]
        })
        
        # Define Regression Function to be applied on the rolling window.
        def regression(df):
        
            # Required module (scikit-learn) for regression.
            # This import statement is required inside the function to avoid errors.
            from sklearn.linear_model import LinearRegression
        
            model = LinearRegression()
            X = df[['value2', 'value3']]  # Independent variables
            y = df['value1']  # Dependent variable
            model.fit(X, y)
            ret = pd.Series([model.intercept_, model.coef_[0]], index=['Intercept', 'Slope'])
        
            return ret # Return intercept and slope as a Series
        
        # Compute the rolling regression for each group of `id`
        # Using a rolling window of size 3 and the regression function.
        rolled_df = (
            df.groupby('id')
            .augment_rolling_apply(
                date_column='date',
                window=3,
                window_func=[('regression', regression)]
            )
            .dropna()
        )
        
        # Format the results to have each regression output (slope and intercept) in
        # separate columns.
        
        regression_wide_df = pd.concat(rolled_df['rolling_regression_win_3'].to_list(), axis=1).T
        
        regression_wide_df = pd.concat([rolled_df.reset_index(drop = True), regression_wide_df], axis=1)
        
        display(regression_wide_df)
        ```
        """
        ...
    def augment_rolling_risk_metrics(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The augment_rolling_risk_metrics function calculates rolling risk-adjusted performance
        metrics for a financial time series using either pandas or polars engine, and returns
        the augmented DataFrame with columns for Sharpe Ratio, Sortino Ratio, and other metrics.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data can be a pandas DataFrame or a pandas DataFrameGroupBy object
            containing the time series data for risk metric calculations.
        date_column : str
            The name of the column containing dates or timestamps.
        close_column : str
            The column containing closing prices to calculate returns and risk metrics from.
        window : int, optional
            The rolling window size for calculations (e.g., 252 for annual). Default is 252.
        risk_free_rate : float, optional
            The assumed risk-free rate (e.g., 0.0 for 0%). Default is 0.0.
        benchmark_column : str or None, optional
            The column containing benchmark returns (e.g., market index) for Treynor and Information Ratios.
            Default is None.
        annualization_factor : int, optional
            The factor to annualize returns and volatility (e.g., 252 for daily data). Default is 252.
        metrics : List[str] or None, optional
            The list of risk metrics to calculate. Choose from: 'sharpe_ratio', 'sortino_ratio',
            'treynor_ratio', 'information_ratio', 'omega_ratio', 'volatility_annualized',
            'skewness', 'kurtosis'. Default is None (all metrics).
        reduce_memory : bool, optional
            If True, reduces memory usage of the DataFrame before calculation. Default is False.
        engine : str, optional
            The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame augmented with columns:
            - {close_column}_sharpe_ratio_{window}: Rolling Sharpe Ratio
            - {close_column}_sortino_ratio_{window}: Rolling Sortino Ratio
            - {close_column}_treynor_ratio_{window}: Rolling Treynor Ratio (if benchmark provided)
            - {close_column}_information_ratio_{window}: Rolling Information Ratio (if benchmark provided)
            - {close_column}_omega_ratio_{window}: Rolling Omega Ratio
            - {close_column}_volatility_annualized_{window}: Rolling annualized volatility
            - {close_column}_skewness_{window}: Rolling skewness of returns
            - {close_column}_kurtosis_{window}: Rolling kurtosis of returns
        
        Notes
        -----
        This function computes returns from closing prices and calculates rolling risk metrics:
        
        - Sharpe Ratio: Excess return over risk-free rate divided by volatility
        - Sortino Ratio: Excess return over risk-free rate divided by downside deviation
        - Treynor Ratio: Excess return over risk-free rate divided by beta (requires benchmark)
        - Information Ratio: Excess return over benchmark divided by tracking error (requires benchmark)
        - Omega Ratio: Ratio of gains to losses above/below a threshold
        - Volatility: Annualized standard deviation of returns
        - Skewness: Asymmetry of return distribution
        - Kurtosis: Fat-tailedness of return distribution
        
        Examples
        --------
        ``` {python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Single stock risk metrics
        risk_df = (
            df.query("symbol == 'AAPL'")
            .augment_rolling_risk_metrics(
                date_column='date',
                close_column='adjusted',
                window=252
            )
        )
        risk_df.head()
        ```
        
        ``` {python}
        # Multiple stocks with groupby and benchmark
        risk_df = (
            df.groupby('symbol')
            .augment_rolling_risk_metrics(
                date_column='date',
                close_column='adjusted',
                # benchmark_column='market_adjusted_returns',  # Use if a benchmark returns column exists
                window=60,
                engine='polars'
            )
        )
        risk_df.head()
        ```
        
        ``` {python}
        # Selective metrics
        risk_df = (
            df.groupby('symbol')
            .augment_rolling_risk_metrics(
                date_column='date',
                close_column='adjusted',
                window=252,
                metrics=['sharpe_ratio', 'sortino_ratio', 'volatility_annualized'],
            )
        )
        risk_df.tail()
        ```
        """
        ...
    def augment_rsi(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `augment_rsi` function calculates the Relative Strength Index (RSI) for a given financial
        instrument using either pandas or polars engine, and returns the augmented DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that can be either a pandas DataFrame or a pandas
            DataFrameGroupBy object. It contains the data on which the RSI will be
            calculated.
        date_column : str
            The name of the column in the data that contains the dates or timestamps.
        close_column : str
            The `close_column` parameter is used to specify the column(s) in the input data that contain the
            values on which the RSI will be calculated. It can be either a single column name (string) or a list
            of column names (if you want to calculate RSI on multiple columns).
        periods : Union[int, Tuple[int, int], List[int]], optional
            The `periods` parameter in the `augment_rsi` function specifies the number of rolling periods over which
            the RSI is calculated. It can be provided as an integer, a tuple of two
            integers (start and end periods), or a list of integers.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is a boolean flag that indicates whether or not to reduce the memory
            usage of the data before performing the RSI calculation. If set to `True`, the function will attempt
            to reduce the memory usage of the input data. If set to `False`, the function will not attempt to reduce the memory usage of the input data.
        engine : str, optional
            The `engine` parameter specifies the computation engine to use for calculating the RSI. It can take two values: 'pandas' or 'polars'.
        
        Returns
        -------
        pd.DataFrame
            The function `augment_rsi` returns a pandas DataFrame that contains the augmented data with the
            Relative Strength Index (RSI) values added.
        
        Notes
        -----
        The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. Developed by J. Welles Wilder Jr. and introduced in his 1978 book "New Concepts in Technical Trading Systems", the RSI is one of the most well-known and widely used technical analysis indicators.
        
        - Range: The RSI oscillates between 0 and 100.
        - Overbought and Oversold Levels: Traditionally, the RSI is
        considered overbought when above 70 and oversold when below
        30. These thresholds can indicate potential reversal points
        where a security is overvalued or undervalued.
        - Divergence: RSI can also be used to identify potential
        reversals by looking for bearish and bullish divergences.
        
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        df
        
        # Example 1 - Calculate RSI for a single column
        rsi_df = (
            df
                .query("symbol == 'AAPL'")
                .augment_rsi(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28]
                )
        )
        rsi_df
        ```
        
        ``` {python}
        # Example 2 - Calculate RSI for multiple groups
        rsi_df = (
            df
                .groupby('symbol')
                .augment_rsi(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28]
                )
        )
        rsi_df.groupby('symbol').tail(1)
        
        ```
        
        ```{python}
        # Example 3 - Calculate RSI for polars engine
        rsi_df = (
            df
                .query("symbol == 'AAPL'")
                .augment_rsi(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28],
                    engine='polars'
                )
        )
        rsi_df
        ```
        
        ```{python}
        # Example 4 - Calculate RSI for polars engine and groups
        rsi_df = (
            df
                .groupby('symbol')
                .augment_rsi(
                    date_column='date',
                    close_column='adjusted',
                    periods=[14, 28],
                    engine='polars'
                )
        )
        rsi_df.groupby('symbol').tail(1)
        ```
        """
        ...
    def augment_stochastic_oscillator(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `augment_stochastic_oscillator` function calculates the Stochastic Oscillator (%K and %D)
        for a financial instrument using either pandas or polars engine, and returns the augmented DataFrame.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data can be a pandas DataFrame or a pandas DataFrameGroupBy object containing
            the time series data for Stochastic Oscillator calculations.
        date_column : str
            The name of the column containing dates or timestamps.
        high_column : str
            The column containing high prices for the financial instrument.
        low_column : str
            The column containing low prices for the financial instrument.
        close_column : str
            The column containing closing prices for the financial instrument.
        k_periods : Union[int, Tuple[int, int], List[int]], optional
            The number of periods for calculating %K (fast stochastic). Can be an integer, a tuple of
            two integers (start and end periods), or a list of integers. Default is 14.
        d_periods : int, optional
            The number of periods for calculating %D (slow stochastic), typically a moving average of %K.
            Default is 3.
        reduce_memory : bool, optional
            If True, reduces memory usage of the DataFrame before calculation. Default is False.
        engine : str, optional
            The computation engine to use: 'pandas' or 'polars'. Default is 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame augmented with columns:
            - {close_column}_stoch_k_{k_period}: Stochastic Oscillator %K for each k_period
            - {close_column}_stoch_d_{k_period}_{d_period}: Stochastic Oscillator %D for each k_period
        
        Notes
        -----
        The Stochastic Oscillator is a momentum indicator that compares a security's closing price to its
        price range over a specific period, developed by George Lane. It consists of two lines:
        
        - %K: Measures the current close relative to the high-low range over k_periods.
        - %D: A moving average of %K over d_periods, smoothing the %K line.
        
        Key interpretations:
        
        - Values above 80 indicate overbought conditions, suggesting a potential price reversal downward.
        - Values below 20 indicate oversold conditions, suggesting a potential price reversal upward.
        - Crossovers of %K and %D can signal buy/sell opportunities.
        - Divergences between price and the oscillator can indicate trend reversals.
        
        Formula:
        
        - %K = 100 * (Close - Lowest Low in k_periods) / (Highest High in k_periods - Lowest Low in k_periods)
        - %D = Moving average of %K over d_periods
        
        References:
        
        - https://www.investopedia.com/terms/s/stochasticoscillator.asp
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates=['date'])
        
        # Example 1 - Single stock stochastic oscillator
        stoch_df = (
            df.query("symbol == 'AAPL'")
            .augment_stochastic_oscillator(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                k_periods=[14, 28],
                d_periods=3
            )
        )
        stoch_df.head()
        ```
        
        ``` {python}
        # Example 2 - Multiple stocks with groupby
        stoch_df = (
            df.groupby('symbol')
            .augment_stochastic_oscillator(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                k_periods=14,
                d_periods=3
            )
        )
        stoch_df.groupby('symbol').tail(1)
        ```
        
        ``` {python}
        # Example 3 - Polars engine for single stock
        stoch_df = (
            df.query("symbol == 'AAPL'")
            .augment_stochastic_oscillator(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                k_periods=[14, 28],
                d_periods=3,
                engine='polars'
            )
        )
        stoch_df.head()
        ```
        
        ``` {python}
        # Example 4 - Polars engine with groupby
        stoch_df = (
            df.groupby('symbol')
            .augment_stochastic_oscillator(
                date_column='date',
                high_column='high',
                low_column='low',
                close_column='close',
                k_periods=14,
                d_periods=3,
                engine='polars'
            )
        )
        stoch_df.groupby('symbol').tail(1)
        """
        ...
    def augment_timeseries_signature(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The function `augment_timeseries_signature` takes a DataFrame and a date
        column as input and returns the original DataFrame with the **29 different
        date and time based features** added as new columns with the feature name
        based on the date_column.
        
        Parameters
        ----------
        data : pd.DataFrame
            The `data` parameter is a pandas DataFrame that contains the time series
            data.
        date_column : str
            The `date_column` parameter is a string that represents the name of the
            date column in the `data` DataFrame.
        reduce_memory : bool, optional
            The `reduce_memory` parameter is used to specify whether to reduce the memory usage of the DataFrame by converting int, float to smaller bytes and str to categorical data. This reduces memory for large data but may impact resolution of float and will change str to categorical. Default is False.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting datetime features. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for feature generation. This is generally faster than using "pandas"
              for large datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame with 29 datetime features added to it.
        
        - _index_num: An int64 feature that captures the entire datetime as a numeric value to the second
        - _year: The year of the datetime
        - _year_iso: The iso year of the datetime
        - _yearstart: Logical (0,1) indicating if first day of year (defined by frequency)
        - _yearend: Logical (0,1) indicating if last day of year (defined by frequency)
        - _leapyear: Logical (0,1) indicating if the date belongs to a leap year
        - _half: Half year of the date: Jan-Jun = 1, July-Dec = 2
        - _quarter: Quarter of the date: Jan-Mar = 1, Apr-Jun = 2, Jul-Sep = 3, Oct-Dec = 4
        - _quarteryear: Quarter of the date + relative year
        - _quarterstart: Logical (0,1) indicating if first day of quarter (defined by frequency)
        - _quarterend: Logical (0,1) indicating if last day of quarter (defined by frequency)
        - _month: The month of the datetime
        - _month_lbl: The month label of the datetime
        - _monthstart: Logical (0,1) indicating if first day of month (defined by frequency)
        - _monthend: Logical (0,1) indicating if last day of month (defined by frequency)
        - _yweek: The week ordinal of the year
        - _mweek: The week ordinal of the month
        - _wday: The number of the day of the week with Monday=1, Sunday=6
        - _wday_lbl: The day of the week label
        - _mday: The day of the datetime
        - _qday: The days of the relative quarter
        - _yday: The ordinal day of year
        - _weekend: Logical (0,1) indicating if the day is a weekend
        - _hour: The hour of the datetime
        - _minute: The minutes of the datetime
        - _second: The seconds of the datetime
        - _msecond: The microseconds of the datetime
        - _nsecond: The nanoseconds of the datetime
        - _am_pm: Half of the day, AM = ante meridiem, PM = post meridiem
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
        ```
        
        ```{python}
        # Adds 29 new time series features as columns to the original DataFrame (pandas engine)
        (
            df
                .augment_timeseries_signature(date_column='order_date', engine ='pandas')
                .glimpse()
        )
        ```
        
        ```{python}
        # Adds 29 new time series features as columns to the original DataFrame (polars engine)
        (
            df
                .augment_timeseries_signature(date_column='order_date', engine ='polars')
                .glimpse()
        )
        ```
        """
        ...
    def augment_wavelet(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
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
        ...
    def binarize(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `binarize` function prepares data for `correlate`, which is used for analyzing correlationfunnel plots.
        
        Binarization does the following:
        
        1. Takes in a pandas DataFrame or DataFrameGroupBy object, converts non-numeric
        columns to categorical,
        2. Replaces boolean columns with integers,
        3. Checks for data type and missing
        values,
        4. fixes low cardinality numeric data,
        5. fixes high skew numeric data, and
        6. finally applies a
        transformation to create a new DataFrame with binarized data.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that you want to binarize. It can be either a pandas
            DataFrame or a DataFrameGroupBy object.
        n_bins : int
            The `n_bins` parameter specifies the number of bins to use when binarizing numeric data. It is used
            in the `create_recipe` function to determine the number of bins for each numeric column.
            `pd.qcut()` is used to bin the numeric data.
        thresh_infreq : float
            The `thresh_infreq` parameter is a float that represents the threshold for infrequent categories.
            Categories that have a frequency below this threshold will be grouped together and labeled with the
            name specified in the `name_infreq` parameter. By default, the threshold is set to 0.01.
        name_infreq : str
            The `name_infreq` parameter is used to specify the name that will be assigned to the category
            representing infrequent values in a column. This is applicable when performing binarization on
            non-numeric columns. By default, the name assigned is "-OTHER".
        one_hot : bool
            The `one_hot` parameter is a boolean flag that determines whether or not to perform one-hot
            encoding on the categorical variables after binarization. If `one_hot` is set to `True`, the
            categorical variables will be one-hot encoded, creating binary columns for each unique category.
        
        Returns
        -------
            The function `binarize` returns the transformed data after applying various data preprocessing
            steps such as converting non-numeric columns to categorical, replacing boolean columns with
            integers, fixing low cardinality numeric data, fixing high skew numeric data, and creating a recipe
            for binarization.
        
        See Also
        --------
        - `correlate()` : Calculates the correlation between a target variable and all other variables in a pandas DataFrame.
        
        Examples
        --------
        
        ``` {python}
        # NON-TIMESERIES EXAMPLE ----
        
        import pandas as pd
        import numpy as np
        import pytimetk as tk
        
        # Set a random seed for reproducibility
        np.random.seed(0)
        
        # Define the number of rows for your DataFrame
        num_rows = 200
        
        # Create fake data for the columns
        data = {
            'Age': np.random.randint(18, 65, size=num_rows),
            'Gender': np.random.choice(['Male', 'Female'], size=num_rows),
            'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], size=num_rows),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], size=num_rows),
            'Years_Playing': np.random.randint(0, 30, size=num_rows),
            'Average_Income': np.random.randint(20000, 100000, size=num_rows),
            'Member_Status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_rows),
            'Number_Children': np.random.randint(0, 5, size=num_rows),
            'Own_House_Flag': np.random.choice([True, False], size=num_rows),
            'Own_Car_Count': np.random.randint(0, 3, size=num_rows),
            'PersonId': range(1, num_rows + 1),  # Add a PersonId column as a row count
            'Client': np.random.choice(['A', 'B'], size=num_rows)  # Add a Client column with random values 'A' or 'B'
        }
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Binarize the data
        df_binarized = df.binarize(n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)
        
        df_binarized.glimpse()
        ```
        
        ``` {python}
        df_correlated = df_binarized.correlate(target='Member_Status__Platinum')
        df_correlated.head(10)
        ```
        
        ``` {python}
        # Interactive
        df_correlated.plot_correlation_funnel(
            engine='plotly',
            height=600
        )
        ```
        
        ``` {python}
        # Static
        df_correlated.plot_correlation_funnel(
            engine ='plotnine',
            height = 900
        )
        ```
        """
        ...
    def correlate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The `correlate` function calculates the correlation between a target variable and all other
        variables in a pandas DataFrame, and returns the results sorted by absolute correlation in
        descending order.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter is the input data that you want to calculate correlations for. It can be
            either a pandas DataFrame or a grouped DataFrame obtained from a groupby operation.
        target : str
            The `target` parameter is a string that represents the column name in the DataFrame for which you
            want to calculate the correlation with other columns.
        method : str, default = 'pearson'
            The `method` parameter in the `correlate` function is used to specify the method for calculating
            the correlation coefficient. The available options for the `method` parameter are:
        
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
        
        
        Returns
        -------
            The function `correlate` returns a DataFrame with two columns: 'feature' and 'correlation'. The
            'feature' column contains the names of the features in the input data, and the 'correlation' column
            contains the correlation coefficients between each feature and the target variable. The DataFrame is
            sorted in descending order based on the absolute correlation values.
        
        See Also
        --------
        - `binarize()` : Prepares data for `correlate`, which is used for analyzing correlationfunnel plots.
        
        Examples
        --------
        
        ``` {python}
        # NON-TIMESERIES EXAMPLE ----
        
        import pandas as pd
        import numpy as np
        import pytimetk as tk
        
        # Set a random seed for reproducibility
        np.random.seed(0)
        
        # Define the number of rows for your DataFrame
        num_rows = 200
        
        # Create fake data for the columns
        data = {
            'Age': np.random.randint(18, 65, size=num_rows),
            'Gender': np.random.choice(['Male', 'Female'], size=num_rows),
            'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], size=num_rows),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], size=num_rows),
            'Years_Playing': np.random.randint(0, 30, size=num_rows),
            'Average_Income': np.random.randint(20000, 100000, size=num_rows),
            'Member_Status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_rows),
            'Number_Children': np.random.randint(0, 5, size=num_rows),
            'Own_House_Flag': np.random.choice([True, False], size=num_rows),
            'Own_Car_Count': np.random.randint(0, 3, size=num_rows),
            'PersonId': range(1, num_rows + 1),  # Add a PersonId column as a row count
            'Client': np.random.choice(['A', 'B'], size=num_rows)  # Add a Client column with random values 'A' or 'B'
        }
        
        # Create a DataFrame
        df = pd.DataFrame(data)
        
        # Binarize the data
        df_binarized = df.binarize(n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)
        
        df_binarized.glimpse()
        ```
        
        ``` {python}
        df_correlated = df_binarized.correlate(target='Member_Status__Platinum')
        df_correlated
        ```
        
        ``` {python}
        # Interactive
        df_correlated.plot_correlation_funnel(
            engine='plotly',
            height=400
        )
        ```
        
        ``` {python}
        # Static
        fig = df_correlated.plot_correlation_funnel(
            engine ='plotnine',
            height = 600
        )
        fig
        ```
        """
        ...
    def filter_by_time(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Filters a DataFrame or GroupBy object based on a specified date range.
        
        This function filters data in a pandas DataFrame or a pandas GroupBy object
        by a given date range. It supports various date formats and can handle both
        DataFrame and GroupBy objects.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The data to be filtered. It can be a pandas DataFrame or a pandas
            GroupBy object.
        date_column : str
            The name of the column in `data` that contains date information.
            This column is used for filtering the data based on the date range.
        start_date : str
            The start date of the filtering range. The format of the date can be
            YYYY, YYYY-MM, YYYY-MM-DD, YYYY-MM-DD HH, YYYY-MM-DD HH:SS, or YYYY-MM-DD HH:MM:SS.
            Default: 'start', which will filter from the earliest date in the data.
        end_date : str
            The end date of the filtering range. It supports the same formats as
            `start_date`.
            Default: 'end', which will filter until the latest date in the data.
        engine : str, default = 'pandas'
            The engine to be used for filtering the data. Currently, only 'pandas'.
        
        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the filtered data within the specified
            date range.
        
        Raises
        ------
        ValueError
            If the provided date strings do not match any of the supported formats.
        
        Notes
        -----
        - The function uses pd.to_datetime to convert the start date
          (e.g. start_date = "2014" becomes "2014-01-01").
        - The function internally uses the `parse_end_date` function to convert the
          end dates (e.g. end_date = "2014" becomes "2014-12-31").
        
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        import pandas as pd
        import datetime
        
        m4_daily_df = tk.datasets.load_dataset('m4_daily', parse_dates = ['date'])
        
        ```
        
        ```{python}
        # Example 1 - Filter by date
        
        df_filtered = tk.filter_by_time(
            data        = m4_daily_df,
            date_column = 'date',
            start_date  = '2014-07-03',
            end_date    = '2014-07-10'
        )
        
        df_filtered
        
        ```
        
        ```{python}
        # Example 2 - Filter by month.
        # Note: This will filter by the first day of the month.
        
        df_filtered = tk.filter_by_time(
            data        = m4_daily_df,
            date_column = 'date',
            start_date  = '2014-07',
            end_date    = '2014-09'
        )
        
        df_filtered
        
        ```
        
        ```{python}
        # Example 3 - Filter by year.
        # Note: This will filter by the first day of the year.
        
        df_filtered = tk.filter_by_time(
            data        = m4_daily_df,
            date_column = 'date',
            start_date  = '2014',
            end_date    = '2014'
        )
        
        df_filtered
        
        ```
        
        ```{python}
        # Example 4 - Filter by day/hour/minute/second
        # Here we'll use an hourly dataset, however this will also work for minute/second data
        
        # Load data and format date column appropriately
        m4_hourly_df = tk.datasets.load_dataset('m4_hourly', parse_dates = ['date'])
        
        df_filtered = tk.filter_by_time(
            data        = m4_hourly_df,
            date_column = "date",
            start_date  = '2015-07-01 12:00:00',
            end_date    = '2015-07-01 20:00:00'
        )
        
        df_filtered
        ```
        
        ```{python}
        # Example 5 - Combine year/month/day/hour/minute/second filters
        df_filtered = tk.filter_by_time(
            data        = m4_hourly_df,
            date_column = "date",
            start_date  = '2015-07-01',
            end_date    = '2015-07-29'
        )
        
        df_filtered
        
        ```
        
        ```{python}
        # Example 6 - Filter a GroupBy object
        
        df_filtered = (
            m4_hourly_df
                .groupby('id')
                .filter_by_time(
                    date_column = "date",
                    start_date  = '2015-07-01 12:00:00',
                    end_date    = '2015-07-01 20:00:00'
                )
        )
        
        df_filtered
        ```
        """
        ...
    def future_frame(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
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
        """
        ...
    def pad_by_time(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Make irregular time series regular by padding with missing dates.
        
        The `pad_by_time` function inserts missing dates into a Pandas DataFrame or
        DataFrameGroupBy object, through the process making an irregularly spaced
        time series regularly spaced.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter can be either a Pandas DataFrame or a Pandas
            DataFrameGroupBy object. It represents the data that you want to pad
            with missing dates.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to determine the minimum and maximum dates in theDataFrame, and to
            generate the regular date range for padding.
        freq : str, optional
            The `freq` parameter specifies the frequency at which the missing
            timestamps should be generated. It accepts a string representing a
            pandas frequency alias. Some common frequency aliases include:
        
            - S: secondly frequency
            - min: minute frequency
            - H: hourly frequency
            - B: business day frequency
            - D: daily frequency
            - W: weekly frequency
            - M: month end frequency
            - MS: month start frequency
            - BMS: Business month start
            - Q: quarter end frequency
            - QS: quarter start frequency
            - Y: year end frequency
            - YS: year start frequency
        start_date : str, optional
            Specifies the start of the padded series.  If NULL, it will use the
            lowest value of the input variable. In the case of groups, it will use
            the lowest value by group.
        
        end_date  : str, optional;
            Specifies the end of the padded series.  If NULL, it will use the highest
            value of the input variable.  In the case of groups, it will use the
            highest value by group.
        
        
        Returns
        -------
        pd.DataFrame
            The function `pad_by_time` returns a Pandas DataFrame that has been
            extended with future dates.
        
        Notes
        -----
        
        ## Performance
        
        This function uses a number of techniques to speed up computation for large
        datasets with many time series groups.
        
        - We use a vectorized approach to generate the Cartesian product of all
          unique group values and all dates in the date range.
        - We then merge this Cartesian product with the original data to introduce
          NaN values for missing rows. This approach is much faster than looping
          through each group and applying a function to each group.
        
        Note: There is no parallel processing since the vectorized approach is
              almost always faster.
        
        Examples
        --------
        ```{python}
        import pandas as pd
        import pytimetk as tk
        
        df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
        df
        ```
        
        ```{python}
        # Pad Single Time Series: Fill missing dates
        padded_df = (
            df
                .query('symbol == "AAPL"')
                .pad_by_time(
                    date_column = 'date',
                    freq        = 'D'
                )
        )
        padded_df
        ```
        
        ```{python}
        # Pad by Group: Pad each group with missing dates
        padded_df = (
            df
                .groupby('symbol')
                .pad_by_time(
                    date_column = 'date',
                    freq        = 'D'
                )
        )
        padded_df
        ```
        
        ```{python}
        # Pad with end dates specified
        padded_df = (
            df
                .groupby('symbol')
                .pad_by_time(
                    date_column = 'date',
                    freq        = 'D',
                    start_date  = '2013-01-01',
                    end_date    = '2023-09-22'
                )
        )
        padded_df.query('symbol == "AAPL"')
        ```
        """
        ...
    def parallel_apply(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
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
        """
        ...
    def plot_anomalies(self, *args: Any, **kwargs: Any) -> Any:
        """
        Creates plot of anomalies in time series data using Plotly, Matplotlib,
        or Plotnine. See the `anomalize()` function required to prepare the
        data for plotting.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data for the plot. It can be either a pandas DataFrame or a
            pandas DataFrameGroupBy object.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the dataframe that contains the dates for the plot.
        facet_ncol : int, optional
            The `facet_ncol` parameter determines the number of columns in the facet
            grid. It specifies how many subplots will be arranged horizontally in
            the plot.
        facet_nrow : int
            The `facet_nrow` parameter determines the number of rows in the facet
            grid. It specifies how many subplots will be arranged vertically in the
            grid.
        facet_scales : str, optional
            The `facet_scales` parameter determines the scaling of the y-axis in the
            facetted plots. It can take the following values:
            - "free_y": The y-axis scale will be free for each facet, but the x-axis
              scale will be fixed for all facets. This is the default value.
            - "free_x": The y-axis scale will be free for each facet, but the x-axis
              scale will be fixed for all facets.
            - "free": The y-axis scale will be free for each facet (subplot). This
              is the default value.
        
        facet_dir : str, optional
            The `facet_dir` parameter determines the direction in which the facets
            (subplots) are arranged. It can take two possible values:
            - "h": The facets will be arranged horizontally (in rows). This is the
              default value.
            - "v": The facets will be arranged vertically (in columns).
        line_color : str, optional
            The `line_color` parameter is used to specify the color of the lines in
            the time series plot. It accepts a string value representing a color
            code or name. The default value is "#2c3e50", which corresponds to a
            dark blue color.
        line_size : float
            The `line_size` parameter is used to specify the size of the lines in
            the time series plot. It determines the thickness of the lines.
        line_type : str, optional
            The `line_type` parameter is used to specify the type of line to be used
            in the time series plot.
        line_alpha : float
            The `line_alpha` parameter controls the transparency of the lines in the
            time series plot. It accepts a value between 0 and 1, where 0 means
            completely transparent (invisible) and 1 means completely opaque (solid).
        anom_color : str, optional
            The `anom_color` parameter is used to specify the color of the anomalies
            in the plot. It accepts a string value representing a color code or name.
            The default value is `#E31A1C`, which corresponds to a shade of red.
        anom_alpha : float
            The `anom_alpha` parameter controls the transparency (alpha) of the
            anomaly points in the plot. It accepts a float value between 0 and 1,
            where 0 means completely transparent and 1 means completely opaque.
        anom_size : Optional[float]
            The `anom_size` parameter is used to specify the size of the markers
            used to represent anomalies in the plot. It is an optional parameter,
            and if not provided, a default value will be used.
        ribbon_fill : str, optional
            The `ribbon_fill` parameter is used to specify the fill color of the
            ribbon that represents the range of anomalies in the plot. It accepts a
            string value representing a color code or name.
        ribbon_alpha : float
            The parameter `ribbon_alpha` controls the transparency of the ribbon
            fill in the plot. It accepts a float value between 0 and 1, where 0
            means completely transparent and 1 means completely opaque. A higher
            value will make the ribbon fill more visible, while a lower value will
            make it
        y_intercept : float
            The `y_intercept` parameter is used to add a horizontal line to the plot
            at a specific y-value. It can be set to a numeric value to specify the
            y-value of the intercept. If set to `None` (default), no y-intercept
            line will be added to the plot
        y_intercept_color : str, optional
            The `y_intercept_color` parameter is used to specify the color of the
            y-intercept line in the plot. It accepts a string value representing a
            color code or name. The default value is "#2c3e50", which corresponds to
            a dark blue color. You can change this value.
        x_intercept : str
            The `x_intercept` parameter is used to add a vertical line at a specific
            x-axis value on the plot. It is used to highlight a specific point or
            event in the time series data.
            - By default, it is set to `None`, which means no vertical line will be
              added.
            - You can use a date string to specify the x-axis value of the intercept.
              For example, "2020-01-01" would add a vertical line at the beginning
              of the year 2020.
        x_intercept_color : str, optional
            The `x_intercept_color` parameter is used to specify the color of the
            vertical line that represents the x-intercept in the plot. By default,
            it is set to "#2c3e50", which is a dark blue color. You can change this
            value to any valid color code.
        legend_show : bool, optional
            The `legend_show` parameter is a boolean indicating whether or not to
            show the legend in the plot. If set to True, the legend will be
            displayed. The default value is True.
        title : str, optional
            The title of the plot.
        x_lab : str
            The `x_lab` parameter is used to specify the label for the x-axis in the
            plot. It is a string that represents the label text.
        y_lab : str
            The `y_lab` parameter is used to specify the label for the y-axis in the
            plot. It is a string that represents the label for the y-axis.
        color_lab : str, optional
            The `color_lab` parameter is used to specify the label for the legend or
            color scale in the plot. It is used to provide a description of the
            colors used in the plot, typically when a color column is specified.
        x_axis_date_labels : str, optional
            The `x_axis_date_labels` parameter is used to specify the format of the
            date labels on the x-axis of the plot. It accepts a string representing
            the format of the date labels. For  example, "%b %Y" would display the
            month abbreviation and year (e.g., Jan 2020).
        base_size : float, optional
            The `base_size` parameter is used to set the base font size for the plot.
            It determines the size of the text elements such as axis labels, titles,
            and legends.
        width : int
            The `width` parameter is used to specify the width of the plot. It
            determines the horizontal size of the plot in pixels.
        height : int
            The `height` parameter is used to specify the height of the plot in
            pixels. It determines the vertical size of the plot when it is rendered.
        engine : str, optional
            The `engine` parameter specifies the plotting library to use for
            creating the time series plot. It can take one of the following values:
        
            - "plotly" (interactive): Use the plotly library to create the plot.
               This is the default value.
            - "plotnine" (static): Use the plotnine library to create the plot.
              This is the default value.
            - "matplotlib" (static): Use the matplotlib library to create the plot.
        plotly_dropdown : bool
            For analyzing many plots. When set to True and groups are provided, the function switches from
            faceting to create a dropdown menu to switch between different groups. Default: `False`.
        plotly_dropdown_x : float
            The x-axis location of the dropdown. Default: 0.
        plotly_dropdown_y : float
            The y-axis location of the dropdown. Default: 1.
        
        Returns
        -------
            A plot object, depending on the specified `engine` parameter:
            - If `engine` is set to 'plotnine' or 'matplotlib', the function returns
              a plot object that can be further customized or displayed.
            - If `engine` is set to 'plotly', the function returns a plotly figure
              object.
        
        See Also
        --------
        `anomalize()`: The `anomalize()` function is used to prepare the data for
                       plotting anomalies in a time series data.
        
        Examples
        --------
        ```{python}
        # EXAMPLE 1: SINGLE TIME SERIES
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
        
        # Anomalize the data
        anomalize_df = tk.anomalize(
            df, "date", "value",
            method = "twitter",
            iqr_alpha = 0.10,
            clean_alpha = 0.75,
            clean = "min_max",
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands, plotly engine
        (
             anomalize_df
                .plot_anomalies(
                    date_column = "date",
                    engine = "plotly",
                )
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands, plotly engine
        (
             anomalize_df
                .plot_anomalies(
                    date_column = "date",
                    engine = "plotnine",
                )
        )
        ```
        
        ``` {python}
        # EXAMPLE 2: MULTIPLE TIME SERIES
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[["id", "Date", "Weekly_Sales"]]
        
        anomalize_df = (
            df
                .groupby('id')
                .anomalize(
                    "Date", "Weekly_Sales",
                )
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands, plotly engine
        (
            anomalize_df
                .groupby(["id"])
                .plot_anomalies(
                    date_column = "Date",
                    facet_ncol = 2,
                    width = 800,
                    height = 800,
                    engine = "plotly",
                )
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands, plotly engine, plotly dropdown
        (
            anomalize_df
                .groupby(["id"])
                .plot_anomalies(
                    date_column = "Date",
                    engine = "plotly",
                    plotly_dropdown=True,
                    plotly_dropdown_x=1.05,
                    plotly_dropdown_y=1.15
                )
        )
        ```
        
        ``` {python}
        # Visualize the anomaly bands, matplotlib engine
        (
            anomalize_df
                .groupby(["id"])
                .plot_anomalies(
                    date_column = "Date",
                    facet_ncol = 2,
                    width = 800,
                    height = 800,
                    engine = "matplotlib",
                )
        )
        ```
        """
        ...
    def plot_anomalies_cleaned(self, *args: Any, **kwargs: Any) -> Any:
        """
        The `plot_anomalies_cleaned` function takes in data from the `anomalize()`
        function, and returns a plot of the anomalies cleaned.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data for the plot from `anomalize`. It can be either a pandas
            DataFrame or a pandas DataFrameGroupBy object.
        date_column : str
            The name of the column in the data that contains the dates.
        facet_ncol : int, optional
            The number of columns in the facet grid. It is an optional parameter,
        line_color : str, optional
            The color of the line in the plot. It is specified as a hexadecimal
            color code. The default value is "#2c3e50".
        line_color_cleaned : str, optional
            The color of the line in the plot. It is specified as a hexadecimal or a matplotlib color name. The default value is "#e31a1c".
        line_size : Optional[float]
            The line_size parameter determines the thickness of the lines in the
            plot. It is an optional parameter, so if you don't specify a value, the
            default line size will be used.
        line_type : str, optional
            The `line_type` parameter specifies the type of line to be used in the
            plot. It can take the following values:
            - "solid" (default): a solid line
            - "dashed": a dashed line
        line_alpha : float
            The `line_alpha` parameter controls the transparency of the lines in the
            plot. It accepts a float value between 0 and 1, where 0 means completely
            transparent and 1 means completely opaque.
        y_intercept : Optional[float]
            The `y_intercept` parameter is an optional float value that specifies
            the y-coordinate of a horizontal line to be plotted on the graph. This
            line can be used to indicate a specific threshold or reference value. If
            not specified, no horizontal line will be plotted.
        y_intercept_color : str, optional
            The `y_intercept_color` parameter is used to specify the color of the
            y-intercept line on the plot. By default, it is set to `"#2c3e50"`,
            which is a dark blue color. You can change this parameter to any valid
            color code or name to change the color of the line.
        x_intercept : Optional[str]
            The `x_intercept` parameter is used to specify the value on the x-axis
            where you want to draw a vertical line. This can be useful for
            highlighting a specific point or event in the data.
        x_intercept_color : str, optional
            The `x_intercept_color` parameter is used to specify the color of the
            vertical line representing the x-intercept on the plot. By default, it
            is set to "#2c3e50", which is a dark blue color. You can change this
            parameter to any valid color code or name to change the color of the line.
        title : str, optional
            The title of the plot. It is set to "Anomalies Cleaned Plot" by default.
        x_lab : str
            The x_lab parameter is used to specify the label for the x-axis of the
            plot. It is a string that represents the label text.
        y_lab : str
            The `y_lab` parameter is used to specify the label for the y-axis of the
            plot. It is a string that represents the label text.
        x_axis_date_labels : str, optional
            The `x_axis_date_labels` parameter is used to specify the format of the
            date labels on the x-axis of the plot. It accepts a string representing
            the format of the date labels. For example, "%b %Y" would display the
            month abbreviation and year (e.g., Jan 2019).
        base_size : float, optional
            The `base_size` parameter determines the base font size for the plot. It
            is used to control the size of the text elements in the plot, such as
            axis labels, titles, and tick labels. The default value is 11, but you
            can adjust it to make the text larger or smaller
        width : Optional[int]
            The width parameter determines the width of the plot in pixels. It is an
            optional parameter, so if you don't specify a value, the plot will be
            displayed with the default width.
        height : Optional[int]
            The height parameter determines the height of the plot in pixels. It is
            an optional parameter, so if you don't specify a value, the plot will be
            displayed with a default height.
        engine : str, optional
            The `engine` parameter specifies the plotting engine to use. It can be
            set to either "plotly", "plotnine", or "matplotlib".
        
        Returns
        -------
            A plotly, plotnine, or matplotlib plot.
        
        See Also
        --------
        1. anomalize : Function that calculates the anomalies and formats the data
        for visualization.
        2. plot_anomalies : Function that plots the anomalies.
        
        Examples
        --------
        
        ``` {python}
        # EXAMPLE 1: SINGLE TIME SERIES
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
        
        # Anomalize the data
        anomalize_df = tk.anomalize(
            df, "date", "value",
            method = "twitter",
            iqr_alpha = 0.10,
            clean_alpha = 0.75,
            clean = "min_max",
            verbose = True,
        )
        
        # Visualize the results
        anomalize_df.plot_anomalies_cleaned("date")
        ```
        
        ``` {python}
        # EXAMPLE 2: MULTIPLE TIME SERIES
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[["id", "Date", "Weekly_Sales"]]
        
        anomalize_df = (
            df
                .groupby('id')
                .anomalize(
                    "Date", "Weekly_Sales",
                    period = 52,
                    trend = 52,
                    threads = 1
                )
        )
        
        # Visualize the decomposition results
        
        (
            anomalize_df
                .groupby("id")
                .plot_anomalies_cleaned(
                    date_column = "Date",
                    line_color = "steelblue",
                    width = 600,
                    height = 1000,
                    x_axis_date_labels = "%y",
                    engine = 'plotly',
                )
        )
        ```
        """
        ...
    def plot_anomalies_decomp(self, *args: Any, **kwargs: Any) -> Any:
        """
        The `plot_anomalies_decomp` function takes in data from the `anomalize()`
        function, and returns a plot of the anomaly decomposition.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The input data for the plot from `anomalize`. It can be either a pandas
            DataFrame or a pandas DataFrameGroupBy object.
        date_column : str
            The name of the column in the data that contains the dates.
        line_color : str, optional
            The color of the line in the plot. It is specified as a hexadecimal
            color code. The default value is "#2c3e50".
        line_size : Optional[float]
            The line_size parameter determines the thickness of the lines in the
            plot. It is an optional parameter, so if you don't specify a value, the
            default line size will be used.
        line_type : str, optional
            The `line_type` parameter specifies the type of line to be used in the
            plot. It can take the following values:
            - "solid" (default): a solid line
            - "dashed": a dashed line
        line_alpha : float
            The `line_alpha` parameter controls the transparency of the lines in the
            plot. It accepts a float value between 0 and 1, where 0 means completely
            transparent and 1 means completely opaque.
        y_intercept : Optional[float]
            The `y_intercept` parameter is an optional float value that specifies
            the y-coordinate of a horizontal line to be plotted on the graph. This
            line can be used to indicate a specific threshold or reference value. If
            not specified, no horizontal line will be plotted.
        y_intercept_color : str, optional
            The `y_intercept_color` parameter is used to specify the color of the
            y-intercept line on the plot. By default, it is set to `"#2c3e50"`,
            which is a dark blue color. You can change this parameter to any valid
            color code or name to change the color of the line.
        x_intercept : Optional[str]
            The `x_intercept` parameter is used to specify the value on the x-axis
            where you want to draw a vertical line. This can be useful for
            highlighting a specific point or event in the data.
        x_intercept_color : str, optional
            The `x_intercept_color` parameter is used to specify the color of the
            vertical line representing the x-intercept on the plot. By default, it
            is set to "#2c3e50", which is a dark blue color. You can change this
            parameter to any valid color code or name to change the color of the line.
        title : str, optional
            The title of the plot. It is set to "Anomaly Decomposition Plot" by default.
        x_lab : str
            The x_lab parameter is used to specify the label for the x-axis of the
            plot. It is a string that represents the label text.
        y_lab : str
            The `y_lab` parameter is used to specify the label for the y-axis of the
            plot. It is a string that represents the label text.
        x_axis_date_labels : str, optional
            The `x_axis_date_labels` parameter is used to specify the format of the
            date labels on the x-axis of the plot. It accepts a string representing
            the format of the date labels. For example, "%b %Y" would display the
            month abbreviation and year (e.g., Jan 2019).
        base_size : float, optional
            The `base_size` parameter determines the base font size for the plot. It
            is used to control the size of the text elements in the plot, such as
            axis labels, titles, and tick labels. The default value is 11, but you
            can adjust it to make the text larger or smaller
        width : Optional[int]
            The width parameter determines the width of the plot in pixels. It is an
            optional parameter, so if you don't specify a value, the plot will be
            displayed with the default width.
        height : Optional[int]
            The height parameter determines the height of the plot in pixels. It is
            an optional parameter, so if you don't specify a value, the plot will be
            displayed with a default height.
        engine : str, optional
            The `engine` parameter specifies the plotting engine to use. It can be
            set to either "plotly", "plotnine", or "matplotlib".
        
        Returns
        -------
            A plotly, plotnine, or matplotlib plot.
        
        See Also
        --------
        1. anomalize : Function that calculates the anomalies and formats the data
        for visualization.
        2. plot_anomalies : Function that plots the anomalies.
        
        Examples
        --------
        
        ``` {python}
        # EXAMPLE 1: SINGLE TIME SERIES
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
        
        # Anomalize the data
        anomalize_df = tk.anomalize(
            df, "date", "value",
            method = "twitter",
            iqr_alpha = 0.10,
            clean_alpha = 0.75,
            clean = "min_max",
            verbose = True,
        )
        
        # Visualize the results, plotly
        anomalize_df.plot_anomalies_decomp("date", engine = 'plotly')
        ```
        
        ```{python}
        # Visualize the results, plotnine
        anomalize_df.plot_anomalies_decomp("date", engine = "plotnine")
        ```
        
        ``` {python}
        # EXAMPLE 2: MULTIPLE TIME SERIES
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[["id", "Date", "Weekly_Sales"]]
        
        anomalize_df = (
            df
                .groupby('id')
                .anomalize(
                    "Date", "Weekly_Sales",
                    period = 52,
                    trend = 52,
                    threads = 1
                )
        )
        
        # Visualize the decomposition results, plotly
        (
            anomalize_df
                .groupby("id")
                .plot_anomalies_decomp(
                    date_column = "Date",
                    line_color = "steelblue",
                    width = 1200,
                    height = 800,
                    x_axis_date_labels = "%y",
                    engine = 'plotly',
                )
        )
        ```
        
        ```{python}
        # Visualize the decomposition results, plotnine
        
        (
            anomalize_df
                .groupby("id")
                .plot_anomalies_decomp(
                    date_column = "Date",
                    line_color = "steelblue",
                    width = 1200,
                    height = 800,
                    x_axis_date_labels = "%y",
                    engine = 'plotnine',
                )
        )
        ```
        """
        ...
    def plot_timeseries(self, *args: Any, **kwargs: Any) -> Any:
        """
        Creates time series plots using different plotting engines such as Plotnine,
        Matplotlib, and Plotly.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The input data for the plot. It can be either a Pandas DataFrame or a
            Pandas DataFrameGroupBy object.
        date_column : str
            The name of the column in the DataFrame that contains the dates for the
            time series data.
        value_column : str or list
            The `value_column` parameter is used to specify the name of the column
            in the DataFrame that contains the values for the time series data. This
            column will be plotted on the y-axis of the time series plot.
        
            LONG-FORMAT PLOTTING:
            If the `value_column` parameter is a string, it will be treated as a
            single column name. To plot multiple time series,
            group the DataFrame first using pd.DataFrame.groupby().
        
            WIDE-FORMAT PLOTTING:
            If the `value_column` parameter is a list, it will plotted
            as multiple time series (wide-format).
        color_column : str
            The `color_column` parameter is an optional parameter that specifies the
            column in the DataFrame that will be used to assign colors to the
            different time series. If this parameter is not provided, all time
            series will have the same color.
        
            LONG-FORMAT PLOTTING:
            The `color_column` parameter is a single column name.
        
            WIDE-FORMAT PLOTTING:
            The `color_column` parameter must be the same list
            as the `value_column` parameter to color the different time series when performing wide-format plotting.
        color_palette : list, optional
            The `color_palette` parameter is used to specify the colors to be used
            for the different time series. It accepts a list of color codes or names.
            If the `color_column` parameter is not provided, the `tk.palette_timetk()`
            color palette will be used.
        facet_ncol : int, optional
            The `facet_ncol` parameter determines the number of columns in the facet
            grid. It specifies how many subplots will be arranged horizontally in
            the plot.
        facet_nrow : int
            The `facet_nrow` parameter determines the number of rows in the facet
            grid. It specifies how many subplots will be arranged vertically in the
            grid.
        facet_scales : str, optional
            The `facet_scales` parameter determines the scaling of the y-axis in the
            facetted plots. It can take the following values:
            - "free_y": The y-axis scale will be free for each facet, but the x-axis
            scale will be fixed for all facets. This is the default value.
            - "free_x": The y-axis scale will be free for each facet, but the x-axis
            scale will be fixed for all facets.
            - "free": The y-axis scale will be free for each facet (subplot). This
            is the default value.
        
        facet_dir : str, optional
            The `facet_dir` parameter determines the direction in which the facets
            (subplots) are arranged. It can take two possible values:
            - "h": The facets will be arranged horizontally (in rows). This is the
              default value.
            - "v": The facets will be arranged vertically (in columns).
        line_color : str, optional
            The `line_color` parameter is used to specify the color of the lines in
            the time series plot. It accepts a string value representing a color
            code or name. The default value is "#2c3e50", which corresponds to a
            dark blue color.
        line_size : float
            The `line_size` parameter is used to specify the size of the lines in
            the time series plot. It determines the thickness of the lines.
        line_type : str, optional
            The `line_type` parameter is used to specify the type of line to be used
            in the time series plot.
        line_alpha : float
            The `line_alpha` parameter controls the transparency of the lines in the
            time series plot. It accepts a value between 0 and 1, where 0 means
            completely transparent (invisible) and 1 means completely opaque (solid).
        y_intercept : float
            The `y_intercept` parameter is used to add a horizontal line to the plot
            at a specific y-value. It can be set to a numeric value to specify the
            y-value of the intercept. If set to `None` (default), no y-intercept
            line will be added to the plot
        y_intercept_color : str, optional
            The `y_intercept_color` parameter is used to specify the color of the
            y-intercept line in the plot. It accepts a string value representing a
            color code or name. The default value is "#2c3e50", which corresponds to
            a dark blue color. You can change this value.
        x_intercept : str
            The `x_intercept` parameter is used to add a vertical line at a specific
            x-axis value on the plot. It is used to highlight a specific point or
            event in the time series data.
            - By default, it is set to `None`, which means no vertical line will be
              added.
            - You can use a date string to specify the x-axis value of the intercept.
              For example, "2020-01-01" would add a vertical line at the beginning
              of the year 2020.
        x_intercept_color : str, optional
            The `x_intercept_color` parameter is used to specify the color of the
            vertical line that represents the x-intercept in the plot. By default,
            it is set to "#2c3e50", which is a dark blue color. You can change this
            value to any valid color code.
        smooth : bool, optional
            The `smooth` parameter is a boolean indicating whether or not to apply
            smoothing to the time eries data. If set to True, the time series will
            be smoothed using the lowess algorithm. The default value is True.
        smooth_color : str, optional
            The `smooth_color` parameter is used to specify the color of the
            smoothed line in the time series plot. It accepts a string value
            representing a color code or name. The default value is `#3366FF`,
            which corresponds to a shade of blue. You can change this value to any
            valid color code.
        smooth_frac : float
            The `smooth_frac` parameter is used to control the fraction of data
            points used for smoothing the time series. It determines the degree of
            smoothing applied to the data. A smaller value of `smooth_frac` will
            result in more smoothing, while a larger value will result in less
            smoothing. The default value is 0.2.
        smooth_size : float
            The `smooth_size` parameter is used to specify the size of the line used
            to plot the smoothed values in the time series plot. It is a numeric
            value that controls the thickness of the line. A larger value will result
            in a thicker line, while a smaller value will result in a thinner line
        smooth_alpha : float
            The `smooth_alpha` parameter controls the transparency of the smoothed
            line in the plot. It accepts a value between 0 and 1, where 0 means
            completely transparent and 1 means completely opaque.
        legend_show : bool, optional
            The `legend_show` parameter is a boolean indicating whether or not to
            show the legend in the plot. If set to True, the legend will be
            displayed. The default value is True.
        title : str, optional
            The title of the plot.
        x_lab : str
            The `x_lab` parameter is used to specify the label for the x-axis in the
            plot. It is a string that represents the label text.
        y_lab : str
            The `y_lab` parameter is used to specify the label for the y-axis in the
            plot. It is a string that represents the label for the y-axis.
        color_lab : str, optional
            The `color_lab` parameter is used to specify the label for the legend or
            color scale in the plot. It is used to provide a description of the colors
            used in the plot, typically when a color column is specified.
        x_axis_date_labels : str, optional
            The `x_axis_date_labels` parameter is used to specify the format of the
            date labels on the x-axis of the plot. It accepts a string representing
            the format of the date labels. For  example, "%b %Y" would display the
            month abbreviation and year (e.g., Jan 2020).
        base_size : float, optional
            The `base_size` parameter is used to set the base font size for the plot.
            It determines the size of the text elements such as axis labels, titles,
            and legends.
        width : int
            The `width` parameter is used to specify the width of the plot. It
            determines the horizontal size of the plot in pixels.
        height : int
            The `height` parameter is used to specify the height of the plot in
            pixels. It determines the vertical size of the plot when it is rendered.
        engine : str, optional
            The `engine` parameter specifies the plotting library to use for
            creating the time series plot. It can take one of the following values:
        
            - "plotly" (interactive): Use the plotly library to create the plot.
              This is the default value.
            - "plotnine" (static): Use the plotnine library to create the plot.
              This is the default value.
            - "matplotlib" (static): Use the matplotlib library to create the plot.
        plotly_dropdown : bool
            For analyzing many plots. When set to True and groups are provided, the function switches from
            faceting to create a dropdown menu to switch between different groups. Default: `False`.
        plotly_dropdown_x : float
            The x-axis location of the dropdown. Default: 0.
        plotly_dropdown_y : float
            The y-axis location of the dropdown. Default: 1.
        
        
        
        Returns
        -------
            The function `plot_timeseries` returns a plot object, depending on the
            specified `engine` parameter.
            - If `engine` is set to 'plotnine' or 'matplotlib', the function returns
              a plot object that can be further customized or displayed.
            - If `engine` is set to 'plotly', the function returns a plotly figure
              object.
        
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        
        df = tk.load_dataset('m4_monthly', parse_dates = ['date'])
        
        # Plotly Object: Single Time Series
        fig = (
            df
                .query('id == "M750"')
                .plot_timeseries(
                    'date', 'value',
                    facet_ncol = 1,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly',
                )
        )
        fig
        ```
        
        ```{python}
        # Plotly Object: Grouped Time Series (Facets)
        fig = (
            df
                .groupby('id')
                .plot_timeseries(
                    'date', 'value',
                    facet_ncol = 2,
                    facet_scales = "free_y",
                    smooth_frac = 0.2,
                    smooth_size = 2.0,
                    y_intercept = None,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly',
                    width = 600,
                    height = 500,
                )
        )
        fig
        ```
        
        ```{python}
        # Plotly Object: Grouped Time Series (Plotly Dropdown)
        fig = (
            df
                .groupby('id')
                .plot_timeseries(
                    'date', 'value',
                    facet_scales = "free_y",
                    smooth_frac = 0.2,
                    smooth_size = 2.0,
                    y_intercept = None,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly',
                    width = 600,
                    height = 500,
                    plotly_dropdown = True, # Plotly Dropdown
                )
        )
        fig
        ```
        
        ```{python}
        # Plotly Object: Color Column
        fig = (
            df
                .plot_timeseries(
                    'date', 'value',
                    color_column = 'id',
                    smooth = False,
                    y_intercept = 0,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly',
                )
        )
        fig
        ```
        
        ```{python}
        # Plotnine Object: Single Time Series
        fig = (
            df
                .query('id == "M1"')
                .plot_timeseries(
                    'date', 'value',
                    x_axis_date_labels = "%Y",
                    engine = 'plotnine'
                )
        )
        fig
        ```
        
        ```{python}
        # Plotnine Object: Grouped Time Series
        fig = (
            df
                .groupby('id')
                .plot_timeseries(
                    'date', 'value',
                    facet_ncol = 2,
                    facet_scales = "free",
                    line_size = 0.35,
                    x_axis_date_labels = "%Y",
                    engine = 'plotnine'
                )
        )
        fig
        ```
        
        ```{python}
        # Plotnine Object: Color Column
        fig = (
            df
                .plot_timeseries(
                    'date', 'value',
                    color_column = 'id',
                    smooth = False,
                    y_intercept = 0,
                    x_axis_date_labels = "%Y",
                    engine = 'plotnine',
                )
        )
        fig
        ```
        
        ```{python}
        # Matplotlib object (same as plotnine, but converted to matplotlib object)
        fig = (
            df
                .groupby('id')
                .plot_timeseries(
                    'date', 'value',
                    color_column = 'id',
                    facet_ncol = 2,
                    x_axis_date_labels = "%Y",
                    engine = 'matplotlib',
                )
        )
        fig
        ```
        
        ``` {python}
        # Wide-Format Plotting
        
        # Imports
        import pandas as pd
        import numpy as np
        import pytimetk as tk
        
        # Set a random seed for reproducibility
        np.random.seed(42)
        
        # Create a date range
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        
        # Generate random sales data and compute expenses and profit
        sales = np.random.uniform(1000, 5000, len(dates))
        expenses = sales * np.random.uniform(0.5, 0.8, len(dates))
        profit = sales - expenses
        
        # Create the DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sales': sales,
            'expenses': expenses,
            'profit': profit
        })
        
        (
            df
                .plot_timeseries(
                    date_column = 'date',
                    value_column = ['sales', 'expenses', 'profit'],
                    color_column = ['sales', 'expenses', 'profit'],
                    smooth = True,
                    x_axis_date_labels = "%Y",
                    engine = 'plotly',
                    plotly_dropdown = True, # Plotly Dropdown
                )
        )
        ```
        """
        ...
    def progress_apply(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
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
        """
        ...
    def reduce_memory_usage(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Iterate through all columns of a Pandas DataFrame and modify the dtypes to reduce memory usage.
        
        Parameters:
        -----------
        data: pd.DataFrame
            Input dataframe to reduce memory usage.
        
        Returns:
        --------
        pd.DataFrame
          Dataframe with reduced memory usage.
        """
        ...
    def sort_dataframe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        The function `sort_dataframe` sorts a DataFrame by a specified date column, handling both regular
        DataFrames and grouped DataFrames.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
            The `data` parameter in the `sort_dataframe` function can accept either a pandas DataFrame or a
            grouped DataFrame (DataFrameGroupBy object).
        date_column
            The `date_column` parameter in the `sort_dataframe` method is used to specify the column in the
            DataFrame by which the sorting will be performed. This column contains dates that will be used as
            the basis for sorting the DataFrame or DataFrameGroupBy object.
        keep_grouped_df
            If `True` and `data` is a grouped data frame, a grouped data frame will be returned. If `False`, an ungrouped data frame is returned.
        
        Returns
        -------
            The `sort_dataframe` function returns a sorted DataFrame based on the specified date column. If the
            input data is a regular DataFrame, it sorts the DataFrame by the specified date column. If the input
            data is a grouped DataFrame (DataFrameGroupBy object), it sorts the DataFrame by the group names and
            the specified date column. The function returns the sorted DataFrame.
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('walmart_sales_weekly', parse_dates=['Date'])
        
        df.sort_dataframe('Date')
        
        df.groupby('id').sort_dataframe('Date').obj
        
        df.groupby(['id', 'Store', 'Dept']).sort_dataframe('Date').obj
        ```
        """
        ...
    def summarize_by_time(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Summarize a DataFrame or GroupBy object by time.
        
        The `summarize_by_time` function aggregates data by a specified time period
        and one or more numeric columns, allowing for grouping and customization of
        the time-based aggregation.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            A pandas DataFrame or a pandas GroupBy object. This is the data that you
            want to summarize by time.
        date_column : str
            The name of the column in the data frame that contains the dates or
            timestamps to be aggregated by. This column must be of type datetime64.
        value_column : str or list
            The `value_column` parameter is the name of one or more columns in the
            DataFrame that you want to aggregate by. It can be either a string
            representing a single column name, or a list of strings representing
            multiple column names.
        freq : str, optional
            The `freq` parameter specifies the frequency at which the data should be
            aggregated. It accepts a string representing a pandas frequency offset,
            such as "D" for daily or "MS" for month start. The default value is "D",
            which means the data will be aggregated on a daily basis. Some common
            frequency aliases include:
        
            - S: secondly frequency
            - min: minute frequency
            - H: hourly frequency
            - D: daily frequency
            - W: weekly frequency
            - M: month end frequency
            - MS: month start frequency
            - Q: quarter end frequency
            - QS: quarter start frequency
            - Y: year end frequency
            - YS: year start frequency
        
        agg_func : list, optional
            The `agg_func` parameter is used to specify one or more aggregating
            functions to apply to the value column(s) during the summarization
            process. It can be a single function or a list of functions. The default
            value is `"sum"`, which represents the sum function. Some common
            aggregating functions include:
        
            - "sum": Sum of values
            - "mean": Mean of values
            - "median": Median of values
            - "min": Minimum of values
            - "max": Maximum of values
            - "std": Standard deviation of values
            - "var": Variance of values
            - "first": First value in group
            - "last": Last value in group
            - "count": Count of values
            - "nunique": Number of unique values
            - "corr": Correlation between values
        
            Pandas Engine Only:
            Custom `lambda` aggregating functions can be used too. Here are several
            common examples:
        
            - ("q25", lambda x: x.quantile(0.25)): 25th percentile of values
            - ("q75", lambda x: x.quantile(0.75)): 75th percentile of values
            - ("iqr", lambda x: x.quantile(0.75) - x.quantile(0.25)): Interquartile range of values
            - ("range", lambda x: x.max() - x.min()): Range of values
        
        wide_format : bool, optional
            A boolean parameter that determines whether the output should be in
            "wide" or "long" format. If set to `True`, the output will be in wide
            format, where each group is represented by a separate column. If set to
            False, the output will be in long format, where each group is represented
            by a separate row. The default value is `False`.
        fillna : int, optional
            The `fillna` parameter is used to specify the value to fill missing data
            with. By default, it is set to 0. If you want to keep missing values as
            NaN, you can use `np.nan` as the value for `fillna`.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            summarizing the data. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library
              for summarizing the data. This can be faster than using "pandas" for
              large datasets.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame that is summarized by time.
        
        Examples
        --------
        ```{python}
        import pytimetk as tk
        import pandas as pd
        
        df = tk.load_dataset('bike_sales_sample', parse_dates = ['order_date'])
        
        df
        ```
        
        ```{python}
        # Example 1 - Summarize by time with a DataFrame object, pandas engine
        (
            df
                .summarize_by_time(
                    date_column  = 'order_date',
                    value_column = 'total_price',
                    freq         = "MS",
                    agg_func     = ['mean', 'sum'],
                    engine       = 'pandas'
                )
        )
        ```
        
        ```{python}
        # Example 2 - Summarize by time with a GroupBy object (Wide Format), polars engine
        (
            df
                .groupby(['category_1', 'frame_material'])
                .summarize_by_time(
                    date_column  = 'order_date',
                    value_column = ['total_price', 'quantity'],
                    freq         = 'MS',
                    agg_func     = 'sum',
                    wide_format  = True,
                    engine       = 'polars'
                )
        )
        ```
        
        ```{python}
        # Example 3 - Summarize by time with a GroupBy object (Wide Format)
        (
            df
                .groupby('category_1')
                .summarize_by_time(
                    date_column  = 'order_date',
                    value_column = 'total_price',
                    freq         = 'MS',
                    agg_func     = 'sum',
                    wide_format  = True,
                    engine       = 'pandas'
                )
        )
        ```
        
        ```{python}
        # Example 4 - Summarize by time with a GroupBy object and multiple value columns and summaries (Wide Format)
        # Note - This example only works with the pandas engine
        (
            df
                .groupby('category_1')
                .summarize_by_time(
                    date_column  = 'order_date',
                    value_column = ['total_price', 'quantity'],
                    freq         = 'MS',
                    agg_func     = [
                        'sum',
                        'mean',
                        ('q25', lambda x: x.quantile(0.25)),
                        ('q75', lambda x: x.quantile(0.75))
                    ],
                    wide_format  = False,
                    engine       = 'pandas'
                )
        )
        ```
        """
        ...
    def ts_features(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
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
        """
        ...
    def ts_summary(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Computes summary statistics for a time series data, either for the entire
        dataset or grouped by a specific column.
        
        Parameters
        ----------
        data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
            The `data` parameter can be either a Pandas DataFrame or a Pandas
            DataFrameGroupBy object. It represents the data that you want to
            summarize.
        date_column : str
            The `date_column` parameter is a string that specifies the name of the
            column in the DataFrame that contains the dates. This column will be
            used to compute summary statistics for the time series data.
        engine : str, optional
            The `engine` parameter is used to specify the engine to use for
            augmenting lags. It can be either "pandas" or "polars".
        
            - The default value is "pandas".
        
            - When "polars", the function will internally use the `polars` library.
            This can be faster than using "pandas" for large datasets.
        
        Returns
        -------
        pd.DataFrame
            The `ts_summary` function returns a summary of time series data. The
            summary includes the following statistics:
            - If grouped data is provided, the returned data will contain the
              grouping columns first.
            - `date_n`: The number of observations in the time series.
            - `date_tz`: The time zone of the time series.
            - `date_start`: The first date in the time series.
            - `date_end`: The last date in the time series.
            - `freq_inferred_unit`: The inferred frequency of the time series from
                                   `pandas`.
            - `freq_median_timedelta`: The median time difference between
                                       consecutive observations in the time series.
            - `freq_median_scale`: The median time difference between consecutive
                                   observations in the time series, scaled to a
                                  common unit.
            - `freq_median_unit`: The unit of the median time difference between
                                  consecutive observations in the time series.
            - `diff_min`: The minimum time difference between consecutive
                          observations in the time series as a timedelta.
            - `diff_q25`: The 25th percentile of the time difference between
                          consecutive observations in the time series as a timedelta.
            - `diff_median`: The median time difference between consecutive
                             observations in the time series as a timedelta.
            - `diff_mean`: The mean time difference between consecutive observations
                           in the time series as a timedelta.
            - `diff_q75`: The 75th percentile of the time difference between
                          consecutive observations in the time series as a timedelta.
            - `diff_max`: The maximum time difference between consecutive
                          observations in the time series as a timedelta.
            - `diff_min_seconds`: The minimum time difference between consecutive
                                  observations in the time series in seconds.
            - `diff_q25_seconds`: The 25th percentile of the time difference between
                                  consecutive observations in the time series in
                                  seconds.
            - `diff_median_seconds`: The median time difference between consecutive
                                     observations in the time series in seconds.
            - `diff_mean_seconds`: The mean time difference between consecutive
                                   observations in the time series in seconds.
            - `diff_q75_seconds`: The 75th percentile of the time difference between
                                  consecutive observations in the time series in seconds.
            - `diff_max_seconds`: The maximum time difference between consecutive
                                  observations in the time series in seconds.
        
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
        import pytimetk as tk
        import pandas as pd
        
        dates = pd.to_datetime(["2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-09", "2023-10-10"])
        df = pd.DataFrame(dates, columns = ["date"])
        
        df.ts_summary(date_column = 'date')
        ```
        
        ```{python}
        # Grouped ts_summary
        df = tk.load_dataset('stocks_daily', parse_dates = ['date'])
        
        df.groupby('symbol').ts_summary(date_column = 'date')
        ```
        
        ```{python}
        # Parallelized grouped ts_summary
        (
            df
                .groupby('symbol')
                .ts_summary(
                    date_column = 'date',
                    threads = 2,
                    show_progress = True
                )
        )
        ```
        """
        ...

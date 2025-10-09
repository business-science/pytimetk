from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy as _BaseGroupBy

class DataFrameGroupBy(
    _BaseGroupBy,
):

    def agg(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Aggregate using one or more operations over the specified axis.
        
        Parameters
        ----------
        func : function, str, list, dict or None
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.
        
            Accepted combinations are:
        
            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
            - None, in which case ``**kwargs`` are used with Named Aggregation. Here the
              output has one column for each element in ``**kwargs``. The name of the
              column is keyword, whereas the value determines the aggregation used to compute
              the values in the column.
        
              Can also accept a Numba JIT function with
              ``engine='numba'`` specified. Only passing a single function is supported
              with this engine.
        
              If the ``'numba'`` engine is chosen, the function must be
              a user defined function with ``values`` and ``index`` as the
              first and second arguments respectively in the function signature.
              Each group's index will be passed to the user defined function
              and optionally available for use.
        
        *args
            Positional arguments to pass to func.
        engine : str, default None
            * ``'cython'`` : Runs the function through C-extensions from cython.
            * ``'numba'`` : Runs the function through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function
        
        **kwargs
            * If ``func`` is None, ``**kwargs`` are used to define the output names and
              aggregations via Named Aggregation. See ``func`` entry.
            * Otherwise, keyword arguments to be passed into func.
        
        Returns
        -------
        DataFrame
        
        See Also
        --------
        DataFrame.groupby.apply : Apply function func group-wise
            and combine the results together.
        DataFrame.groupby.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate : Aggregate using one or more
            operations over the specified axis.
        
        Notes
        -----
        When using ``engine='numba'``, there will be no "fall back" behavior internally.
        The group data and group index will be passed as numpy arrays to the JITed
        user defined function, and no alternative execution attempts will be tried.
        
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.
        
        Examples
        --------
        >>> data = {"A": [1, 1, 2, 2],
        ...         "B": [1, 2, 3, 4],
        ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}
        >>> df = pd.DataFrame(data)
        >>> df
           A  B         C
        0  1  1  0.362838
        1  1  2  0.227877
        2  2  3  1.267767
        3  2  4 -0.562860
        
        The aggregation is for each column.
        
        >>> df.groupby('A').agg('min')
           B         C
        A
        1  1  0.227877
        2  3 -0.562860
        
        Multiple aggregations
        
        >>> df.groupby('A').agg(['min', 'max'])
            B             C
          min max       min       max
        A
        1   1   2  0.227877  0.362838
        2   3   4 -0.562860  1.267767
        
        Select a column for aggregation
        
        >>> df.groupby('A').B.agg(['min', 'max'])
           min  max
        A
        1    1    2
        2    3    4
        
        User-defined function for aggregation
        
        >>> df.groupby('A').agg(lambda x: sum(x) + 2)
            B          C
        A
        1       5       2.590715
        2       9       2.704907
        
        Different aggregations per column
        
        >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
            B             C
          min max       sum
        A
        1   1   2  0.590715
        2   3   4  0.704907
        
        To control the output names with different aggregations per column,
        pandas supports "named aggregation"
        
        >>> df.groupby("A").agg(
        ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
        ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")
        ... )
           b_min     c_sum
        A
        1      1  0.590715
        2      3  0.704907
        
        - The keywords are the *output* column names
        - The values are tuples whose first element is the column to select
          and the second element is the aggregation to apply to that column.
          Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
          ``['column', 'aggfunc']`` to make it clearer what the arguments are.
          As usual, the aggregation can be a callable or a string alias.
        
        See :ref:`groupby.aggregate.named` for more.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the aggregating function.
        
        >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
              B
        A
        1   1.0
        2   3.0
        """
        ...
    def aggregate(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Aggregate using one or more operations over the specified axis.
        
        Parameters
        ----------
        func : function, str, list, dict or None
            Function to use for aggregating the data. If a function, must either
            work when passed a DataFrame or when passed to DataFrame.apply.
        
            Accepted combinations are:
        
            - function
            - string function name
            - list of functions and/or function names, e.g. ``[np.sum, 'mean']``
            - dict of axis labels -> functions, function names or list of such.
            - None, in which case ``**kwargs`` are used with Named Aggregation. Here the
              output has one column for each element in ``**kwargs``. The name of the
              column is keyword, whereas the value determines the aggregation used to compute
              the values in the column.
        
              Can also accept a Numba JIT function with
              ``engine='numba'`` specified. Only passing a single function is supported
              with this engine.
        
              If the ``'numba'`` engine is chosen, the function must be
              a user defined function with ``values`` and ``index`` as the
              first and second arguments respectively in the function signature.
              Each group's index will be passed to the user defined function
              and optionally available for use.
        
        *args
            Positional arguments to pass to func.
        engine : str, default None
            * ``'cython'`` : Runs the function through C-extensions from cython.
            * ``'numba'`` : Runs the function through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function
        
        **kwargs
            * If ``func`` is None, ``**kwargs`` are used to define the output names and
              aggregations via Named Aggregation. See ``func`` entry.
            * Otherwise, keyword arguments to be passed into func.
        
        Returns
        -------
        DataFrame
        
        See Also
        --------
        DataFrame.groupby.apply : Apply function func group-wise
            and combine the results together.
        DataFrame.groupby.transform : Transforms the Series on each group
            based on the given function.
        DataFrame.aggregate : Aggregate using one or more
            operations over the specified axis.
        
        Notes
        -----
        When using ``engine='numba'``, there will be no "fall back" behavior internally.
        The group data and group index will be passed as numpy arrays to the JITed
        user defined function, and no alternative execution attempts will be tried.
        
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.
        
        Examples
        --------
        >>> data = {"A": [1, 1, 2, 2],
        ...         "B": [1, 2, 3, 4],
        ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}
        >>> df = pd.DataFrame(data)
        >>> df
           A  B         C
        0  1  1  0.362838
        1  1  2  0.227877
        2  2  3  1.267767
        3  2  4 -0.562860
        
        The aggregation is for each column.
        
        >>> df.groupby('A').agg('min')
           B         C
        A
        1  1  0.227877
        2  3 -0.562860
        
        Multiple aggregations
        
        >>> df.groupby('A').agg(['min', 'max'])
            B             C
          min max       min       max
        A
        1   1   2  0.227877  0.362838
        2   3   4 -0.562860  1.267767
        
        Select a column for aggregation
        
        >>> df.groupby('A').B.agg(['min', 'max'])
           min  max
        A
        1    1    2
        2    3    4
        
        User-defined function for aggregation
        
        >>> df.groupby('A').agg(lambda x: sum(x) + 2)
            B          C
        A
        1       5       2.590715
        2       9       2.704907
        
        Different aggregations per column
        
        >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
            B             C
          min max       sum
        A
        1   1   2  0.590715
        2   3   4  0.704907
        
        To control the output names with different aggregations per column,
        pandas supports "named aggregation"
        
        >>> df.groupby("A").agg(
        ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
        ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")
        ... )
           b_min     c_sum
        A
        1      1  0.590715
        2      3  0.704907
        
        - The keywords are the *output* column names
        - The values are tuples whose first element is the column to select
          and the second element is the aggregation to apply to that column.
          Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
          ``['column', 'aggfunc']`` to make it clearer what the arguments are.
          As usual, the aggregation can be a callable or a string alias.
        
        See :ref:`groupby.aggregate.named` for more.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the aggregating function.
        
        >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
              B
        A
        1   1.0
        2   3.0
        """
        ...
    def all(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return True if all values in the group are truthful, else False.
        
        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.
        
        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if all elements
            are True within its respective group, False otherwise.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).all()
        a     True
        b    False
        dtype: bool
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 0, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  5  6
        parrot   7  8  9
        >>> df.groupby(by=["a"]).all()
               b      c
        a
        1  False   True
        7   True   True
        """
        ...
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
    def any(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return True if any value in the group is truthful, else False.
        
        Parameters
        ----------
        skipna : bool, default True
            Flag to ignore nan values during truth testing.
        
        Returns
        -------
        Series or DataFrame
            DataFrame or Series of boolean values, where a value is True if any element
            is True within its respective group, False otherwise.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 0], index=lst)
        >>> ser
        a    1
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).any()
        a     True
        b    False
        dtype: bool
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 0, 3], [1, 0, 6], [7, 1, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["ostrich", "penguin", "parrot"])
        >>> df
                 a  b  c
        ostrich  1  0  3
        penguin  1  0  6
        parrot   7  1  9
        >>> df.groupby(by=["a"]).any()
               b      c
        a
        1  False   True
        7   True   True
        """
        ...
    def apply(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply function ``func`` group-wise and combine the results together.
        
        The function passed to ``apply`` must take a dataframe as its first
        argument and return a DataFrame, Series or scalar. ``apply`` will
        then take care of combining the results back together into a single
        dataframe or series. ``apply`` is therefore a highly flexible
        grouping method.
        
        While ``apply`` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like ``agg`` or ``transform``. Pandas offers a wide range of method that will
        be much faster than using ``apply`` for their specific purposes, so try to
        use them before reaching for ``apply``.
        
        Parameters
        ----------
        func : callable
            A callable that takes a dataframe as its first argument, and
            returns a dataframe, a series or a scalar. In addition the
            callable may take positional and keyword arguments.
        include_groups : bool, default True
            When True, will attempt to apply ``func`` to the groupings in
            the case that they are columns of the DataFrame. If this raises a
            TypeError, the result will be computed with the groupings excluded.
            When False, the groupings will be excluded when applying ``func``.
        
            .. versionadded:: 2.2.0
        
            .. deprecated:: 2.2.0
        
               Setting include_groups to True is deprecated. Only the value
               False will be allowed in a future version of pandas.
        
        args, kwargs : tuple and dict
            Optional positional and keyword arguments to pass to ``func``.
        
        Returns
        -------
        Series or DataFrame
        
        See Also
        --------
        pipe : Apply function to the full GroupBy object instead of to each
            group.
        aggregate : Apply aggregate function to the GroupBy object.
        transform : Apply function column-by-column to the GroupBy object.
        Series.apply : Apply a function to a Series.
        DataFrame.apply : Apply a function to each row or column of a DataFrame.
        
        Notes
        -----
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.
        
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.
        
        Examples
        --------
        
        >>> df = pd.DataFrame({'A': 'a a b'.split(),
        ...                    'B': [1, 2, 3],
        ...                    'C': [4, 6, 5]})
        >>> g1 = df.groupby('A', group_keys=False)
        >>> g2 = df.groupby('A', group_keys=True)
        
        Notice that ``g1`` and ``g2`` have two groups, ``a`` and ``b``, and only
        differ in their ``group_keys`` argument. Calling `apply` in various ways,
        we can get different grouping results:
        
        Example 1: below the function passed to `apply` takes a DataFrame as
        its argument and returns a DataFrame. `apply` combines the result for
        each group together into a new DataFrame:
        
        >>> g1[['B', 'C']].apply(lambda x: x / x.sum())
                  B    C
        0  0.333333  0.4
        1  0.666667  0.6
        2  1.000000  1.0
        
        In the above, the groups are not part of the index. We can have them included
        by using ``g2`` where ``group_keys=True``:
        
        >>> g2[['B', 'C']].apply(lambda x: x / x.sum())
                    B    C
        A
        a 0  0.333333  0.4
          1  0.666667  0.6
        b 2  1.000000  1.0
        
        Example 2: The function passed to `apply` takes a DataFrame as
        its argument and returns a Series.  `apply` combines the result for
        each group together into a new DataFrame.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the passed ``func``.
        
        >>> g1[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
             B    C
        A
        a  1.0  2.0
        b  0.0  0.0
        
        >>> g2[['B', 'C']].apply(lambda x: x.astype(float).max() - x.min())
             B    C
        A
        a  1.0  2.0
        b  0.0  0.0
        
        The ``group_keys`` argument has no effect here because the result is not
        like-indexed (i.e. :ref:`a transform <groupby.transform>`) when compared
        to the input.
        
        Example 3: The function passed to `apply` takes a DataFrame as
        its argument and returns a scalar. `apply` combines the result for
        each group together into a Series, including setting the index as
        appropriate:
        
        >>> g1.apply(lambda x: x.C.max() - x.B.min(), include_groups=False)
        A
        a    5
        b    2
        dtype: int64
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
    def bfill(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Backward fill the values.
        
        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.
        
        Returns
        -------
        Series or DataFrame
            Object with missing values filled.
        
        See Also
        --------
        Series.bfill :  Backward fill the missing values in the dataset.
        DataFrame.bfill:  Backward fill the missing values in the dataset.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        
        Examples
        --------
        
        With Series:
        
        >>> index = ['Falcon', 'Falcon', 'Parrot', 'Parrot', 'Parrot']
        >>> s = pd.Series([None, 1, None, None, 3], index=index)
        >>> s
        Falcon    NaN
        Falcon    1.0
        Parrot    NaN
        Parrot    NaN
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill()
        Falcon    1.0
        Falcon    1.0
        Parrot    3.0
        Parrot    3.0
        Parrot    3.0
        dtype: float64
        >>> s.groupby(level=0).bfill(limit=1)
        Falcon    1.0
        Falcon    1.0
        Parrot    NaN
        Parrot    3.0
        Parrot    3.0
        dtype: float64
        
        With DataFrame:
        
        >>> df = pd.DataFrame({'A': [1, None, None, None, 4],
        ...                    'B': [None, None, 5, None, 7]}, index=index)
        >>> df
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  NaN       5.0
        Parrot  NaN       NaN
        Parrot  4.0       7.0
        >>> df.groupby(level=0).bfill()
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  4.0       5.0
        Parrot  4.0       7.0
        Parrot  4.0       7.0
        >>> df.groupby(level=0).bfill(limit=1)
                  A         B
        Falcon  1.0       NaN
        Falcon  NaN       NaN
        Parrot  NaN       5.0
        Parrot  4.0       7.0
        Parrot  4.0       7.0
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
    def boxplot(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Make box plots from DataFrameGroupBy data.
        
        Parameters
        ----------
        grouped : Grouped DataFrame
        subplots : bool
            * ``False`` - no subplots will be used
            * ``True`` - create a subplot for each group.
        
        column : column name or list of names, or vector
            Can be any valid input to groupby.
        fontsize : float or str
        rot : label rotation angle
        grid : Setting this to True will show the grid
        ax : Matplotlib axis object, default None
        figsize : A tuple (width, height) in inches
        layout : tuple (optional)
            The layout of the plot: (rows, columns).
        sharex : bool, default False
            Whether x-axes will be shared among subplots.
        sharey : bool, default True
            Whether y-axes will be shared among subplots.
        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
            specify the ``plotting.backend`` for the whole session, set
            ``pd.options.plotting.backend``.
        **kwargs
            All other plotting keyword arguments to be passed to
            matplotlib's boxplot function.
        
        Returns
        -------
        dict of key/value = group key/DataFrame.boxplot return value
        or DataFrame.boxplot return value in case subplots=figures=False
        
        Examples
        --------
        You can create boxplots for grouped data and show them as separate subplots:
        
        .. plot::
            :context: close-figs
        
            >>> import itertools
            >>> tuples = [t for t in itertools.product(range(1000), range(4))]
            >>> index = pd.MultiIndex.from_tuples(tuples, names=['lvl0', 'lvl1'])
            >>> data = np.random.randn(len(index), 4)
            >>> df = pd.DataFrame(data, columns=list('ABCD'), index=index)
            >>> grouped = df.groupby(level='lvl1')
            >>> grouped.boxplot(rot=45, fontsize=12, figsize=(8, 10))  # doctest: +SKIP
        
        The ``subplots=False`` option shows the boxplots in a single figure.
        
        .. plot::
            :context: close-figs
        
            >>> grouped.boxplot(subplots=False, rot=45, fontsize=12)  # doctest: +SKIP
        """
        ...
    def corr(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute pairwise correlation of columns, excluding NA/null values.
        
        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:
        
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float. Note that the returned matrix from corr
                will have 1 along the diagonals and will be symmetric
                regardless of the callable's behavior.
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result. Currently only available for Pearson
            and Spearman correlation.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.
        
        Returns
        -------
        DataFrame
            Correlation matrix.
        
        See Also
        --------
        DataFrame.corrwith : Compute pairwise correlation with another
            DataFrame or Series.
        Series.corr : Compute the correlation between two Series.
        
        Notes
        -----
        Pearson, Kendall and Spearman correlation are currently computed using pairwise complete observations.
        
        * `Pearson correlation coefficient <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_
        * `Kendall rank correlation coefficient <https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient>`_
        * `Spearman's rank correlation coefficient <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
        
        Examples
        --------
        >>> def histogram_intersection(a, b):
        ...     v = np.minimum(a, b).sum().round(decimals=1)
        ...     return v
        >>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(method=histogram_intersection)
              dogs  cats
        dogs   1.0   0.3
        cats   0.3   1.0
        
        >>> df = pd.DataFrame([(1, 1), (2, np.nan), (np.nan, 3), (4, 4)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr(min_periods=3)
              dogs  cats
        dogs   1.0   NaN
        cats   NaN   1.0
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
    def corrwith(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute pairwise correlation.
        
        Pairwise correlation is computed between rows or columns of
        DataFrame with rows or columns of Series or DataFrame. DataFrames
        are first aligned along both axes before computing the
        correlations.
        
        Parameters
        ----------
        other : DataFrame, Series
            Object with which to compute correlations.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' to compute row-wise, 1 or 'columns' for
            column-wise.
        drop : bool, default False
            Drop missing indices from result.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:
        
            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float.
        
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.
        
        Returns
        -------
        Series
            Pairwise correlations.
        
        See Also
        --------
        DataFrame.corr : Compute pairwise correlation of columns.
        
        Examples
        --------
        >>> index = ["a", "b", "c", "d", "e"]
        >>> columns = ["one", "two", "three", "four"]
        >>> df1 = pd.DataFrame(np.arange(20).reshape(5, 4), index=index, columns=columns)
        >>> df2 = pd.DataFrame(np.arange(16).reshape(4, 4), index=index[:4], columns=columns)
        >>> df1.corrwith(df2)
        one      1.0
        two      1.0
        three    1.0
        four     1.0
        dtype: float64
        
        >>> df2.corrwith(df1, axis=1)
        a    1.0
        b    1.0
        c    1.0
        d    1.0
        e    NaN
        dtype: float64
        """
        ...
    def count(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute count of group, excluding missing values.
        
        Returns
        -------
        Series or DataFrame
            Count of values within each group.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, np.nan], index=lst)
        >>> ser
        a    1.0
        a    2.0
        b    NaN
        dtype: float64
        >>> ser.groupby(level=0).count()
        a    2
        b    0
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, np.nan, 3], [1, np.nan, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a         b     c
        cow     1       NaN     3
        horse   1       NaN     6
        bull    7       8.0     9
        >>> df.groupby("a").count()
            b   c
        a
        1   0   2
        7   1   1
        
        For Resampler:
        
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').count()
        2023-01-01    2
        2023-02-01    2
        Freq: MS, dtype: int64
        """
        ...
    def cov(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute pairwise covariance of columns, excluding NA/null values.
        
        Compute the pairwise covariance among the series of a DataFrame.
        The returned data frame is the `covariance matrix
        <https://en.wikipedia.org/wiki/Covariance_matrix>`__ of the columns
        of the DataFrame.
        
        Both NA and null values are automatically excluded from the
        calculation. (See the note below about bias from missing values.)
        A threshold can be set for the minimum number of
        observations for each value created. Comparisons with observations
        below this threshold will be returned as ``NaN``.
        
        This method is generally used for the analysis of time series data to
        understand the relationship between different measures
        across time.
        
        Parameters
        ----------
        min_periods : int, optional
            Minimum number of observations required per pair of columns
            to have a valid result.
        
        ddof : int, default 1
            Delta degrees of freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
            This argument is applicable only when no ``nan`` is in the dataframe.
        
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.
        
        Returns
        -------
        DataFrame
            The covariance matrix of the series of the DataFrame.
        
        See Also
        --------
        Series.cov : Compute covariance with another Series.
        core.window.ewm.ExponentialMovingWindow.cov : Exponential weighted sample
            covariance.
        core.window.expanding.Expanding.cov : Expanding sample covariance.
        core.window.rolling.Rolling.cov : Rolling sample covariance.
        
        Notes
        -----
        Returns the covariance matrix of the DataFrame's time series.
        The covariance is normalized by N-ddof.
        
        For DataFrames that have Series that are missing data (assuming that
        data is `missing at random
        <https://en.wikipedia.org/wiki/Missing_data#Missing_at_random>`__)
        the returned covariance matrix will be an unbiased estimate
        of the variance and covariance between the member Series.
        
        However, for many applications this estimate may not be acceptable
        because the estimate covariance matrix is not guaranteed to be positive
        semi-definite. This could lead to estimate correlations having
        absolute values which are greater than one, and/or a non-invertible
        covariance matrix. See `Estimation of covariance matrices
        <https://en.wikipedia.org/w/index.php?title=Estimation_of_covariance_
        matrices>`__ for more details.
        
        Examples
        --------
        >>> df = pd.DataFrame([(1, 2), (0, 3), (2, 0), (1, 1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.cov()
                  dogs      cats
        dogs  0.666667 -1.000000
        cats -1.000000  1.666667
        
        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.randn(1000, 5),
        ...                   columns=['a', 'b', 'c', 'd', 'e'])
        >>> df.cov()
                  a         b         c         d         e
        a  0.998438 -0.020161  0.059277 -0.008943  0.014144
        b -0.020161  1.059352 -0.008543 -0.024738  0.009826
        c  0.059277 -0.008543  1.010670 -0.001486 -0.000271
        d -0.008943 -0.024738 -0.001486  0.921297 -0.013692
        e  0.014144  0.009826 -0.000271 -0.013692  0.977795
        
        **Minimum number of periods**
        
        This method also supports an optional ``min_periods`` keyword
        that specifies the required minimum number of non-NA observations for
        each column pair in order to have a valid result:
        
        >>> np.random.seed(42)
        >>> df = pd.DataFrame(np.random.randn(20, 3),
        ...                   columns=['a', 'b', 'c'])
        >>> df.loc[df.index[:5], 'a'] = np.nan
        >>> df.loc[df.index[5:10], 'b'] = np.nan
        >>> df.cov(min_periods=12)
                  a         b         c
        a  0.316741       NaN -0.150812
        b       NaN  1.248003  0.191417
        c -0.150812  0.191417  0.895202
        """
        ...
    def cumcount(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Number each item in each group from 0 to the length of that group - 1.
        
        Essentially this is equivalent to
        
        .. code-block:: python
        
            self.apply(lambda x: pd.Series(np.arange(len(x)), x.index))
        
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from length of group - 1 to 0.
        
        Returns
        -------
        Series
            Sequence number of each element within each group.
        
        See Also
        --------
        .ngroup : Number the groups themselves.
        
        Examples
        --------
        >>> df = pd.DataFrame([['a'], ['a'], ['a'], ['b'], ['b'], ['a']],
        ...                   columns=['A'])
        >>> df
           A
        0  a
        1  a
        2  a
        3  b
        4  b
        5  a
        >>> df.groupby('A').cumcount()
        0    0
        1    1
        2    2
        3    0
        4    1
        5    3
        dtype: int64
        >>> df.groupby('A').cumcount(ascending=False)
        0    3
        1    2
        2    1
        3    1
        4    0
        5    0
        dtype: int64
        """
        ...
    def cummax(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Cumulative max for each group.
        
        Returns
        -------
        Series or DataFrame
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 6, 2, 3, 1, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    1
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummax()
        a    1
        a    6
        a    6
        b    3
        b    3
        b    4
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 1, 0], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   1   0
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cummax()
                b   c
        cow     8   2
        horse   8   2
        bull    6   9
        """
        ...
    def cummin(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Cumulative min for each group.
        
        Returns
        -------
        Series or DataFrame
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([1, 6, 2, 3, 0, 4], index=lst)
        >>> ser
        a    1
        a    6
        a    2
        b    3
        b    0
        b    4
        dtype: int64
        >>> ser.groupby(level=0).cummin()
        a    1
        a    1
        a    1
        b    3
        b    0
        b    0
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 0, 2], [1, 1, 5], [6, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["snake", "rabbit", "turtle"])
        >>> df
                a   b   c
        snake   1   0   2
        rabbit  1   1   5
        turtle  6   6   9
        >>> df.groupby("a").groups
        {1: ['snake', 'rabbit'], 6: ['turtle']}
        >>> df.groupby("a").cummin()
                b   c
        snake   0   2
        rabbit  0   2
        turtle  6   9
        """
        ...
    def cumprod(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Cumulative product for each group.
        
        Returns
        -------
        Series or DataFrame
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumprod()
        a    6
        a   12
        b    0
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["cow", "horse", "bull"])
        >>> df
                a   b   c
        cow     1   8   2
        horse   1   2   5
        bull    2   6   9
        >>> df.groupby("a").groups
        {1: ['cow', 'horse'], 2: ['bull']}
        >>> df.groupby("a").cumprod()
                b   c
        cow     8   2
        horse  16  10
        bull    6   9
        """
        ...
    def cumsum(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Cumulative sum for each group.
        
        Returns
        -------
        Series or DataFrame
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([6, 2, 0], index=lst)
        >>> ser
        a    6
        a    2
        b    0
        dtype: int64
        >>> ser.groupby(level=0).cumsum()
        a    6
        a    8
        b    0
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["fox", "gorilla", "lion"])
        >>> df
                  a   b   c
        fox       1   8   2
        gorilla   1   2   5
        lion      2   6   9
        >>> df.groupby("a").groups
        {1: ['fox', 'gorilla'], 2: ['lion']}
        >>> df.groupby("a").cumsum()
                  b   c
        fox       8   2
        gorilla  10   7
        lion      6   9
        """
        ...
    def describe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Generate descriptive statistics.
        
        Descriptive statistics include those that summarize the central
        tendency, dispersion and shape of a
        dataset's distribution, excluding ``NaN`` values.
        
        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.
        
        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output. All should
            fall between 0 and 1. The default is
            ``[.25, .5, .75]``, which returns the 25th, 50th, and
            75th percentiles.
        include : 'all', list-like of dtypes or None (default), optional
            A white list of data types to include in the result. Ignored
            for ``Series``. Here are the options:
        
            - 'all' : All columns of the input will be included in the output.
            - A list-like of dtypes : Limits the results to the
              provided data types.
              To limit the result to numeric types submit
              ``numpy.number``. To limit it instead to object columns submit
              the ``numpy.object`` data type. Strings
              can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              select pandas categorical columns, use ``'category'``
            - None (default) : The result will include all numeric columns.
        exclude : list-like of dtypes or None (default), optional,
            A black list of data types to omit from the result. Ignored
            for ``Series``. Here are the options:
        
            - A list-like of dtypes : Excludes the provided data types
              from the result. To exclude numeric types submit
              ``numpy.number``. To exclude object columns submit the data
              type ``numpy.object``. Strings can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(exclude=['O'])``). To
              exclude pandas categorical columns, use ``'category'``
            - None (default) : The result will exclude nothing.
        
        Returns
        -------
        Series or DataFrame
            Summary statistics of the Series or Dataframe provided.
        
        See Also
        --------
        DataFrame.count: Count number of non-NA/null observations.
        DataFrame.max: Maximum of the values in the object.
        DataFrame.min: Minimum of the values in the object.
        DataFrame.mean: Mean of the values.
        DataFrame.std: Standard deviation of the observations.
        DataFrame.select_dtypes: Subset of a DataFrame including/excluding
            columns based on their dtype.
        
        Notes
        -----
        For numeric data, the result's index will include ``count``,
        ``mean``, ``std``, ``min``, ``max`` as well as lower, ``50`` and
        upper percentiles. By default the lower percentile is ``25`` and the
        upper percentile is ``75``. The ``50`` percentile is the
        same as the median.
        
        For object data (e.g. strings or timestamps), the result's index
        will include ``count``, ``unique``, ``top``, and ``freq``. The ``top``
        is the most common value. The ``freq`` is the most common value's
        frequency. Timestamps also include the ``first`` and ``last`` items.
        
        If multiple object values have the highest count, then the
        ``count`` and ``top`` results will be arbitrarily chosen from
        among those with the highest count.
        
        For mixed data types provided via a ``DataFrame``, the default is to
        return only an analysis of numeric columns. If the dataframe consists
        only of object and categorical data without any numeric columns, the
        default is to return an analysis of both the object and categorical
        columns. If ``include='all'`` is provided as an option, the result
        will include a union of attributes of each type.
        
        The `include` and `exclude` parameters can be used to limit
        which columns in a ``DataFrame`` are analyzed for the output.
        The parameters are ignored when analyzing a ``Series``.
        
        Examples
        --------
        Describing a numeric ``Series``.
        
        >>> s = pd.Series([1, 2, 3])
        >>> s.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        dtype: float64
        
        Describing a categorical ``Series``.
        
        >>> s = pd.Series(['a', 'a', 'b', 'c'])
        >>> s.describe()
        count     4
        unique    3
        top       a
        freq      2
        dtype: object
        
        Describing a timestamp ``Series``.
        
        >>> s = pd.Series([
        ...     np.datetime64("2000-01-01"),
        ...     np.datetime64("2010-01-01"),
        ...     np.datetime64("2010-01-01")
        ... ])
        >>> s.describe()
        count                      3
        mean     2006-09-01 08:00:00
        min      2000-01-01 00:00:00
        25%      2004-12-31 12:00:00
        50%      2010-01-01 00:00:00
        75%      2010-01-01 00:00:00
        max      2010-01-01 00:00:00
        dtype: object
        
        Describing a ``DataFrame``. By default only numeric fields
        are returned.
        
        >>> df = pd.DataFrame({'categorical': pd.Categorical(['d', 'e', 'f']),
        ...                    'numeric': [1, 2, 3],
        ...                    'object': ['a', 'b', 'c']
        ...                    })
        >>> df.describe()
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0
        
        Describing all columns of a ``DataFrame`` regardless of data type.
        
        >>> df.describe(include='all')  # doctest: +SKIP
               categorical  numeric object
        count            3      3.0      3
        unique           3      NaN      3
        top              f      NaN      a
        freq             1      NaN      1
        mean           NaN      2.0    NaN
        std            NaN      1.0    NaN
        min            NaN      1.0    NaN
        25%            NaN      1.5    NaN
        50%            NaN      2.0    NaN
        75%            NaN      2.5    NaN
        max            NaN      3.0    NaN
        
        Describing a column from a ``DataFrame`` by accessing it as
        an attribute.
        
        >>> df.numeric.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.5
        50%      2.0
        75%      2.5
        max      3.0
        Name: numeric, dtype: float64
        
        Including only numeric columns in a ``DataFrame`` description.
        
        >>> df.describe(include=[np.number])
               numeric
        count      3.0
        mean       2.0
        std        1.0
        min        1.0
        25%        1.5
        50%        2.0
        75%        2.5
        max        3.0
        
        Including only string columns in a ``DataFrame`` description.
        
        >>> df.describe(include=[object])  # doctest: +SKIP
               object
        count       3
        unique      3
        top         a
        freq        1
        
        Including only categorical columns from a ``DataFrame`` description.
        
        >>> df.describe(include=['category'])
               categorical
        count            3
        unique           3
        top              d
        freq             1
        
        Excluding numeric columns from a ``DataFrame`` description.
        
        >>> df.describe(exclude=[np.number])  # doctest: +SKIP
               categorical object
        count            3      3
        unique           3      3
        top              f      a
        freq             1      1
        
        Excluding object columns from a ``DataFrame`` description.
        
        >>> df.describe(exclude=[object])  # doctest: +SKIP
               categorical  numeric
        count            3      3.0
        unique           3      NaN
        top              f      NaN
        freq             1      NaN
        mean           NaN      2.0
        std            NaN      1.0
        min            NaN      1.0
        25%            NaN      1.5
        50%            NaN      2.0
        75%            NaN      2.5
        max            NaN      3.0
        """
        ...
    def diff(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        First discrete difference of element.
        
        Calculates the difference of each element compared with another
        element in the group (default is element in previous row).
        
        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.
        axis : axis to shift, default 0
            Take difference over rows (0) or columns (1).
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        Returns
        -------
        Series or DataFrame
            First differences.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).diff()
        a    NaN
        a   -5.0
        a    6.0
        b    NaN
        b   -1.0
        b    0.0
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).diff()
                 a    b
          dog  NaN  NaN
          dog  2.0  3.0
          dog  2.0  4.0
        mouse  NaN  NaN
        mouse  0.0  0.0
        mouse  1.0 -2.0
        mouse -5.0 -1.0
        """
        ...
    def ewm(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return an ewm grouper, providing ewm functionality per group.
        
        Returns
        -------
        pandas.api.typing.ExponentialMovingWindowGroupby
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        """
        ...
    def expanding(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return an expanding grouper, providing expanding
        functionality per group.
        
        Returns
        -------
        pandas.api.typing.ExpandingGroupby
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        """
        ...
    def ffill(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Forward fill the values.
        
        Parameters
        ----------
        limit : int, optional
            Limit of how many values to fill.
        
        Returns
        -------
        Series or DataFrame
            Object with missing values filled.
        
        See Also
        --------
        Series.ffill: Returns Series with minimum number of char in object.
        DataFrame.ffill: Object with missing values filled or None if inplace=True.
        Series.fillna: Fill NaN values of a Series.
        DataFrame.fillna: Fill NaN values of a DataFrame.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> key = [0, 0, 1, 1]
        >>> ser = pd.Series([np.nan, 2, 3, np.nan], index=key)
        >>> ser
        0    NaN
        0    2.0
        1    3.0
        1    NaN
        dtype: float64
        >>> ser.groupby(level=0).ffill()
        0    NaN
        0    2.0
        1    3.0
        1    3.0
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN
        
        Propagate non-null values forward or backward within each group along columns.
        
        >>> df.groupby("key").ffill()
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN 2.0
        
        Propagate non-null values forward or backward within each group along rows.
        
        >>> df.T.groupby(np.array([0, 0, 1, 1])).ffill().T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN
        
        Only replace the first NaN element within a group along rows.
        
        >>> df.groupby("key").ffill(limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        """
        ...
    def fillna(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Fill NA/NaN values using the specified method within groups.
        
        .. deprecated:: 2.2.0
            This method is deprecated and will be removed in a future version.
            Use the :meth:`.DataFrameGroupBy.ffill` or :meth:`.DataFrameGroupBy.bfill`
            for forward or backward filling instead. If you want to fill with a
            single value, use :meth:`DataFrame.fillna` instead.
        
        Parameters
        ----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list. Users wanting to use the ``value`` argument and not ``method``
            should prefer :meth:`.DataFrame.fillna` as this
            will produce the same result and be more performant.
        method : {{'bfill', 'ffill', None}}, default None
            Method to use for filling holes. ``'ffill'`` will propagate
            the last valid observation forward within a group.
            ``'bfill'`` will use next valid observation to fill the gap.
        axis : {0 or 'index', 1 or 'columns'}
            Axis along which to fill missing values. When the :class:`DataFrameGroupBy`
            ``axis`` argument is ``0``, using ``axis=1`` here will produce
            the same results as :meth:`.DataFrame.fillna`. When the
            :class:`DataFrameGroupBy` ``axis`` argument is ``1``, using ``axis=0``
            or ``axis=1`` here will produce the same results.
        inplace : bool, default False
            Broken. Do not set to True.
        limit : int, default None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill within a group. In other words,
            if there is a gap with more than this number of consecutive NaNs,
            it will only be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).
        
        Returns
        -------
        DataFrame
            Object with missing values filled.
        
        See Also
        --------
        ffill : Forward fill values within a group.
        bfill : Backward fill values within a group.
        
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "key": [0, 0, 1, 1, 1],
        ...         "A": [np.nan, 2, np.nan, 3, np.nan],
        ...         "B": [2, 3, np.nan, np.nan, np.nan],
        ...         "C": [np.nan, np.nan, 2, np.nan, np.nan],
        ...     }
        ... )
        >>> df
           key    A    B   C
        0    0  NaN  2.0 NaN
        1    0  2.0  3.0 NaN
        2    1  NaN  NaN 2.0
        3    1  3.0  NaN NaN
        4    1  NaN  NaN NaN
        
        Propagate non-null values forward or backward within each group along columns.
        
        >>> df.groupby("key").fillna(method="ffill")
             A    B   C
        0  NaN  2.0 NaN
        1  2.0  3.0 NaN
        2  NaN  NaN 2.0
        3  3.0  NaN 2.0
        4  3.0  NaN 2.0
        
        >>> df.groupby("key").fillna(method="bfill")
             A    B   C
        0  2.0  2.0 NaN
        1  2.0  3.0 NaN
        2  3.0  NaN 2.0
        3  3.0  NaN NaN
        4  NaN  NaN NaN
        
        Propagate non-null values forward or backward within each group along rows.
        
        >>> df.T.groupby(np.array([0, 0, 1, 1])).fillna(method="ffill").T
           key    A    B    C
        0  0.0  0.0  2.0  2.0
        1  0.0  2.0  3.0  3.0
        2  1.0  1.0  NaN  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  1.0  NaN  NaN
        
        >>> df.T.groupby(np.array([0, 0, 1, 1])).fillna(method="bfill").T
           key    A    B    C
        0  0.0  NaN  2.0  NaN
        1  0.0  2.0  3.0  NaN
        2  1.0  NaN  2.0  2.0
        3  1.0  3.0  NaN  NaN
        4  1.0  NaN  NaN  NaN
        
        Only replace the first NaN element within a group along rows.
        
        >>> df.groupby("key").fillna(method="ffill", limit=1)
             A    B    C
        0  NaN  2.0  NaN
        1  2.0  3.0  NaN
        2  NaN  NaN  2.0
        3  3.0  NaN  2.0
        4  3.0  NaN  NaN
        """
        ...
    def filter(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Filter elements from groups that don't satisfy a criterion.
        
        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.
        
        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.
        
        Returns
        -------
        DataFrame
        
        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.
        
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.
        
        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
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
    def first(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute the first entry of each column within each group.
        
        Defaults to skipping NA elements.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        
            .. versionadded:: 2.2.1
        
        Returns
        -------
        Series or DataFrame
            First values within each group.
        
        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.last : Compute the last non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.
        
        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[None, 5, 6], C=[1, 2, 3],
        ...                        D=['3/11/2000', '3/12/2000', '3/13/2000']))
        >>> df['D'] = pd.to_datetime(df['D'])
        >>> df.groupby("A").first()
             B  C          D
        A
        1  5.0  1 2000-03-11
        3  6.0  3 2000-03-13
        >>> df.groupby("A").first(min_count=2)
            B    C          D
        A
        1 NaN  1.0 2000-03-11
        3 NaN  NaN        NaT
        >>> df.groupby("A").first(numeric_only=True)
             B  C
        A
        1  5.0  1
        3  6.0  3
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
    def get_group(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Construct DataFrame from group with provided name.
        
        Parameters
        ----------
        name : object
            The name of the group to get as a DataFrame.
        obj : DataFrame, default None
            The DataFrame to take the DataFrame out of.  If
            it is None, the object groupby was called on will
            be used.
        
            .. deprecated:: 2.1.0
                The obj is deprecated and will be removed in a future version.
                Do ``df.iloc[gb.indices.get(name)]``
                instead of ``gb.get_group(name, obj=df)``.
        
        Returns
        -------
        same type as obj
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        dtype: int64
        >>> ser.groupby(level=0).get_group("a")
        a    1
        a    2
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby(by=["a"]).get_group((1,))
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        
        For Resampler:
        
        >>> ser = pd.Series([1, 2, 3, 4], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01', '2023-02-15']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64
        >>> ser.resample('MS').get_group('2023-01-01')
        2023-01-01    1
        2023-01-15    2
        dtype: int64
        """
        ...
    def head(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return first n rows of each group.
        
        Similar to ``.apply(lambda x: x.head(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).
        
        Parameters
        ----------
        n : int
            If positive: number of entries to include from start of each group.
            If negative: number of entries to exclude from end of each group.
        
        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        
        >>> df = pd.DataFrame([[1, 2], [1, 4], [5, 6]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').head(1)
           A  B
        0  1  2
        2  5  6
        >>> df.groupby('A').head(-1)
           A  B
        0  1  2
        """
        ...
    def hist(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Make a histogram of the DataFrame's columns.
        
        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`matplotlib.pyplot.hist`, on each series in
        the DataFrame, resulting in one histogram per column.
        
        .. _histogram: https://en.wikipedia.org/wiki/Histogram
        
        Parameters
        ----------
        data : DataFrame
            The pandas object holding the data.
        column : str or sequence, optional
            If passed, will be used to limit data to a subset of columns.
        by : object, optional
            If passed, then used to form histograms for separate groups.
        grid : bool, default True
            Whether to show axis grid lines.
        xlabelsize : int, default None
            If specified changes the x-axis label size.
        xrot : float, default None
            Rotation of x axis labels. For example, a value of 90 displays the
            x labels rotated 90 degrees clockwise.
        ylabelsize : int, default None
            If specified changes the y-axis label size.
        yrot : float, default None
            Rotation of y axis labels. For example, a value of 90 displays the
            y labels rotated 90 degrees clockwise.
        ax : Matplotlib axes object, default None
            The axes to plot the histogram on.
        sharex : bool, default True if ax is None else False
            In case subplots=True, share x axis and set some x axis labels to
            invisible; defaults to True if ax is None otherwise False if an ax
            is passed in.
            Note that passing in both an ax and sharex=True will alter all x axis
            labels for all subplots in a figure.
        sharey : bool, default False
            In case subplots=True, share y axis and set some y axis labels to
            invisible.
        figsize : tuple, optional
            The size in inches of the figure to create. Uses the value in
            `matplotlib.rcParams` by default.
        layout : tuple, optional
            Tuple of (rows, columns) for the layout of the histograms.
        bins : int or sequence, default 10
            Number of histogram bins to be used. If an integer is given, bins + 1
            bin edges are calculated and returned. If bins is a sequence, gives
            bin edges, including left edge of first bin and right edge of last
            bin. In this case, bins is returned unmodified.
        
        backend : str, default None
            Backend to use instead of the backend specified in the option
            ``plotting.backend``. For instance, 'matplotlib'. Alternatively, to
            specify the ``plotting.backend`` for the whole session, set
            ``pd.options.plotting.backend``.
        
        legend : bool, default False
            Whether to show the legend.
        
        **kwargs
            All other plotting keyword arguments to be passed to
            :meth:`matplotlib.pyplot.hist`.
        
        Returns
        -------
        matplotlib.AxesSubplot or numpy.ndarray of them
        
        See Also
        --------
        matplotlib.pyplot.hist : Plot a histogram using matplotlib.
        
        Examples
        --------
        This example draws a histogram based on the length and width of
        some animals, displayed in three bins
        
        .. plot::
            :context: close-figs
        
            >>> data = {'length': [1.5, 0.5, 1.2, 0.9, 3],
            ...         'width': [0.7, 0.2, 0.15, 0.2, 1.1]}
            >>> index = ['pig', 'rabbit', 'duck', 'chicken', 'horse']
            >>> df = pd.DataFrame(data, index=index)
            >>> hist = df.hist(bins=3)
        """
        ...
    def idxmax(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return index of first occurrence of maximum over requested axis.
        
        NA/null values are excluded.
        
        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.
        
            .. versionchanged:: 2.0.0
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
        Returns
        -------
        Series
            Indexes of maxima along the specified axis.
        
        Raises
        ------
        ValueError
            * If the row/column is empty
        
        See Also
        --------
        Series.idxmax : Return index of the maximum element.
        
        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmax``.
        
        Examples
        --------
        Consider a dataset containing food consumption in Argentina.
        
        >>> df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
        ...                    'co2_emissions': [37.2, 19.66, 1712]},
        ...                   index=['Pork', 'Wheat Products', 'Beef'])
        
        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00
        
        By default, it returns the index for the maximum value in each column.
        
        >>> df.idxmax()
        consumption     Wheat Products
        co2_emissions             Beef
        dtype: object
        
        To return the index for the maximum value in each row, use ``axis="columns"``.
        
        >>> df.idxmax(axis="columns")
        Pork              co2_emissions
        Wheat Products     consumption
        Beef              co2_emissions
        dtype: object
        """
        ...
    def idxmin(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return index of first occurrence of minimum over requested axis.
        
        NA/null values are excluded.
        
        Parameters
        ----------
        axis : {{0 or 'index', 1 or 'columns'}}, default None
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            If axis is not provided, grouper's axis is used.
        
            .. versionchanged:: 2.0.0
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
        Returns
        -------
        Series
            Indexes of minima along the specified axis.
        
        Raises
        ------
        ValueError
            * If the row/column is empty
        
        See Also
        --------
        Series.idxmin : Return index of the minimum element.
        
        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmin``.
        
        Examples
        --------
        Consider a dataset containing food consumption in Argentina.
        
        >>> df = pd.DataFrame({'consumption': [10.51, 103.11, 55.48],
        ...                    'co2_emissions': [37.2, 19.66, 1712]},
        ...                   index=['Pork', 'Wheat Products', 'Beef'])
        
        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00
        
        By default, it returns the index for the minimum value in each column.
        
        >>> df.idxmin()
        consumption                Pork
        co2_emissions    Wheat Products
        dtype: object
        
        To return the index for the minimum value in each row, use ``axis="columns"``.
        
        >>> df.idxmin(axis="columns")
        Pork                consumption
        Wheat Products    co2_emissions
        Beef                consumption
        dtype: object
        """
        ...
    def last(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute the last entry of each column within each group.
        
        Defaults to skipping NA elements.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns. If None, will attempt to use
            everything, then use only numeric data.
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` valid values are present the result will be NA.
        skipna : bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        
            .. versionadded:: 2.2.1
        
        Returns
        -------
        Series or DataFrame
            Last of values within each group.
        
        See Also
        --------
        DataFrame.groupby : Apply a function groupby to each row or column of a
            DataFrame.
        pandas.core.groupby.DataFrameGroupBy.first : Compute the first non-null entry
            of each column.
        pandas.core.groupby.DataFrameGroupBy.nth : Take the nth row from each group.
        
        Examples
        --------
        >>> df = pd.DataFrame(dict(A=[1, 1, 3], B=[5, None, 6], C=[1, 2, 3]))
        >>> df.groupby("A").last()
             B  C
        A
        1  5.0  2
        3  6.0  3
        """
        ...
    def max(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute max of group values.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None``.
        
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        
        engine : str, default None None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``
        
        engine_kwargs : dict, default None None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
                ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.
        
        Returns
        -------
        Series or DataFrame
            Computed max of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).max()
        a    2
        b    4
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").max()
            b  c
        a
        1   8  5
        2   6  9
        """
        ...
    def mean(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute mean of groups, excluding missing values.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None`` and defaults to ``False``.
        
        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``
        
            .. versionadded:: 1.4.0
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``
        
            .. versionadded:: 1.4.0
        
        Returns
        -------
        pandas.Series or pandas.DataFrame
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])
        
        Groupby one column and return the mean of the remaining columns in
        each group.
        
        >>> df.groupby('A').mean()
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000
        
        Groupby two columns and return the mean of the remaining column.
        
        >>> df.groupby(['A', 'B']).mean()
                 C
        A B
        1 2.0  2.0
          4.0  1.0
        2 3.0  1.0
          5.0  2.0
        
        Groupby one column and return the mean of only particular column in
        the group.
        
        >>> df.groupby('A')['B'].mean()
        A
        1    3.0
        2    4.0
        Name: B, dtype: float64
        """
        ...
    def median(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute median of groups, excluding missing values.
        
        For multiple groupings, the result index will be a MultiIndex
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None`` and defaults to False.
        
        Returns
        -------
        Series or DataFrame
            Median of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).median()
        a    7.0
        b    3.0
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).median()
                 a    b
        dog    3.0  4.0
        mouse  7.0  3.0
        
        For Resampler:
        
        >>> ser = pd.Series([1, 2, 3, 3, 4, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').median()
        2023-01-01    2.0
        2023-02-01    4.0
        Freq: MS, dtype: float64
        """
        ...
    def min(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute min of group values.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None``.
        
        min_count : int, default -1
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        
        engine : str, default None None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``
        
        engine_kwargs : dict, default None None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
                ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.
        
        Returns
        -------
        Series or DataFrame
            Computed min of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).min()
        a    1
        b    3
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").min()
            b  c
        a
        1   2  2
        2   5  8
        """
        ...
    def ngroup(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Number each group from 0 to the number of groups - 1.
        
        This is the enumerative complement of cumcount.  Note that the
        numbers given to the groups match the order in which the groups
        would be seen when iterating over the groupby object, not the
        order they are first observed.
        
        Groups with missing keys (where `pd.isna()` is True) will be labeled with `NaN`
        and will be skipped from the count.
        
        Parameters
        ----------
        ascending : bool, default True
            If False, number in reverse, from number of group - 1 to 0.
        
        Returns
        -------
        Series
            Unique numbers for each group.
        
        See Also
        --------
        .cumcount : Number the rows in each group.
        
        Examples
        --------
        >>> df = pd.DataFrame({"color": ["red", None, "red", "blue", "blue", "red"]})
        >>> df
           color
        0    red
        1   None
        2    red
        3   blue
        4   blue
        5    red
        >>> df.groupby("color").ngroup()
        0    1.0
        1    NaN
        2    1.0
        3    0.0
        4    0.0
        5    1.0
        dtype: float64
        >>> df.groupby("color", dropna=False).ngroup()
        0    1
        1    2
        2    1
        3    0
        4    0
        5    1
        dtype: int64
        >>> df.groupby("color", dropna=False).ngroup(ascending=False)
        0    1
        1    0
        2    1
        3    2
        4    2
        5    1
        dtype: int64
        """
        ...
    def nunique(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.
        
        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.
        
        Returns
        -------
        nunique: DataFrame
        
        Examples
        --------
        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')})
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        
        >>> df.groupby('id').nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1
        
        Check for rows with the same id but conflicting values:
        
        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """
        ...
    def ohlc(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute open, high, low and close values of a group, excluding missing values.
        
        For multiple groupings, the result index will be a MultiIndex
        
        Returns
        -------
        DataFrame
            Open, high, low and close values within each group.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC', 'SPX', 'CAC',]
        >>> ser = pd.Series([3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 0.1, 0.5], index=lst)
        >>> ser
        SPX     3.4
        CAC     9.0
        SPX     7.2
        CAC     5.2
        SPX     8.8
        CAC     9.4
        SPX     0.1
        CAC     0.5
        dtype: float64
        >>> ser.groupby(level=0).ohlc()
             open  high  low  close
        CAC   9.0   9.4  0.5    0.5
        SPX   3.4   8.8  0.1    0.1
        
        For DataFrameGroupBy:
        
        >>> data = {2022: [1.2, 2.3, 8.9, 4.5, 4.4, 3, 2 , 1],
        ...         2023: [3.4, 9.0, 7.2, 5.2, 8.8, 9.4, 8.2, 1.0]}
        >>> df = pd.DataFrame(data, index=['SPX', 'CAC', 'SPX', 'CAC',
        ...                   'SPX', 'CAC', 'SPX', 'CAC'])
        >>> df
             2022  2023
        SPX   1.2   3.4
        CAC   2.3   9.0
        SPX   8.9   7.2
        CAC   4.5   5.2
        SPX   4.4   8.8
        CAC   3.0   9.4
        SPX   2.0   8.2
        CAC   1.0   1.0
        >>> df.groupby(level=0).ohlc()
            2022                 2023
            open high  low close open high  low close
        CAC  2.3  4.5  1.0   1.0  9.0  9.4  1.0   1.0
        SPX  1.2  8.9  1.2   2.0  3.4  8.8  3.4   8.2
        
        For Resampler:
        
        >>> ser = pd.Series([1, 3, 2, 4, 3, 5],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').ohlc()
                    open  high  low  close
        2023-01-01     1     3    1      2
        2023-02-01     4     5    3      5
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
    def pct_change(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Calculate pct_change of each value to previous entry in group.
        
        Returns
        -------
        Series or DataFrame
            Percentage changes within each group.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).pct_change()
        a         NaN
        a    1.000000
        b         NaN
        b    0.333333
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").pct_change()
                    b  c
            tuna    NaN    NaN
          salmon    1.5  1.000
         catfish    NaN    NaN
        goldfish    0.2  0.125
        """
        ...
    def pipe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Apply a ``func`` with arguments to this GroupBy object and return its result.
        
        Use `.pipe` when you want to improve readability by chaining together
        functions that expect Series, DataFrames, GroupBy or Resampler objects.
        Instead of writing
        
        >>> h = lambda x, arg2, arg3: x + 1 - arg2 * arg3
        >>> g = lambda x, arg1: x * 5 / arg1
        >>> f = lambda x: x ** 4
        >>> df = pd.DataFrame([["a", 4], ["b", 5]], columns=["group", "value"])
        >>> h(g(f(df.groupby('group')), arg1=1), arg2=2, arg3=3)  # doctest: +SKIP
        
        You can write
        
        >>> (df.groupby('group')
        ...    .pipe(f)
        ...    .pipe(g, arg1=1)
        ...    .pipe(h, arg2=2, arg3=3))  # doctest: +SKIP
        
        which is much more readable.
        
        Parameters
        ----------
        func : callable or tuple of (callable, str)
            Function to apply to this GroupBy object or, alternatively,
            a `(callable, data_keyword)` tuple where `data_keyword` is a
            string indicating the keyword of `callable` that expects the
            GroupBy object.
        args : iterable, optional
               Positional arguments passed into `func`.
        kwargs : dict, optional
                 A dictionary of keyword arguments passed into `func`.
        
        Returns
        -------
        the return type of `func`.
        
        See Also
        --------
        Series.pipe : Apply a function with arguments to a series.
        DataFrame.pipe: Apply a function with arguments to a dataframe.
        apply : Apply function to each group instead of to the
            full GroupBy object.
        
        Notes
        -----
        See more `here
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#piping-function-calls>`_
        
        Examples
        --------
        >>> df = pd.DataFrame({'A': 'a b a b'.split(), 'B': [1, 2, 3, 4]})
        >>> df
           A  B
        0  a  1
        1  b  2
        2  a  3
        3  b  4
        
        To get the difference between each groups maximum and minimum value in one
        pass, you can do
        
        >>> df.groupby('A').pipe(lambda x: x.max() - x.min())
           B
        A
        a  2
        b  2
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
    def prod(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute prod of group values.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None``.
        
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        
        Returns
        -------
        Series or DataFrame
            Computed prod of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).prod()
        a    2
        b   12
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").prod()
             b    c
        a
        1   16   10
        2   30   72
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
    def quantile(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return group values at the given quantile, a la numpy.percentile.
        
        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            Value(s) between 0 and 1 providing the quantile(s) to compute.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            Method to use when the desired quantile falls between two points.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
        
                numeric_only now defaults to ``False``.
        
        Returns
        -------
        Series or DataFrame
            Return type determined by caller of GroupBy object.
        
        See Also
        --------
        Series.quantile : Similar method for Series.
        DataFrame.quantile : Similar method for DataFrame.
        numpy.percentile : NumPy method to compute qth percentile.
        
        Examples
        --------
        >>> df = pd.DataFrame([
        ...     ['a', 1], ['a', 2], ['a', 3],
        ...     ['b', 1], ['b', 3], ['b', 5]
        ... ], columns=['key', 'val'])
        >>> df.groupby('key').quantile()
            val
        key
        a    2.0
        b    3.0
        """
        ...
    def rank(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Provide the rank of values within each group.
        
        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group.
            * min: lowest rank in group.
            * max: highest rank in group.
            * first: ranks assigned in order they appear in the array.
            * dense: like 'min', but rank always increases by 1 between groups.
        ascending : bool, default True
            False for ranks by high (1) to low (N).
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            * keep: leave NA values where they are.
            * top: smallest rank if ascending.
            * bottom: smallest rank if descending.
        pct : bool, default False
            Compute percentage rank of data within each group.
        axis : int, default 0
            The axis of the object over which to compute the rank.
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        Returns
        -------
        DataFrame with ranking of values within each group
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "group": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"],
        ...         "value": [2, 4, 2, 3, 5, 1, 2, 4, 1, 5],
        ...     }
        ... )
        >>> df
          group  value
        0     a      2
        1     a      4
        2     a      2
        3     a      3
        4     a      5
        5     b      1
        6     b      2
        7     b      4
        8     b      1
        9     b      5
        >>> for method in ['average', 'min', 'max', 'dense', 'first']:
        ...     df[f'{method}_rank'] = df.groupby('group')['value'].rank(method)
        >>> df
          group  value  average_rank  min_rank  max_rank  dense_rank  first_rank
        0     a      2           1.5       1.0       2.0         1.0         1.0
        1     a      4           4.0       4.0       4.0         3.0         4.0
        2     a      2           1.5       1.0       2.0         1.0         2.0
        3     a      3           3.0       3.0       3.0         2.0         3.0
        4     a      5           5.0       5.0       5.0         4.0         5.0
        5     b      1           1.5       1.0       2.0         1.0         1.0
        6     b      2           3.0       3.0       3.0         2.0         3.0
        7     b      4           4.0       4.0       4.0         3.0         4.0
        8     b      1           1.5       1.0       2.0         1.0         2.0
        9     b      5           5.0       5.0       5.0         4.0         5.0
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
    def resample(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Provide resampling when using a TimeGrouper.
        
        Given a grouper, the function resamples it according to a string
        "string" -> "frequency".
        
        See the :ref:`frequency aliases <timeseries.offset_aliases>`
        documentation for more details.
        
        Parameters
        ----------
        rule : str or DateOffset
            The offset string or object representing target grouper conversion.
        *args
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.
        include_groups : bool, default True
            When True, will attempt to include the groupings in the operation in
            the case that they are columns of the DataFrame. If this raises a
            TypeError, the result will be computed with the groupings excluded.
            When False, the groupings will be excluded when applying ``func``.
        
            .. versionadded:: 2.2.0
        
            .. deprecated:: 2.2.0
        
               Setting include_groups to True is deprecated. Only the value
               False will be allowed in a future version of pandas.
        
        **kwargs
            Possible arguments are `how`, `fill_method`, `limit`, `kind` and
            `on`, and other arguments of `TimeGrouper`.
        
        Returns
        -------
        pandas.api.typing.DatetimeIndexResamplerGroupby,
        pandas.api.typing.PeriodIndexResamplerGroupby, or
        pandas.api.typing.TimedeltaIndexResamplerGroupby
            Return a new groupby object, with type depending on the data
            being resampled.
        
        See Also
        --------
        Grouper : Specify a frequency to resample with when
            grouping by a key.
        DatetimeIndex.resample : Frequency conversion and resampling of
            time series.
        
        Examples
        --------
        >>> idx = pd.date_range('1/1/2000', periods=4, freq='min')
        >>> df = pd.DataFrame(data=4 * [range(2)],
        ...                   index=idx,
        ...                   columns=['a', 'b'])
        >>> df.iloc[2, 0] = 5
        >>> df
                            a  b
        2000-01-01 00:00:00  0  1
        2000-01-01 00:01:00  0  1
        2000-01-01 00:02:00  5  1
        2000-01-01 00:03:00  0  1
        
        Downsample the DataFrame into 3 minute bins and sum the values of
        the timestamps falling into a bin.
        
        >>> df.groupby('a').resample('3min', include_groups=False).sum()
                                 b
        a
        0   2000-01-01 00:00:00  2
            2000-01-01 00:03:00  1
        5   2000-01-01 00:00:00  1
        
        Upsample the series into 30 second bins.
        
        >>> df.groupby('a').resample('30s', include_groups=False).sum()
                            b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:00:30  0
            2000-01-01 00:01:00  1
            2000-01-01 00:01:30  0
            2000-01-01 00:02:00  0
            2000-01-01 00:02:30  0
            2000-01-01 00:03:00  1
        5   2000-01-01 00:02:00  1
        
        Resample by month. Values are assigned to the month of the period.
        
        >>> df.groupby('a').resample('ME', include_groups=False).sum()
                    b
        a
        0   2000-01-31  3
        5   2000-01-31  1
        
        Downsample the series into 3 minute bins as above, but close the right
        side of the bin interval.
        
        >>> (
        ...     df.groupby('a')
        ...     .resample('3min', closed='right', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   1999-12-31 23:57:00  1
            2000-01-01 00:00:00  2
        5   2000-01-01 00:00:00  1
        
        Downsample the series into 3 minute bins and close the right side of
        the bin interval, but label each bin using the right edge instead of
        the left.
        
        >>> (
        ...     df.groupby('a')
        ...     .resample('3min', closed='right', label='right', include_groups=False)
        ...     .sum()
        ... )
                                 b
        a
        0   2000-01-01 00:00:00  1
            2000-01-01 00:03:00  2
        5   2000-01-01 00:03:00  1
        """
        ...
    def rolling(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return a rolling grouper, providing rolling functionality per group.
        
        Parameters
        ----------
        window : int, timedelta, str, offset, or BaseIndexer subclass
            Size of the moving window.
        
            If an integer, the fixed number of observations used for
            each window.
        
            If a timedelta, str, or offset, the time period of each window. Each
            window will be a variable sized based on the observations included in
            the time-period. This is only valid for datetimelike indexes.
            To learn more about the offsets & frequency strings, please see `this link
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`__.
        
            If a BaseIndexer subclass, the window boundaries
            based on the defined ``get_window_bounds`` method. Additional rolling
            keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
            ``step`` will be passed to ``get_window_bounds``.
        
        min_periods : int, default None
            Minimum number of observations in window required to have a value;
            otherwise, result is ``np.nan``.
        
            For a window that is specified by an offset,
            ``min_periods`` will default to 1.
        
            For a window that is specified by an integer, ``min_periods`` will default
            to the size of the window.
        
        center : bool, default False
            If False, set the window labels as the right edge of the window index.
        
            If True, set the window labels as the center of the window index.
        
        win_type : str, default None
            If ``None``, all points are evenly weighted.
        
            If a string, it must be a valid `scipy.signal window function
            <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.
        
            Certain Scipy window types require additional parameters to be passed
            in the aggregation function. The additional parameters must match
            the keywords specified in the Scipy window type method signature.
        
        on : str, optional
            For a DataFrame, a column label or Index level on which
            to calculate the rolling window, rather than the DataFrame's index.
        
            Provided integer column is ignored and excluded from result since
            an integer index is not used to calculate the rolling window.
        
        axis : int or str, default 0
            If ``0`` or ``'index'``, roll across the rows.
        
            If ``1`` or ``'columns'``, roll across the columns.
        
            For `Series` this parameter is unused and defaults to 0.
        
        closed : str, default None
            If ``'right'``, the first point in the window is excluded from calculations.
        
            If ``'left'``, the last point in the window is excluded from calculations.
        
            If ``'both'``, no points in the window are excluded from calculations.
        
            If ``'neither'``, the first and last points in the window are excluded
            from calculations.
        
            Default ``None`` (``'right'``).
        
        method : str {'single', 'table'}, default 'single'
            Execute the rolling operation per single column or row (``'single'``)
            or over the entire object (``'table'``).
        
            This argument is only implemented when specifying ``engine='numba'``
            in the method call.
        
        Returns
        -------
        pandas.api.typing.RollingGroupby
            Return a new grouper with our rolling appended.
        
        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby.
        
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]})
        >>> df
              A  B      C
        0     1  1  0.362
        1     1  2  0.227
        2     2  3  1.267
        3     2  4 -0.562
        
        >>> df.groupby('A').rolling(2).sum()
            B      C
        A
        1 0  NaN    NaN
          1  3.0  0.589
        2 2  NaN    NaN
          3  7.0  0.705
        
        >>> df.groupby('A').rolling(2, min_periods=1).sum()
            B      C
        A
        1 0  1.0  0.362
          1  3.0  0.589
        2 2  3.0  1.267
          3  7.0  0.705
        
        >>> df.groupby('A').rolling(2, on='B').sum()
            B      C
        A
        1 0  1    NaN
          1  2  0.589
        2 2  3    NaN
          3  4  0.705
        """
        ...
    def sample(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return a random sample of items from each group.
        
        You can use `random_state` for reproducibility.
        
        Parameters
        ----------
        n : int, optional
            Number of items to return for each group. Cannot be used with
            `frac` and must be no larger than the smallest group unless
            `replace` is True. Default is one if `frac` is None.
        frac : float, optional
            Fraction of items to return. Cannot be used with `n`.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
        weights : list-like, optional
            Default None results in equal probability weighting.
            If passed a list-like then values must have the same length as
            the underlying DataFrame or Series object and will be used as
            sampling probabilities after normalization within each group.
            Values must be non-negative with at least one positive element
            within each group.
        random_state : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
            If int, array-like, or BitGenerator, seed for random number generator.
            If np.random.RandomState or np.random.Generator, use as given.
        
            .. versionchanged:: 1.4.0
        
                np.random.Generator objects now accepted
        
        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing items randomly
            sampled within each group from the caller object.
        
        See Also
        --------
        DataFrame.sample: Generate random samples from a DataFrame object.
        numpy.random.choice: Generate a random sample from a given 1-D numpy
            array.
        
        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {"a": ["red"] * 2 + ["blue"] * 2 + ["black"] * 2, "b": range(6)}
        ... )
        >>> df
               a  b
        0    red  0
        1    red  1
        2   blue  2
        3   blue  3
        4  black  4
        5  black  5
        
        Select one row at random for each distinct value in column a. The
        `random_state` argument can be used to guarantee reproducibility:
        
        >>> df.groupby("a").sample(n=1, random_state=1)
               a  b
        4  black  4
        2   blue  2
        1    red  1
        
        Set `frac` to sample fixed proportions rather than counts:
        
        >>> df.groupby("a")["b"].sample(frac=0.5, random_state=2)
        5    5
        2    2
        0    0
        Name: b, dtype: int64
        
        Control sample probabilities within groups by setting weights:
        
        >>> df.groupby("a").sample(
        ...     n=1,
        ...     weights=[1, 1, 1, 0, 0, 1],
        ...     random_state=1,
        ... )
               a  b
        5  black  5
        2   blue  2
        0    red  0
        """
        ...
    def sem(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute standard error of the mean of groups, excluding missing values.
        
        For multiple groupings, the result index will be a MultiIndex.
        
        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
        
                numeric_only now defaults to ``False``.
        
        Returns
        -------
        Series or DataFrame
            Standard error of the mean of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([5, 10, 8, 14], index=lst)
        >>> ser
        a     5
        a    10
        b     8
        b    14
        dtype: int64
        >>> ser.groupby(level=0).sem()
        a    2.5
        b    3.0
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 12, 11], [1, 15, 2], [2, 5, 8], [2, 6, 12]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a   b   c
            tuna   1  12  11
          salmon   1  15   2
         catfish   2   5   8
        goldfish   2   6  12
        >>> df.groupby("a").sem()
              b  c
        a
        1    1.5  4.5
        2    0.5  2.0
        
        For Resampler:
        
        >>> ser = pd.Series([1, 3, 2, 4, 3, 8],
        ...                 index=pd.DatetimeIndex(['2023-01-01',
        ...                                         '2023-01-10',
        ...                                         '2023-01-15',
        ...                                         '2023-02-01',
        ...                                         '2023-02-10',
        ...                                         '2023-02-15']))
        >>> ser.resample('MS').sem()
        2023-01-01    0.577350
        2023-02-01    1.527525
        Freq: MS, dtype: float64
        """
        ...
    def shift(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Shift each group by periods observations.
        
        If freq is passed, the index will be increased using the periods and the freq.
        
        Parameters
        ----------
        periods : int | Sequence[int], default 1
            Number of periods to shift. If a list of values, shift each group by
            each period.
        freq : str, optional
            Frequency string.
        axis : axis to shift, default 0
            Shift direction.
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        fill_value : optional
            The scalar value to use for newly introduced missing values.
        
            .. versionchanged:: 2.1.0
                Will raise a ``ValueError`` if ``freq`` is provided too.
        
        suffix : str, optional
            A string to add to each shifted column if there are multiple periods.
            Ignored otherwise.
        
        Returns
        -------
        Series or DataFrame
            Object shifted within each group.
        
        See Also
        --------
        Index.shift : Shift values of Index.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).shift(1)
        a    NaN
        a    1.0
        b    NaN
        b    3.0
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 2, 3], [1, 5, 6], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tuna", "salmon", "catfish", "goldfish"])
        >>> df
                   a  b  c
            tuna   1  2  3
          salmon   1  5  6
         catfish   2  5  8
        goldfish   2  6  9
        >>> df.groupby("a").shift(1)
                      b    c
            tuna    NaN  NaN
          salmon    2.0  3.0
         catfish    NaN  NaN
        goldfish    5.0  8.0
        """
        ...
    def size(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute group sizes.
        
        Returns
        -------
        DataFrame or Series
            Number of rows in each group as a Series if as_index is True
            or a DataFrame if as_index is False.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b']
        >>> ser = pd.Series([1, 2, 3], index=lst)
        >>> ser
        a     1
        a     2
        b     3
        dtype: int64
        >>> ser.groupby(level=0).size()
        a    2
        b    1
        dtype: int64
        
        >>> data = [[1, 2, 3], [1, 5, 6], [7, 8, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["owl", "toucan", "eagle"])
        >>> df
                a  b  c
        owl     1  2  3
        toucan  1  5  6
        eagle   7  8  9
        >>> df.groupby("a").size()
        a
        1    2
        7    1
        dtype: int64
        
        For Resampler:
        
        >>> ser = pd.Series([1, 2, 3], index=pd.DatetimeIndex(
        ...                 ['2023-01-01', '2023-01-15', '2023-02-01']))
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        dtype: int64
        >>> ser.resample('MS').size()
        2023-01-01    2
        2023-02-01    1
        Freq: MS, dtype: int64
        """
        ...
    def skew(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return unbiased skew within groups.
        
        Normalized by N-1.
        
        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            Axis for the function to be applied on.
        
            Specifying ``axis=None`` will apply the aggregation across both axes.
        
            .. versionadded:: 2.0.0
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
        **kwargs
            Additional keyword arguments to be passed to the function.
        
        Returns
        -------
        DataFrame
        
        See Also
        --------
        DataFrame.skew : Return unbiased skew over requested axis.
        
        Examples
        --------
        >>> arrays = [['falcon', 'parrot', 'cockatoo', 'kiwi',
        ...            'lion', 'monkey', 'rabbit'],
        ...           ['bird', 'bird', 'bird', 'bird',
        ...            'mammal', 'mammal', 'mammal']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=('name', 'class'))
        >>> df = pd.DataFrame({'max_speed': [389.0, 24.0, 70.0, np.nan,
        ...                                  80.5, 21.5, 15.0]},
        ...                   index=index)
        >>> df
                        max_speed
        name     class
        falcon   bird        389.0
        parrot   bird         24.0
        cockatoo bird         70.0
        kiwi     bird          NaN
        lion     mammal       80.5
        monkey   mammal       21.5
        rabbit   mammal       15.0
        >>> gb = df.groupby(["class"])
        >>> gb.skew()
                max_speed
        class
        bird     1.628296
        mammal   1.669046
        >>> gb.skew(skipna=False)
                max_speed
        class
        bird          NaN
        mammal   1.669046
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
    def std(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute standard deviation of groups, excluding missing values.
        
        For multiple groupings, the result index will be a MultiIndex.
        
        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        
        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``
        
            .. versionadded:: 1.4.0
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``
        
            .. versionadded:: 1.4.0
        
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
        
                numeric_only now defaults to ``False``.
        
        Returns
        -------
        Series or DataFrame
            Standard deviation of values within each group.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).std()
        a    3.21455
        b    0.57735
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).std()
                      a         b
        dog    2.000000  3.511885
        mouse  2.217356  1.500000
        """
        ...
    def sum(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute sum of group values.
        
        Parameters
        ----------
        numeric_only : bool, default False
            Include only float, int, boolean columns.
        
            .. versionchanged:: 2.0.0
        
                numeric_only no longer accepts ``None``.
        
        min_count : int, default 0
            The required number of valid values to perform the operation. If fewer
            than ``min_count`` non-NA values are present the result will be NA.
        
        engine : str, default None None
            * ``'cython'`` : Runs rolling apply through C-extensions from cython.
            * ``'numba'`` : Runs rolling apply through JIT compiled code from numba.
                Only available when ``raw`` is set to ``True``.
            * ``None`` : Defaults to ``'cython'`` or globally setting ``compute.use_numba``
        
        engine_kwargs : dict, default None None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
                and ``parallel`` dictionary keys. The values must either be ``True`` or
                ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
                ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
                applied to both the ``func`` and the ``apply`` groupby aggregation.
        
        Returns
        -------
        Series or DataFrame
            Computed sum of values within each group.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'b', 'b']
        >>> ser = pd.Series([1, 2, 3, 4], index=lst)
        >>> ser
        a    1
        a    2
        b    3
        b    4
        dtype: int64
        >>> ser.groupby(level=0).sum()
        a    3
        b    7
        dtype: int64
        
        For DataFrameGroupBy:
        
        >>> data = [[1, 8, 2], [1, 2, 5], [2, 5, 8], [2, 6, 9]]
        >>> df = pd.DataFrame(data, columns=["a", "b", "c"],
        ...                   index=["tiger", "leopard", "cheetah", "lion"])
        >>> df
                  a  b  c
          tiger   1  8  2
        leopard   1  2  5
        cheetah   2  5  8
           lion   2  6  9
        >>> df.groupby("a").sum()
             b   c
        a
        1   10   7
        2   11  17
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
    def tail(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return last n rows of each group.
        
        Similar to ``.apply(lambda x: x.tail(n))``, but it returns a subset of rows
        from the original DataFrame with original index and order preserved
        (``as_index`` flag is ignored).
        
        Parameters
        ----------
        n : int
            If positive: number of entries to include from end of each group.
            If negative: number of entries to exclude from start of each group.
        
        Returns
        -------
        Series or DataFrame
            Subset of original Series or DataFrame as determined by n.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        
        >>> df = pd.DataFrame([['a', 1], ['a', 2], ['b', 1], ['b', 2]],
        ...                   columns=['A', 'B'])
        >>> df.groupby('A').tail(1)
           A  B
        1  a  2
        3  b  2
        >>> df.groupby('A').tail(-1)
           A  B
        1  a  2
        3  b  2
        """
        ...
    def take(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return the elements in the given *positional* indices in each group.
        
        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.
        
        If a requested index does not exist for some group, this method will raise.
        To get similar behavior that ignores indices that don't exist, see
        :meth:`.DataFrameGroupBy.nth`.
        
        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
        
            .. deprecated:: 2.1.0
                For axis=1, operate on the underlying object instead. Otherwise
                the axis keyword is not necessary.
        
        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.
        
        Returns
        -------
        DataFrame
            An DataFrame containing the elements taken from each group.
        
        See Also
        --------
        DataFrame.take : Take elements from a Series along an axis.
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.
        
        Examples
        --------
        >>> df = pd.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan),
        ...                    ('rabbit', 'mammal', 15.0)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[4, 3, 2, 1, 0])
        >>> df
             name   class  max_speed
        4  falcon    bird      389.0
        3  parrot    bird       24.0
        2    lion  mammal       80.5
        1  monkey  mammal        NaN
        0  rabbit  mammal       15.0
        >>> gb = df.groupby([1, 1, 2, 2, 2])
        
        Take elements at positions 0 and 1 along the axis 0 (default).
        
        Note how the indices selected in the result do not correspond to
        our input indices 0 and 1. That's because we are selecting the 0th
        and 1st rows, not rows whose indices equal 0 and 1.
        
        >>> gb.take([0, 1])
               name   class  max_speed
        1 4  falcon    bird      389.0
          3  parrot    bird       24.0
        2 2    lion  mammal       80.5
          1  monkey  mammal        NaN
        
        The order of the specified indices influences the order in the result.
        Here, the order is swapped from the previous example.
        
        >>> gb.take([1, 0])
               name   class  max_speed
        1 3  parrot    bird       24.0
          4  falcon    bird      389.0
        2 1  monkey  mammal        NaN
          2    lion  mammal       80.5
        
        Take elements at indices 1 and 2 along the axis 1 (column selection).
        
        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.
        
        >>> gb.take([-1, -2])
               name   class  max_speed
        1 3  parrot    bird       24.0
          4  falcon    bird      389.0
        2 0  rabbit  mammal       15.0
          1  monkey  mammal        NaN
        """
        ...
    def transform(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Call function producing a same-indexed DataFrame on each group.
        
        Returns a DataFrame having the same indexes as the original object
        filled with the transformed values.
        
        Parameters
        ----------
        f : function, str
            Function to apply to each group. See the Notes section below for requirements.
        
            Accepted inputs are:
        
            - String
            - Python function
            - Numba JIT function with ``engine='numba'`` specified.
        
            Only passing a single function is supported with this engine.
            If the ``'numba'`` engine is chosen, the function must be
            a user defined function with ``values`` and ``index`` as the
            first and second arguments respectively in the function signature.
            Each group's index will be passed to the user defined function
            and optionally available for use.
        
            If a string is chosen, then it needs to be the name
            of the groupby method you want to use.
        *args
            Positional arguments to pass to func.
        engine : str, default None
            * ``'cython'`` : Runs the function through C-extensions from cython.
            * ``'numba'`` : Runs the function through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or the global setting ``compute.use_numba``
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function
        
        **kwargs
            Keyword arguments to be passed into func.
        
        Returns
        -------
        DataFrame
        
        See Also
        --------
        DataFrame.groupby.apply : Apply function ``func`` group-wise and combine
            the results together.
        DataFrame.groupby.aggregate : Aggregate using one or more
            operations over the specified axis.
        DataFrame.transform : Call ``func`` on self producing a DataFrame with the
            same axis shape as self.
        
        Notes
        -----
        Each group is endowed the attribute 'name' in case you need to know
        which group you are working on.
        
        The current implementation imposes three requirements on f:
        
        * f must return a value that either has the same shape as the input
          subframe or can be broadcast to the shape of the input subframe.
          For example, if `f` returns a scalar it will be broadcast to have the
          same shape as the input subframe.
        * if this is a DataFrame, f must support application column-by-column
          in the subframe. If f also supports application to the entire subframe,
          then a fast path is used starting from the second chunk.
        * f must not mutate groups. Mutation is not supported and may
          produce unexpected results. See :ref:`gotchas.udf-mutation` for more details.
        
        When using ``engine='numba'``, there will be no "fall back" behavior internally.
        The group data and group index will be passed as numpy arrays to the JITed
        user defined function, and no alternative execution attempts will be tried.
        
        .. versionchanged:: 1.3.0
        
            The resulting dtype will reflect the return value of the passed ``func``,
            see the examples below.
        
        .. versionchanged:: 2.0.0
        
            When using ``.transform`` on a grouped DataFrame and the transformation function
            returns a DataFrame, pandas now aligns the result's index
            with the input's index. You can call ``.to_numpy()`` on the
            result of the transformation function to avoid alignment.
        
        Examples
        --------
        
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : ['one', 'one', 'two', 'three',
        ...                           'two', 'two'],
        ...                    'C' : [1, 5, 5, 2, 5, 5],
        ...                    'D' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')[['C', 'D']]
        >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
                C         D
        0 -1.154701 -0.577350
        1  0.577350  0.000000
        2  0.577350  1.154701
        3 -1.154701 -1.000000
        4  0.577350 -0.577350
        5  0.577350  1.000000
        
        Broadcast result of the transformation
        
        >>> grouped.transform(lambda x: x.max() - x.min())
            C    D
        0  4.0  6.0
        1  3.0  8.0
        2  4.0  6.0
        3  3.0  8.0
        4  4.0  6.0
        5  3.0  8.0
        
        >>> grouped.transform("mean")
            C    D
        0  3.666667  4.0
        1  4.000000  5.0
        2  3.666667  4.0
        3  4.000000  5.0
        4  3.666667  4.0
        5  4.000000  5.0
        
        .. versionchanged:: 1.3.0
        
        The resulting dtype will reflect the return value of the passed ``func``,
        for example:
        
        >>> grouped.transform(lambda x: x.astype(int).max())
        C  D
        0  5  8
        1  5  9
        2  5  8
        3  5  9
        4  5  8
        5  5  9
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
    def value_counts(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Return a Series or DataFrame containing counts of unique rows.
        
        .. versionadded:: 1.4.0
        
        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don't include counts of rows that contain NA values.
        
        Returns
        -------
        Series or DataFrame
            Series if the groupby as_index is True, otherwise DataFrame.
        
        See Also
        --------
        Series.value_counts: Equivalent method on Series.
        DataFrame.value_counts: Equivalent method on DataFrame.
        SeriesGroupBy.value_counts: Equivalent method on SeriesGroupBy.
        
        Notes
        -----
        - If the groupby as_index is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby as_index is False then the returned DataFrame will have an
          additional column with the value_counts. The column is labelled 'count' or
          'proportion', depending on the ``normalize`` parameter.
        
        By default, rows that contain any NA values are omitted from
        the result.
        
        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.
        
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        ...     'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        ...     'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
        ... })
        
        >>> df
                gender  education   country
        0       male    low         US
        1       male    medium      FR
        2       female  high        US
        3       male    low         FR
        4       female  high        FR
        5       male    low         FR
        
        >>> df.groupby('gender').value_counts()
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        FR         2
                           US         1
                medium     FR         1
        Name: count, dtype: int64
        
        >>> df.groupby('gender').value_counts(ascending=True)
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        US         1
                medium     FR         1
                low        FR         2
        Name: count, dtype: int64
        
        >>> df.groupby('gender').value_counts(normalize=True)
        gender  education  country
        female  high       FR         0.50
                           US         0.50
        male    low        FR         0.50
                           US         0.25
                medium     FR         0.25
        Name: proportion, dtype: float64
        
        >>> df.groupby('gender', as_index=False).value_counts()
           gender education country  count
        0  female      high      FR      1
        1  female      high      US      1
        2    male       low      FR      2
        3    male       low      US      1
        4    male    medium      FR      1
        
        >>> df.groupby('gender', as_index=False).value_counts(normalize=True)
           gender education country  proportion
        0  female      high      FR        0.50
        1  female      high      US        0.50
        2    male       low      FR        0.50
        3    male       low      US        0.25
        4    male    medium      FR        0.25
        """
        ...
    def var(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute variance of groups, excluding missing values.
        
        For multiple groupings, the result index will be a MultiIndex.
        
        Parameters
        ----------
        ddof : int, default 1
            Degrees of freedom.
        
        engine : str, default None
            * ``'cython'`` : Runs the operation through C-extensions from cython.
            * ``'numba'`` : Runs the operation through JIT compiled code from numba.
            * ``None`` : Defaults to ``'cython'`` or globally setting
              ``compute.use_numba``
        
            .. versionadded:: 1.4.0
        
        engine_kwargs : dict, default None
            * For ``'cython'`` engine, there are no accepted ``engine_kwargs``
            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{{'nopython': True, 'nogil': False, 'parallel': False}}``
        
            .. versionadded:: 1.4.0
        
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.
        
            .. versionadded:: 1.5.0
        
            .. versionchanged:: 2.0.0
        
                numeric_only now defaults to ``False``.
        
        Returns
        -------
        Series or DataFrame
            Variance of values within each group.
        
        See Also
        --------
        Series.groupby : Apply a function groupby to a Series.
        DataFrame.groupby : Apply a function groupby
            to each row or column of a DataFrame.
        
        Examples
        --------
        For SeriesGroupBy:
        
        >>> lst = ['a', 'a', 'a', 'b', 'b', 'b']
        >>> ser = pd.Series([7, 2, 8, 4, 3, 3], index=lst)
        >>> ser
        a     7
        a     2
        a     8
        b     4
        b     3
        b     3
        dtype: int64
        >>> ser.groupby(level=0).var()
        a    10.333333
        b     0.333333
        dtype: float64
        
        For DataFrameGroupBy:
        
        >>> data = {'a': [1, 3, 5, 7, 7, 8, 3], 'b': [1, 4, 8, 4, 4, 2, 1]}
        >>> df = pd.DataFrame(data, index=['dog', 'dog', 'dog',
        ...                   'mouse', 'mouse', 'mouse', 'mouse'])
        >>> df
                 a  b
          dog    1  1
          dog    3  4
          dog    5  8
        mouse    7  4
        mouse    7  4
        mouse    8  2
        mouse    3  1
        >>> df.groupby(level=0).var()
                      a          b
        dog    4.000000  12.333333
        mouse  4.916667   2.250000
        """
        ...

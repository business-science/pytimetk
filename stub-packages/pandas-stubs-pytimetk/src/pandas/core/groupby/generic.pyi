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

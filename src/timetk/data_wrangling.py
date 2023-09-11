


import pandas as pd
import numpy as np

from pandas.core import frame
import pandas_flavor as pf

@pf.register_dataframe_method
def aggregate_by_time(
    data, 
    date_column, 
    value_column, 
    groups = None,
    by="D", 
    agg_func=np.sum, 
    kind="timestamp", 
    wide_format=False, 
    **kwargs
    ):

    """Applies one or more aggregating functions by a Pandas Period to one or more numeric columns.
    Works with Pandas DataFrame objects.

    Args:
        data (DataFrame): 
            A Pandas data frame

        date_column (str, list): 
            A single quoted column name representing the timestamp

        value_column (str, list): 
            One or more quoted column names representing the numeric data to be aggregated

        groups (str, list, optional):
            One or more column names representing groups to aggregate by

        by (str, optional): 
            A pandas frequency (offset) such as "D" for daily or "MS" for Month Start. 
            Defaults to "D".

        agg_func (function, list): 
            One or more aggregating functions such as numpy.sum. 
            Defaults to np.sum.

        kind (str, optional): 
            Either 'timestamp' or 'period'. 
            Is passed to pandas.resample(). 
            Defaults to "timestamp".

        wide_format (bool, optional): 
            Returns either 'wide' or 'long' format. Defaults to False.
        
        **kwargs:
            Additional argmuments passed to pandas.resample()

    Raises:
        TypeError: Accepts either Pandas DataFrames or GroupBy DataFrames.
    """

    # Checks
    if not isinstance(
        data, 
        (pd.core.frame.DataFrame)
    ):
        raise TypeError(
            '`data` must be: pandas.core.frame.DataFrame')

    if type(date_column) is not list:
        date_column = [date_column]

    if type(value_column) is not list:
        value_column = [value_column]

    if type(agg_func) is not list:
        agg_func = [agg_func]

    # Handle Date Column
    if isinstance(
        data, (pd.core.frame.DataFrame)
    ):
        data = data.set_index(date_column)


    # Handle Groups
    if groups is not None:
        if type(groups) is not list:
            groups = [groups]
        data = data.groupby(groups)

    # Resample
    funcs = [agg_func] * len(value_column)
    agg_dict = dict(zip(value_column, funcs))
    ret = data \
        .resample(rule=by, kind=kind, **kwargs) \
        .agg(agg_dict)


    # Handle Wide Format
    if (wide_format):
        if groups is not None:
            ret = ret.unstack(groups)
            if (kind == 'period'):
                ret.index = pd.DatetimeIndex(
                    ret.index, freq="infer").to_period()

    return(ret)


# COMPLEX

# Works but very slow
# Better method is using transform()
# df.set_index('order_date').groupby(['city', 'model'])['price'].resample('Q').transform(np.median)

# def mutate(data, **kwargs):

#     # Checks
#     if not isinstance(data, (pd.core.frame.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
#         raise TypeError(
#             '`data` must be either: pandas.core.frame.DataFrame, or pandas.core.groupby.generic.DataFrameGroupBy.')

#     # Dataframes 
#     if isinstance(data, pd.core.frame.DataFrame):
#         data_copy = data.copy()
#         data_copy = data_copy.assign(**kwargs)

#     # Grouped Data Frames
#     if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
#         l = []
#         for k, v in data:
#             l.append(v.assign(**kwargs))
        
#         data_copy = pd.concat(l).loc[data.obj.index, :]
    
#     return(data_copy)

# mutate(df, a = lambda x: np.median(x.price))

# mutate(df_grouped, a = lambda x: np.median(x.price), b = lambda x: np.mean(x.price))



# def mutate_by_time(data, by = "D", kind='timestamp', **kwargs):
    
#     # Checks
#     if not isinstance(data, (pd.core.frame.DataFrame, pd.core.groupby.generic.DataFrameGroupBy)):
#         raise TypeError(
#             '`data` must be either: pandas.core.frame.DataFrame, or pandas.core.groupby.generic.DataFrameGroupBy.')

#     # Dataframes 
#     if isinstance(data, pd.core.frame.DataFrame):
#         data_copy = data.copy()
#         r = data_copy.resample(by, kind=kind) 
        
#         l = []
#         for k, v in r:
#             l.append(v.assign(**kwargs))
        
#         data_copy = pd.concat(l)
#         # TODO - Need to resort index to original order


#     # Grouped Data Frames
#     if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
#         r = data.resample(by, kind=kind)
#         l = []
#         for k, v in r:
#             print(k)
#             print(v)
#             l.append(v.assign(**kwargs))
#             # for i, j in v:
#             #     l.append(j.assign(**kwargs))
        
#         data_copy = pd.concat(l)
#         # TODO - Need to resort index to original order
    
#     return(data_copy)

# mutate_by_time(
#     data = df[['order_date', 'model', 'price']].set_index('order_date'),
#     by='M',
#     price_median = lambda x: np.median(x.price)
# )

# mutate_by_time(
#     data = df[['order_date', 'model', 'price']].set_index('order_date').groupby('model'),
#     by='M',
#     price_median = lambda x: np.median(x.price)
# )

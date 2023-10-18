import polars as pl

from pytimetk.utils.string_helpers import parse_freq_str

def pandas_to_polars_frequency(pandas_freq_str, default=(1, "d")):
    
    quantity, unit = parse_freq_str(pandas_freq_str) 
    
    unit = unit.upper()
    
    dict_mapping = {
        "S"   : (1, "s"),
        "min" : (1, "m"),
        "T"   : (1, "m"),
        "H"   : (1, "h"),
        "D"   : (1, "d"),
        "W"   : (1, "w"),
        "M"   : (1, "mo"),
        "MS"  : (1, "mo"),
        "Q"   : (3, "mo"),
        "QS"  : (3, "mo"),
        "Y"   : (1, "y"),
        "YS"  : (1, "y")
    }
    
    polars_tup = dict_mapping.get(unit, default)
    
    polars_freq_str = f"{quantity * polars_tup[0]}{polars_tup[1]}"
    
    return polars_freq_str

def pandas_to_polars_aggregation_mapping(column_name):
    
    return {
        'sum'    : pl.col(column_name).sum().suffix("_sum"),
        'mean'   : pl.col(column_name).mean().suffix("_mean"),
        'median' : pl.col(column_name).median().suffix("_median"),
        'min'    : pl.col(column_name).min().suffix("_min"),
        'max'    : pl.col(column_name).max().suffix("_max"),
        'std'    : pl.col(column_name).std().suffix("_std"),
        'var'    : pl.col(column_name).var().suffix("_var"),
        'first'  : pl.col(column_name).first().suffix("_first"),
        'last'   : pl.col(column_name).last().suffix("_last"),
        'count'  : pl.col(column_name).count().suffix("_count"),
        'nunique': pl.col(column_name).n_unique().suffix("_nunique")
    }
    
    
def pl_quantile(**kwargs):
    """Generates configuration for the rolling quantile function in Polars."""
    # Designate this function as a 'configurable' type - this helps 'augment_expanding' recognize and process it appropriately
    func_type = 'configurable'
    # Specify the Polars rolling function to be called, `rolling_<func_name>`
    func_name = 'quantile'
    # Initial parameters for Polars' rolling quantile function
    # Many will be updated by **kwargs or inferred externally based on the dataframe
    default_kwargs = {
        'quantile': None,
        'interpolation': 'midpoint',
        'window_size': None,
        'weights': None, 
        'min_periods': None,
        'center': False,
        'by': None,
        'closed': 'left'
    }
    return func_type, func_name, default_kwargs, kwargs

def update_dict(d1, d2):
    """
    Update values in dictionary `d1` based on matching keys from dictionary `d2`.
    
    This function will only update the values of existing keys in `d1`.
    New keys present in `d2` but not in `d1` will be ignored. 
    """
    for key in d1.keys():
        if key in d2:
            d1[key] = d2[key]
    return d1

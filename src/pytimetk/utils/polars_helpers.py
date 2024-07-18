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
        'sum'    : pl.col(column_name).sum().alias(f"{column_name}_sum"),
        'mean'   : pl.col(column_name).mean().alias(f"{column_name}_mean"),
        'median' : pl.col(column_name).median().alias(f"{column_name}_median"),
        'min'    : pl.col(column_name).min().alias(f"{column_name}_min"),
        'max'    : pl.col(column_name).max().alias(f"{column_name}_max"),
        'std'    : pl.col(column_name).std().alias(f"{column_name}_std"),
        'var'    : pl.col(column_name).var().alias(f"{column_name}_var"),
        'first'  : pl.col(column_name).first().alias(f"{column_name}_first"),
        'last'   : pl.col(column_name).last().alias(f"{column_name}_last"),
        'count'  : pl.col(column_name).count().alias(f"{column_name}_count"),
        'nunique': pl.col(column_name).n_unique().alias(f"{column_name}_nunique")
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
        # 'by': None,
        # 'closed': 'left'
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

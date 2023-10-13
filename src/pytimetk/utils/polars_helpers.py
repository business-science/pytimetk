import polars as pl

from pytimetk.utils.datetime_helpers import parse_freq_str

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
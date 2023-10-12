import polars as pl


def pandas_to_polars_frequency_mapping():
    
    return {
        "S"   : "1s",
        "min" : "1m",
        "T"   : "1m",
        "H"   : "1h",
        "D"   : "1d",
        "W"   : "1w",
        "M"   : "1mo",
        "MS"  : "1mo",
        "Q"   : "3mo",
        "QS"  : "3mo",
        "Y"   : "1y",
        "YS"  : "1y"
    }

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
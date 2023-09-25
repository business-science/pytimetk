from plotnine import *
import pandas as pd
import pandas_flavor as pf
from mizani.breaks import date_breaks
from mizani.formatters import date_format

@pf.register_dataframe_method
def plot_time_series(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_var,
    value,

    facet_vars = None,
    facet_ncol = 1,
    facet_nrow = 1,
    facet_scales = "free_y",
    facet_dir = "h",
    facet_collapse = False,
    facet_collapse_sep = " ",
    facet_strip_remove = False,

    title = "Time Series Plot",
    x_lab = "",
    y_lab = "",
    color_lab = "Legend",

    line_color = "#2c3e50",
    line_size = 0.5,
    line_type = 'solid',
    line_alpha = 1,
    y_intercept = None,
    y_intercept_color = "#2c3e50",
    x_intercept = None,
    x_intercept_color = "#2c3e50",

    interactive = True

):   
    group_names = None 
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj
    
    # Plot setup
    g = ggplot(
        data = data,
        mapping = aes(
            x = date_var,
            y = value
        )
    )

    # Add line
    g = g \
        + geom_line(
                color    = line_color,
                size     = line_size,
                linetype = line_type,
                alpha    = line_alpha
            )
    
    # Add a Y-Intercept if desired
    if y_intercept is not None:
        g = g \
            + geom_hline(
                yintercept = y_intercept,
                color = y_intercept_color
        )

    # Add a X-Intercept if desired
    if x_intercept is not None:
        g = g \
            + geom_vline(
                xintercept = x_intercept,
                color = x_intercept_color
        )

    # Add theme & labs
    g = g + labs(x = x_lab, y = y_lab, title = title, color = color_lab)

    # Add scale to X
    g = g + scale_x_datetime(breaks=date_breaks('1 year'))

    # Add facets
    if group_names is not None:
       g = g + facet_wrap(
            group_names,
            ncol = 2, scales = "free", shrink = True
        )

    return g

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_time_series = plot_time_series
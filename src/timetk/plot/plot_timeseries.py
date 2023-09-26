from plotnine import *
import pandas as pd
import pandas_flavor as pf
from mizani.breaks import date_breaks
from mizani.formatters import date_format

    
@pf.register_dataframe_method
def plot_timeseries(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column,
    value_column,
    
    color_column = None,

    facet_ncol = 1,
    facet_nrow = None,
    facet_scales = "free_y",
    facet_dir = "h",

    line_color = "#2c3e50",
    line_size = 0.3,
    line_type = 'solid',
    line_alpha = 1,
    
    y_intercept = None,
    y_intercept_color = "#2c3e50",
    x_intercept = None,
    x_intercept_color = "#2c3e50",
    
    smooth = True,
    smooth_color = "#3366FF",
    
    
    title = "Time Series Plot",
    x_lab = "",
    y_lab = "",
    color_lab = "Legend",
    
    x_axis_date_labels = "%b %Y",

    interactive = True

):   
    '''
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk
    
    df = tk.load_dataset('m4_monthly', parse_dates = ['date'])
    
    (
        df
            .groupby('id')
            .plot_timeseries('date', 'value', x_axis_date_labels = "%Y")
    )
    
    (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                facet_ncol = 2,
                x_axis_date_labels = "%Y"
            )
    )
    
    
    ```
    
    '''
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
    
    
    group_names = None 
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj
    
    # Plot setup
    g = ggplot(
        data = data,
        mapping = aes(
            x = date_column,
            y = value_column
        )
    )

    # Add line
    if color_column is None:
        g = g \
            + geom_line(
                    color    = line_color,
                    size     = line_size,
                    linetype = line_type,
                    alpha    = line_alpha
                )
    else:
        g = g \
            + geom_line(
                    aes(
                        color = color_column
                    ),
                    size     = line_size,
                    linetype = line_type,
                    alpha    = line_alpha
                ) \
            + scale_color_manual(
                values=list(palette_light().values())
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
    g = g + scale_x_datetime(date_labels = x_axis_date_labels)

    # Add facets
    if group_names is not None:
       g = g + facet_wrap(
            group_names,
            ncol = facet_ncol,
            nrow = facet_nrow, 
            scales = facet_scales, 
            dir = facet_dir, 
            shrink = True
        )

    g = g + \
        theme_tq() 
    
    return g

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_timeseries = plot_timeseries


def theme_tq(base_size = 11, base_family = ['Arial', 'Helvetica', 'sans-serif'], dpi = 100):
    
    # Tidyquant colors
    blue  = "#2c3e50"
    green = "#18BC9C"
    white = "#FFFFFF"
    grey  = "#cccccc"
    
    return theme(
        
        # Base Inherited Elements
        line = element_line(color = blue, size = 0.5,  lineend="butt"),
        rect = element_rect(fill = white, colour = blue, size = 0.5),
        text = element_text(family = base_family, face = "plain", color = blue, size = base_size, lineheight = 0.9, hjust = 0.5, vjust = 0.5, angle = 0,  margin = dict()),
        
        # Axes
        axis_line=element_blank(),
        axis_text=element_text(size=base_size * 0.6),
        axis_ticks=element_line(color=grey, size=0.5),
        axis_title=element_text(size=base_size*1),
        
        axis_text_y=element_text(ha='center', margin=dict(r=25, l=0)),
        
        # Panel
        panel_background=element_rect(fill = white, color = None),
        panel_border=element_rect(fill = None, color = blue, size = 0.5),
        panel_grid_major=element_line(color = grey, size = 0.33),
        panel_grid_minor=element_line(color = grey, size = 0.33),
        panel_grid_minor_x=element_blank(),
        panel_spacing=0.005,
        
        # Legend
        legend_key=element_rect(color = white),
        legend_position="bottom",
        legend_box=element_rect(fill = None, color = None, size = 0.5, linetype=None),
        legend_text=element_text(size=base_size*0.6, color = blue, margin=dict(t=0, b=0, r=5, l=5)),
        # legend_title=element_text(size=base_size*0.7, color = blue, margin=dict(t=0, b=5, r=5, l=5)),
        legend_title=element_blank(),
        legend_background=element_blank(),
        # legend_box_spacing=0.25,
        
        # Strip
        strip_background=element_rect(fill = blue, color = blue),
        strip_text=element_text(size=base_size*0.8, color = white, margin=dict(t=5, b=5)),
        
        # Plot
        plot_title=element_text(size=base_size*1.2, color = blue, margin=dict(t = 0, r = 0, b = 4, l = 0), face="bold"),
        plot_subtitle=element_text(size=base_size*0.9, color = blue, margin=dict(t = 0, r = 0, b = 3, l = 0)),
        plot_margin=0.025,
        
        dpi=dpi,
        # complete=True
    )
    
    
def palette_light():
    return dict(
        blue         = "#2c3e50", # blue
        red          = "#e31a1c", # red
        green        = "#18BC9C", # green
        yellow       = "#CCBE93", # yellow
        steel_blue   = "#a6cee3", # steel_blue
        navy_blue    = "#1f78b4", # navy_blue
        light_green  = "#b2df8a", # light_green
        pink         = "#fb9a99", # pink
        light_orange = "#fdbf6f", # light_orange
        orange       = "#ff7f00", # orange
        light_purple = "#cab2d6", # light_purple
        purple       = "#6a3d9a"  # purple
    )
    
    
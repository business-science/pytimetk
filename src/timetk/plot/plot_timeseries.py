from plotnine import *
import pandas as pd
import numpy as np
import pandas_flavor as pf

from mizani.breaks import date_breaks
from mizani.formatters import date_format

from statsmodels.nonparametric.smoothers_lowess import lowess

from timetk.plot.theme import theme_tq, palette_light


    
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
    line_size = 0.5,
    line_type = 'solid',
    line_alpha = 1,
    
    y_intercept = None,
    y_intercept_color = "#2c3e50",
    x_intercept = None,
    x_intercept_color = "#2c3e50",
    
    smooth = True,
    smooth_color = "#3366FF",
    smooth_frac = 0.2,
    smooth_size = 0.65,
    smooth_alpha = 1,
    
    title = "Time Series Plot",
    x_lab = "",
    y_lab = "",
    color_lab = "Legend",
    
    x_axis_date_labels = "%b %Y",
    base_size = 11,

    engine = 'plotnine'

):   
    '''
    
    Examples
    --------
    ```{python}
    import timetk as tk
    
    df = tk.load_dataset('m4_monthly', parse_dates = ['date'])
    
    # Plotnine Object
    fig = (
        df
            # .groupby('id')
            .query('id == "M1"')
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                facet_ncol = 2,
                x_axis_date_labels = "%Y"
            )
    )
    fig
    
    # Matplotlib object
    fig = (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                facet_ncol = 2,
                x_axis_date_labels = "%Y",
                engine = 'matplotlib'
            )
    )
    fig
    
    
    
    ```
    
    '''
    
    # Check if data is a Pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        if not isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
            raise TypeError("`data` is not a Pandas DataFrame.")
        
    # Handle DataFrames
    if isinstance(data, pd.DataFrame):
        
        data = data.copy()
        
        # Handle smoother
        if smooth:
            data['__smooth'] = lowess(data[value_column], data[date_column], frac=smooth_frac, return_sorted=False)
        
    
    # Handle GroupBy objects
    group_names = None 
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):

        group_names = data.grouper.names
        data = data.obj.copy()
        
        # Handle smoother
        if smooth:
            
            data['__smooth'] = np.nan
            
            for name, group in data.groupby(group_names):
                
                sorted_group = group.sort_values(by=date_column)
                x = np.arange(len(sorted_group))
                y = sorted_group[value_column].to_numpy()
                
                smoothed = lowess(y, x, frac=smooth_frac)  # Adjust frac as needed
                
                # Updating the original DataFrame with smoothed values
                data.loc[sorted_group.index, '__smooth'] = smoothed[:, 1]
            
               
    # print(data.head())  
        
    
        
    
    if engine in ['plotnine', 'matplotlib']:
        fig = _plot_timeseries_plotnine(
            data = data,
            date_column = date_column,
            value_column = value_column,
            color_column = color_column,
            group_names = group_names,

            facet_ncol = facet_ncol,
            facet_nrow = facet_nrow,
            facet_scales = facet_scales,
            facet_dir = facet_dir,

            line_color = line_color,
            line_size = line_size,
            line_type = line_type,
            line_alpha = line_alpha,

            y_intercept = y_intercept,
            y_intercept_color = y_intercept_color,
            x_intercept = x_intercept,
            x_intercept_color = x_intercept_color,

            smooth = smooth,
            smooth_color = smooth_color,
            smooth_size = smooth_size,
            smooth_alpha = smooth_alpha,

            title = title,
            x_lab = x_lab,
            y_lab = y_lab,
            color_lab = color_lab,

            x_axis_date_labels = x_axis_date_labels,
            base_size = base_size,
        )
        
        if engine == 'matplotlib':
            fig = fig.draw()
        
    elif engine == 'plotly':
        1+1

    
    return fig

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_timeseries = plot_timeseries



def _plot_timeseries_plotnine(
    data: pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy,
    date_column,
    value_column,
    
    color_column = None,
    group_names = None,

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
    
    smooth = None,
    smooth_color = "#3366FF",
    smooth_span = 0.75,
    smooth_size = 0.3,
    smooth_alpha = 1,
    
    title = "Time Series Plot",
    x_lab = "",
    y_lab = "",
    color_lab = "Legend",
    
    x_axis_date_labels = "%b %Y",
    base_size = 11,
):
    
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
       
    # Add smoother
    if smooth:
        if color_column is None:
            g = g + geom_line(
                aes(
                    y = '__smooth'
                ),
                color = smooth_color,
                size = smooth_size,
                alpha = smooth_alpha
            )
        else:
            g = g + geom_line(
                aes(
                    y = '__smooth',
                    group = color_column
                ),
                color = smooth_color,
                size = smooth_size,
                alpha = smooth_alpha
            )

    # Add theme
    g = g + \
        theme_tq(base_size=base_size) 
    
    return g
    
    
    
    
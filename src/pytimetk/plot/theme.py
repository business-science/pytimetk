from plotnine import *


    
def theme_timetk(base_size: int = 11, base_family: list = ['Arial', 'Helvetica', 'sans-serif'], dpi: int = 100, width: int = 700, height: int = 500):
    '''
    Returns a `plotnine` theme with timetk styles applied, allowing for 
    customization of the appearance of plots in Python.
    
    Parameters
    ----------
    base_size : int, optional
        The `base_size` parameter determines the base font size for the theme. 
        It is set to 11 by default, but you can change it to any desired value.
    base_family : list
        The `base_family` parameter is a list of font families that will be used 
        as the base font for the theme. The default value is `['Arial', 
        'Helvetica', 'sans-serif']`, which means that the theme will use Arial 
        font if available, otherwise it will try Helvetica, and if that is not 
        available either, it will use the generic sans-serif font.
    dpi : int, optional
        The `dpi` parameter stands for dots per inch and determines the 
        resolution of the plot. It specifies the number of pixels per inch in 
        the output image. Higher dpi values result in higher resolution images.
    width : int, optional
        The `width` parameter is used to specify the width of the plot in pixels 
        at dpi. It determines the horizontal size of the plot. The default value 
        is 700 pixels.
    height : int, optional
        The `height` parameter is used to specify the height of the plot in 
        inches. It is an optional parameter, so if you don't provide a value for 
        it, the default height will be 5 inches (500 pixels).

    
    Returns
    -------
        A theme object that can be used to customize the appearance of plots in 
        Python. The theme object contains various elements such as line, rect, 
        axis, panel, legend, strip, and plot, each with their own set of 
        properties that can be customized.
        
    Examples
    --------
    ```{python}
    import pytimetk as tk
    import pandas as pd
    
    from plotnine import ggplot, aes, geom_line, labs, scale_x_date, facet_wrap
    
    data = {
        'date': pd.date_range(start='2023-01-01', end='2023-01-10'),
        'value': [1, 3, 7, 9, 11, 14, 18, 21, 24, 29]
    }
    df = pd.DataFrame(data)
    
    # Plotnine chart without styling
    fig = (
        ggplot(df, aes(x='date', y='value')) +
            geom_line(color='blue') +
            labs(title='Time Series Plot', x='Date', y='Value') +
            scale_x_date(date_labels='%a')
    )
    fig

    ```
    
    ```{python}
    # Plotnine chart with timetk styling
    fig + tk.theme_timetk()
    ```
    
    ```{python}
    # Faceted plot with timetk styling
    data = {
        'date': pd.date_range(start='2023-01-01', end='2023-01-10').tolist() * 2,
        'value': [1, 3, 7, 9, 11, 14, 18, 21, 24, 29] * 2,
        'category': ['A'] * 10 + ['B'] * 10,
    }
    df = pd.DataFrame(data)
    
    (
        ggplot(df, aes(x='date', y='value')) +
            geom_line(color='blue') +
            labs(title='Faceted Time Series Plot', x='Date', y='Value') +
            facet_wrap('~category') +
            scale_x_date(date_labels='%a') +
            tk.theme_timetk()
    )
    ```
    
    '''
    
    # Tidyquant colors
    blue  = "#2c3e50"
    green = "#18BC9C"
    white = "#FFFFFF"
    grey  = "#cccccc"
    
    return theme(
        
        # # Base Inherited Elements
        line = element_line(color = blue, size = 0.5),
        rect = element_rect(fill = white, colour = blue, size = 0.5),
        
        # Axes
        axis_line=element_blank(),
        axis_text=element_text(size=base_size * 0.6),
        axis_ticks=element_line(color=grey, size=0.5),
        axis_title=element_text(size=base_size*1),
        
        axis_text_y=element_text(margin=dict(r=5)),
        
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
        legend_title=element_blank(),
        legend_background=element_blank(),
        
        # Strip
        strip_background=element_rect(fill = blue, color = blue),
        strip_text=element_text(size=base_size*0.8, color = white, margin=dict(t=5, b=5)),
        
        # Plot
        plot_title=element_text(size=base_size*1.2, color = blue, margin=dict(t = 0, r = 0, b = 4, l = 0), hjust=0),
        plot_subtitle=element_text(size=base_size*0.9, color = blue, margin=dict(t = 0, r = 0, b = 3, l = 0)),
        plot_margin=0.025,
        
        dpi=dpi,
        figure_size=(width/100 if width is not None else 7, height/100 if height is not None else 5),
        
    )
    
    
        
def palette_timetk():
    '''
    The function `palette_timetk` returns a dictionary of color codes for 
    various colors in the timetk theme.
        
    - blue         = "#2c3e50", # blue
    - red          = "#e31a1c", # red
    - green        = "#18BC9C", # green
    - yellow       = "#CCBE93", # yellow
    - steel_blue   = "#a6cee3", # steel_blue
    - navy_blue    = "#1f78b4", # navy_blue
    - light_green  = "#b2df8a", # light_green
    - pink         = "#fb9a99", # pink
    - light_orange = "#fdbf6f", # light_orange
    - orange       = "#ff7f00", # orange
    - light_purple = "#cab2d6", # light_purple
    - purple       = "#6a3d9a"  # purple
    
    Returns
    -------
        The function `palette_timetk` returns a dictionary containing color 
        names as keys and their corresponding hexadecimal color codes as values:
    
    Examples
    --------
    
    ```{python}
    import pytimetk as tk
    
    tk.palette_timetk()
    ```
    
    '''
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

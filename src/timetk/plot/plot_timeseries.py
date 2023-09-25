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
    '''The `plot_timeseries` function is a Python function that creates a time series plot using the
    `ggplot` library, allowing for customization of various plot elements such as line color, size, and
    type, as well as the addition of y and x intercepts and facetting by group variables.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The `data` parameter is the input data for the plot. It can be either a pandas DataFrame or a
    pandas DataFrameGroupBy object.
    date_column
        The name of the column in the dataframe that contains the dates for the time series plot.
    value_column
        The `value_column` parameter is used to specify the column in the DataFrame that contains the
    values to be plotted on the y-axis of the time series plot.
    facet_vars
        The `facet_vars` parameter is used to specify the variables that will be used for faceting the
    plot. It can be a single variable or a list of variables. Each variable will create a separate facet
    in the plot.
    facet_ncol, optional
        The `facet_ncol` parameter determines the number of columns in the facet grid. It specifies how
    many plots should be arranged horizontally in each row of the grid.
    facet_nrow, optional
        The `facet_nrow` parameter specifies the number of rows in the facet grid. It determines how many
    rows of plots will be displayed in the facet grid.
    facet_scales, optional
        The `facet_scales` parameter in the `plot_timeseries` function determines the scaling of the y-axis
    in the facetted plots.
    facet_dir, optional
        The `facet_dir` parameter determines the direction in which the facets are arranged. It can take
    two possible values:
    facet_collapse, optional
        The `facet_collapse` parameter determines whether the facet labels should be collapsed into a
    single line or displayed as separate lines. If `facet_collapse` is set to `True`, the facet labels
    will be collapsed into a single line using the `facet_collapse_sep` parameter as the separator.
    facet_collapse_sep, optional
        The `facet_collapse_sep` parameter is used to specify the separator character(s) to be used when
    collapsing facet labels. This parameter is only applicable when `facet_collapse` is set to `True`.
    By default, the separator is set to a space character (" ").
    facet_strip_remove, optional
        The `facet_strip_remove` parameter is a boolean value that determines whether the facet strip
    labels should be removed or not. If set to `True`, the facet strip labels will be removed from the
    plot. If set to `False` (default), the facet strip labels will be displayed on the plot
    title, optional
        The title of the plot. It is set to "Time Series Plot" by default.
    x_lab
        The label for the x-axis of the plot. It is used to provide a description or name for the x-axis
    variable.
    y_lab
        The `y_lab` parameter in the `plot_timeseries` function is used to specify the label for the y-axis
    of the plot. It is a string that represents the desired label for the y-axis.
    color_lab, optional
        The `color_lab` parameter is used to specify the label for the legend in the plot. It determines
    the title or description that will be displayed for the color scale used in the plot.
    line_color, optional
        The line_color parameter is used to specify the color of the line in the plot. It accepts a string
    value representing a color code or name. The default value is "#2c3e50", which corresponds to a dark
    blue color.
    line_size
        The line_size parameter in the plot_timeseries function is used to specify the size of the line in
    the plot. It determines the thickness of the line.
    line_type, optional
        The parameter "line_type" in the "plot_timeseries" function is used to specify the type of line to
    be used in the plot. It accepts the following values:
    line_alpha, optional
        The parameter "line_alpha" is used to control the transparency of the line in the plot. It accepts
    a value between 0 and 1, where 0 means completely transparent (invisible) and 1 means completely
    opaque (solid).
    y_intercept
        The `y_intercept` parameter is used to add a horizontal line at a specific y-value on the plot. It
    is an optional parameter and its default value is `None`. If you want to add a y-intercept line, you
    can pass the desired y-value to this parameter. For example
    y_intercept_color, optional
        The parameter "y_intercept_color" is used to specify the color of the Y-intercept line in the plot.
    By default, it is set to "#2c3e50", which is a dark gray color. You can change this parameter to any
    valid color value, such as a color
    x_intercept
        The `x_intercept` parameter is used to add a vertical line at a specific x-coordinate on the plot.
    It can be set to a single value or a list of values to add multiple vertical lines. The default
    value is `None`, which means no vertical line will be added.
    x_intercept_color, optional
        The parameter "x_intercept_color" is used to specify the color of the vertical line that represents
    the x-intercept in the plot. By default, it is set to "#2c3e50", which is a dark blue color.
    However, you can change it to any valid color value
    interactive, optional
        The `interactive` parameter is a boolean flag that determines whether the plot should be
    interactive or not. If set to `True`, the plot will be interactive, allowing you to zoom, pan, and
    interact with the plot elements. If set to `False`, the plot will be static and non-
    
    Returns
    -------
        a ggplot object, which represents the plot that will be displayed.
    
    Examples
    --------
    ```{python}
    import pandas as pd
    import timetk as tk
    
    df = tk.load_dataset('m4_monthly', parse_dates = ['date'])
    df
    
    df.groupby('id').plot_timeseries('date', 'value')
    
    ```
    
    '''
    
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
    g = g + scale_x_datetime(date_labels = "%b %Y")

    # Add facets
    if group_names is not None:
       g = g + facet_wrap(
            group_names,
            ncol = 2, scales = "free", shrink = True
        )

    return g

# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_timeseries = plot_timeseries
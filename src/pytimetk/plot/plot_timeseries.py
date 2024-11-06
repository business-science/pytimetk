import pandas as pd
import numpy as np
import pandas_flavor as pf

from plotnine import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.nonparametric.smoothers_lowess import lowess

from pytimetk.plot.theme import theme_timetk, palette_timetk
from pytimetk.utils.plot_helpers import hex_to_rgba, name_to_hex

from typing import Union, Optional, List

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

    
        
    
@pf.register_dataframe_method
def plot_timeseries(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    value_column: Union[str, List[str]],
    
    color_column: Union[str, List[str]] = None,
    color_palette: Optional[Union[dict, list, str]] = None,

    facet_ncol: int = 1,
    facet_nrow: Optional[int] = None,
    facet_scales: str = "free_y",
    facet_dir: str = "h", 

    line_color: str = "#2c3e50",
    line_size: float = None,
    line_type: str = 'solid',
    line_alpha: float = 1.0,
    
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    x_intercept: Optional[str] = None,
    x_intercept_color: str = "#2c3e50",
    
    smooth: bool = True,
    smooth_color: str = "#3366FF",
    smooth_frac: float = 0.2,
    smooth_size: float = 1.0,
    smooth_alpha: float = 1.0,
    
    legend_show: bool = True,
    
    title: str = "Time Series Plot",
    x_lab: str = "",
    y_lab: str = "",
    color_lab: str = "Legend",
    
    x_axis_date_labels: str = "%b %Y",
    base_size: float = 11,
    width: Optional[int] = None,
    height: Optional[int] = None,

    engine: str = 'plotly',
    
    plotly_dropdown: bool = False,
    plotly_dropdown_x: float = 0,
    plotly_dropdown_y: float = 1,

):
    '''
    Creates time series plots using different plotting engines such as Plotnine, 
    Matplotlib, and Plotly.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.core.groupby.generic.DataFrameGroupBy
        The input data for the plot. It can be either a Pandas DataFrame or a 
        Pandas DataFrameGroupBy object.
    date_column : str
        The name of the column in the DataFrame that contains the dates for the 
        time series data.
    value_column : str or list
        The `value_column` parameter is used to specify the name of the column 
        in the DataFrame that contains the values for the time series data. This 
        column will be plotted on the y-axis of the time series plot.
        
        LONG-FORMAT PLOTTING:
        If the `value_column` parameter is a string, it will be treated as a 
        single column name. To plot multiple time series,
        group the DataFrame first using pd.DataFrame.groupby().
        
        WIDE-FORMAT PLOTTING:
        If the `value_column` parameter is a list, it will plotted
        as multiple time series (wide-format).         
    color_column : str
        The `color_column` parameter is an optional parameter that specifies the 
        column in the DataFrame that will be used to assign colors to the 
        different time series. If this parameter is not provided, all time 
        series will have the same color.
        
        LONG-FORMAT PLOTTING:
        The `color_column` parameter is a single column name.
        
        WIDE-FORMAT PLOTTING:
        The `color_column` parameter must be the same list 
        as the `value_column` parameter to color the different time series when performing wide-format plotting.
    color_palette : list, optional
        The `color_palette` parameter is used to specify the colors to be used 
        for the different time series. It accepts a list of color codes or names. 
        If the `color_column` parameter is not provided, the `tk.palette_timetk()` 
        color palette will be used.
    facet_ncol : int, optional
        The `facet_ncol` parameter determines the number of columns in the facet 
        grid. It specifies how many subplots will be arranged horizontally in 
        the plot.
    facet_nrow : int
        The `facet_nrow` parameter determines the number of rows in the facet 
        grid. It specifies how many subplots will be arranged vertically in the 
        grid.
    facet_scales : str, optional
        The `facet_scales` parameter determines the scaling of the y-axis in the 
        facetted plots. It can take the following values:
        - "free_y": The y-axis scale will be free for each facet, but the x-axis 
        scale will be fixed for all facets. This is the default value.
        - "free_x": The y-axis scale will be free for each facet, but the x-axis 
        scale will be fixed for all facets.
        - "free": The y-axis scale will be free for each facet (subplot). This 
        is the default value.
        
    facet_dir : str, optional
        The `facet_dir` parameter determines the direction in which the facets 
        (subplots) are arranged. It can take two possible values:
        - "h": The facets will be arranged horizontally (in rows). This is the 
          default value.
        - "v": The facets will be arranged vertically (in columns).
    line_color : str, optional
        The `line_color` parameter is used to specify the color of the lines in 
        the time series plot. It accepts a string value representing a color 
        code or name. The default value is "#2c3e50", which corresponds to a 
        dark blue color.
    line_size : float
        The `line_size` parameter is used to specify the size of the lines in 
        the time series plot. It determines the thickness of the lines.
    line_type : str, optional
        The `line_type` parameter is used to specify the type of line to be used 
        in the time series plot.         
    line_alpha : float
        The `line_alpha` parameter controls the transparency of the lines in the 
        time series plot. It accepts a value between 0 and 1, where 0 means 
        completely transparent (invisible) and 1 means completely opaque (solid).
    y_intercept : float
        The `y_intercept` parameter is used to add a horizontal line to the plot 
        at a specific y-value. It can be set to a numeric value to specify the 
        y-value of the intercept. If set to `None` (default), no y-intercept 
        line will be added to the plot
    y_intercept_color : str, optional
        The `y_intercept_color` parameter is used to specify the color of the 
        y-intercept line in the plot. It accepts a string value representing a 
        color code or name. The default value is "#2c3e50", which corresponds to 
        a dark blue color. You can change this value.
    x_intercept : str
        The `x_intercept` parameter is used to add a vertical line at a specific 
        x-axis value on the plot. It is used to highlight a specific point or 
        event in the time series data. 
        - By default, it is set to `None`, which means no vertical line will be 
          added. 
        - You can use a date string to specify the x-axis value of the intercept. 
          For example, "2020-01-01" would add a vertical line at the beginning 
          of the year 2020.
    x_intercept_color : str, optional
        The `x_intercept_color` parameter is used to specify the color of the 
        vertical line that represents the x-intercept in the plot. By default, 
        it is set to "#2c3e50", which is a dark blue color. You can change this 
        value to any valid color code.
    smooth : bool, optional
        The `smooth` parameter is a boolean indicating whether or not to apply 
        smoothing to the time eries data. If set to True, the time series will 
        be smoothed using the lowess algorithm. The default value is True.
    smooth_color : str, optional
        The `smooth_color` parameter is used to specify the color of the 
        smoothed line in the time series plot. It accepts a string value 
        representing a color code or name. The default value is `#3366FF`, 
        which corresponds to a shade of blue. You can change this value to any 
        valid color code.
    smooth_frac : float
        The `smooth_frac` parameter is used to control the fraction of data 
        points used for smoothing the time series. It determines the degree of 
        smoothing applied to the data. A smaller value of `smooth_frac` will 
        result in more smoothing, while a larger value will result in less 
        smoothing. The default value is 0.2.
    smooth_size : float
        The `smooth_size` parameter is used to specify the size of the line used 
        to plot the smoothed values in the time series plot. It is a numeric 
        value that controls the thickness of the line. A larger value will result 
        in a thicker line, while a smaller value will result in a thinner line
    smooth_alpha : float
        The `smooth_alpha` parameter controls the transparency of the smoothed 
        line in the plot. It accepts a value between 0 and 1, where 0 means 
        completely transparent and 1 means completely opaque.
    legend_show : bool, optional
        The `legend_show` parameter is a boolean indicating whether or not to 
        show the legend in the plot. If set to True, the legend will be 
        displayed. The default value is True.
    title : str, optional
        The title of the plot.
    x_lab : str
        The `x_lab` parameter is used to specify the label for the x-axis in the 
        plot. It is a string that represents the label text.
    y_lab : str
        The `y_lab` parameter is used to specify the label for the y-axis in the 
        plot. It is a string that represents the label for the y-axis.
    color_lab : str, optional
        The `color_lab` parameter is used to specify the label for the legend or 
        color scale in the plot. It is used to provide a description of the colors 
        used in the plot, typically when a color column is specified.
    x_axis_date_labels : str, optional
        The `x_axis_date_labels` parameter is used to specify the format of the 
        date labels on the x-axis of the plot. It accepts a string representing 
        the format of the date labels. For  example, "%b %Y" would display the 
        month abbreviation and year (e.g., Jan 2020).
    base_size : float, optional
        The `base_size` parameter is used to set the base font size for the plot. 
        It determines the size of the text elements such as axis labels, titles, 
        and legends.
    width : int
        The `width` parameter is used to specify the width of the plot. It 
        determines the horizontal size of the plot in pixels.
    height : int
        The `height` parameter is used to specify the height of the plot in 
        pixels. It determines the vertical size of the plot when it is rendered.
    engine : str, optional
        The `engine` parameter specifies the plotting library to use for 
        creating the time series plot. It can take one of the following values:
        
        - "plotly" (interactive): Use the plotly library to create the plot. 
          This is the default value.
        - "plotnine" (static): Use the plotnine library to create the plot. 
          This is the default value.
        - "matplotlib" (static): Use the matplotlib library to create the plot.
    plotly_dropdown : bool
        For analyzing many plots. When set to True and groups are provided, the function switches from 
        faceting to create a dropdown menu to switch between different groups. Default: `False`.
    plotly_dropdown_x : float
        The x-axis location of the dropdown. Default: 0.
    plotly_dropdown_y : float
        The y-axis location of the dropdown. Default: 1.
    
        
    
    Returns
    -------
        The function `plot_timeseries` returns a plot object, depending on the 
        specified `engine` parameter. 
        - If `engine` is set to 'plotnine' or 'matplotlib', the function returns 
          a plot object that can be further customized or displayed. 
        - If `engine` is set to 'plotly', the function returns a plotly figure 
          object.
    
    
    Examples
    --------
    ```{python}
    import pytimetk as tk
    
    df = tk.load_dataset('m4_monthly', parse_dates = ['date'])
    
    # Plotly Object: Single Time Series
    fig = (
        df
            .query('id == "M750"')
            .plot_timeseries(
                'date', 'value', 
                facet_ncol = 1,
                x_axis_date_labels = "%Y",
                engine = 'plotly',
            )
    )
    fig
    ```
    
    ```{python}
    # Plotly Object: Grouped Time Series (Facets)
    fig = (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value', 
                facet_ncol = 2,
                facet_scales = "free_y",
                smooth_frac = 0.2,
                smooth_size = 2.0,
                y_intercept = None,
                x_axis_date_labels = "%Y",
                engine = 'plotly',
                width = 600,
                height = 500,
            )
    )
    fig
    ```
    
    ```{python}
    # Plotly Object: Grouped Time Series (Plotly Dropdown)
    fig = (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value', 
                facet_scales = "free_y",
                smooth_frac = 0.2,
                smooth_size = 2.0,
                y_intercept = None,
                x_axis_date_labels = "%Y",
                engine = 'plotly',
                width = 600,
                height = 500,
                plotly_dropdown = True, # Plotly Dropdown
            )
    )
    fig
    ```
    
    ```{python}    
    # Plotly Object: Color Column
    fig = (
        df
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                smooth = False,
                y_intercept = 0,
                x_axis_date_labels = "%Y",
                engine = 'plotly',
            )
    )
    fig
    ```
    
    ```{python}
    # Plotnine Object: Single Time Series
    fig = (
        df
            .query('id == "M1"')
            .plot_timeseries(
                'date', 'value', 
                x_axis_date_labels = "%Y",
                engine = 'plotnine'
            )
    )
    fig
    ```
    
    ```{python}
    # Plotnine Object: Grouped Time Series
    fig = (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value',
                facet_ncol = 2,
                facet_scales = "free",
                line_size = 0.35,
                x_axis_date_labels = "%Y",
                engine = 'plotnine'
            )
    )
    fig
    ```
    
    ```{python}
    # Plotnine Object: Color Column
    fig = (
        df
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                smooth = False,
                y_intercept = 0,
                x_axis_date_labels = "%Y",
                engine = 'plotnine',
            )
    )
    fig
    ```
    
    ```{python}
    # Matplotlib object (same as plotnine, but converted to matplotlib object)
    fig = (
        df
            .groupby('id')
            .plot_timeseries(
                'date', 'value', 
                color_column = 'id',
                facet_ncol = 2,
                x_axis_date_labels = "%Y",
                engine = 'matplotlib',
            )
    )
    fig
    ```
    
    ``` {python}
    # Wide-Format Plotting
    
    # Imports
    import pandas as pd
    import numpy as np
    import pytimetk as tk

    # Set a random seed for reproducibility
    np.random.seed(42) 

    # Create a date range
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    # Generate random sales data and compute expenses and profit
    sales = np.random.uniform(1000, 5000, len(dates))
    expenses = sales * np.random.uniform(0.5, 0.8, len(dates))
    profit = sales - expenses

    # Create the DataFrame
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'expenses': expenses,
        'profit': profit
    })
    
    (
        df
            .plot_timeseries(
                date_column = 'date', 
                value_column = ['sales', 'expenses', 'profit'],
                color_column = ['sales', 'expenses', 'profit'], 
                smooth = True,
                x_axis_date_labels = "%Y",
                engine = 'plotly',
                plotly_dropdown = True, # Plotly Dropdown
            )
    )
    ```
    '''
    
    # Common checks
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)

    if isinstance(value_column, list):
        for col in value_column:
            check_value_column(data, col)
    else:
        check_value_column(data, value_column)

    # Handle line_size
    if line_size is None:
        if engine == 'plotnine':
            line_size = 0.35
        elif engine == 'matplotlib':
            line_size = 0.35
        elif engine == 'plotly':
            line_size = 0.65

    # Handle named colors
    line_color = name_to_hex(line_color)
    smooth_color = name_to_hex(smooth_color)
    y_intercept_color = name_to_hex(y_intercept_color)
    x_intercept_color = name_to_hex(x_intercept_color)

    # Initialize group_names
    group_names = None

    # Handle DataFrames and GroupBy objects
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.copy()
    elif isinstance(data, pd.DataFrame):
        data = data.copy()
    else:
        raise ValueError("data must be a DataFrame or GroupBy object")

    # Reshape data if value_column is a list (wide format)
    if isinstance(value_column, list):
        if color_column == value_column:
            color_column = '__value_column'  # Use variable names as the color column
        
        if group_names is None:
            id_vars = [date_column]
        else:
            id_vars = group_names + [date_column]
        
        data = pd.melt(
            data,
            id_vars=id_vars,
            value_vars=value_column,
            var_name='__value_column',
            value_name='__value'
        )
        
        value_column = '__value'  # Update value_column to the new column name
        
        if group_names is None:
            group_names = ['__value_column']
        else:
            group_names = group_names + ['__value_column']
            
        legend_show = False
    else:
        # Ensure value_column is in data
        if value_column not in data.columns:
            raise ValueError(f"value_column '{value_column}' not found in DataFrame.")

    # Handle smoother
    if smooth:
        data['__smooth'] = np.nan
        # Determine grouping columns
        grouping_columns = []
        if group_names is not None:
            grouping_columns.extend(group_names)
        if color_column is not None and color_column not in grouping_columns:
            grouping_columns.append(color_column)
        if grouping_columns:
            # Apply smoother per group
            for _, group in data.groupby(grouping_columns):
                sorted_group = group.sort_values(by=date_column)
                x = np.arange(len(sorted_group))
                y = sorted_group[value_column].to_numpy()
                smoothed = lowess(y, x, frac=smooth_frac)
                data.loc[sorted_group.index, '__smooth'] = smoothed[:, 1]
        else:
            # Apply smoother to the whole data
            sorted_data = data.sort_values(by=date_column)
            x = np.arange(len(sorted_data))
            y = sorted_data[value_column].to_numpy()
            data['__smooth'] = lowess(y, x, frac=smooth_frac, return_sorted=False)

    # Handle color palette
    if color_palette is None:
        unique_colors = data[color_column].nunique() if color_column else 1
        color_palette = list(palette_timetk().values()) * unique_colors
    else:
        if isinstance(color_palette, dict):
            color_palette = list(palette_timetk(color_palette).values())
        elif isinstance(color_palette, list):
            color_palette = color_palette
        elif isinstance(color_palette, str):
            color_palette = [color_palette]
        else:
            raise ValueError("Invalid `color_palette` parameter. It must be a dictionary, list, or string.")


    # Engine
    if engine in ['plotnine', 'matplotlib']:
        fig = _plot_timeseries_plotnine(
            data=data,
            date_column=date_column,
            value_column=value_column,
            color_column=color_column,
            color_palette=color_palette,
            group_names=group_names,
            facet_ncol=facet_ncol,
            facet_nrow=facet_nrow,
            facet_scales=facet_scales,
            facet_dir=facet_dir,
            line_color=line_color,
            line_size=line_size,
            line_type=line_type,
            line_alpha=line_alpha,
            y_intercept=y_intercept,
            y_intercept_color=y_intercept_color,
            x_intercept=x_intercept,
            x_intercept_color=x_intercept_color,
            smooth=smooth,
            smooth_color=smooth_color,
            smooth_size=smooth_size,
            smooth_alpha=smooth_alpha,
            legend_show=legend_show,
            title=title,
            x_lab=x_lab,
            y_lab=y_lab,
            color_lab=color_lab,
            x_axis_date_labels=x_axis_date_labels,
            base_size=base_size,
            width=width,
            height=height,
        )

        if engine == 'matplotlib':
            if width is None:
                width_size = 800  # in pixels for compatibility with plotly
            else:
                width_size = width

            if height is None:
                height_size = 600  # in pixels for compatibility with plotly
            else:
                height_size = height
            fig = fig + theme_timetk(height=height_size, width=width_size)  # Setting default figure size
            fig = fig.draw()

    elif engine == 'plotly':
        fig = _plot_timeseries_plotly(
            data=data,
            date_column=date_column,
            value_column=value_column,
            color_column=color_column,
            color_palette=color_palette,
            group_names=group_names,
            facet_ncol=facet_ncol,
            facet_nrow=facet_nrow,
            facet_scales=facet_scales,
            facet_dir=facet_dir,
            line_color=line_color,
            line_size=line_size,
            line_type=line_type,
            line_alpha=line_alpha,
            y_intercept=y_intercept,
            y_intercept_color=y_intercept_color,
            x_intercept=x_intercept,
            x_intercept_color=x_intercept_color,
            smooth=smooth,
            smooth_color=smooth_color,
            smooth_size=smooth_size,
            smooth_alpha=smooth_alpha,
            legend_show=legend_show,
            title=title,
            x_lab=x_lab,
            y_lab=y_lab,
            color_lab=color_lab,
            x_axis_date_labels=x_axis_date_labels,
            base_size=base_size,
            width=width,
            height=height,
            plotly_dropdown=plotly_dropdown,
            plotly_dropdown_x=plotly_dropdown_x,
            plotly_dropdown_y=plotly_dropdown_y,
        )

    return fig


# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_timeseries = plot_timeseries


def _plot_timeseries_plotly(
    data,
    date_column,
    value_column,
    color_column=None,
    color_palette=None,
    group_names=None,
    plotly_dropdown=False,  
    plotly_dropdown_x=0,
    plotly_dropdown_y=1,
    facet_ncol=1,
    facet_nrow=None,
    facet_scales="free_y",
    facet_dir="h",
    line_color="#2c3e50",
    line_size=0.3,
    line_type='solid',
    line_alpha=1,
    y_intercept=None,
    y_intercept_color="#2c3e50",
    x_intercept=None,
    x_intercept_color="#2c3e50",
    smooth=None,
    smooth_color="#3366FF",
    smooth_size=0.3,
    smooth_alpha=1,
    legend_show=True,
    title="Time Series Plot",
    x_lab="",
    y_lab="",
    color_lab="Legend",
    x_axis_date_labels="%b %Y",
    base_size=11,
    width=None,
    height=None,
):
    """This function is not intended to be called directly. It is used by the `plot_timeseries` function."""
    
    data = data.copy()
    
    # Assign colors to groups
    if color_column is not None:
        if not isinstance(color_column, list):
            color_column = [color_column]
        
        data[color_column] = data[color_column].astype(str)
        
        color_df = data[color_column].drop_duplicates().reset_index(drop=True)
        
        color_df["_color"] = (color_palette * 1_000_000)[:len(color_df)]  
        
        color_df["_color_group_names"] =  color_df[color_column].agg(" | ".join, axis=1)
    
    # Handle dropdown functionality
    if plotly_dropdown and group_names is not None:
        if not isinstance(group_names, list):
            group_names = [group_names]
        
        data[group_names] = data[group_names].astype(str)
        data["_group_names"] = data[group_names].agg(" | ".join, axis=1)
        
        grouped = data.groupby("_group_names", sort=False)
        unique_group_names = data["_group_names"].unique()
        
        traces = []
        visibility_per_trace = []
        group_values = []
        shapes_per_group = {}
        
        for i, (group_name, group) in enumerate(grouped):
            if color_column is not None:
                color_group = group.merge(color_df, on=color_column, how="left")
                for name, color_subgroup in color_group.groupby("_color_group_names"):
                    color = color_subgroup['_color'].unique()[0]
                    color = name_to_hex(color)
                    trace = go.Scatter(
                        x=color_subgroup[date_column],
                        y=color_subgroup[value_column],
                        mode='lines',
                        line=dict(color=hex_to_rgba(color, alpha=line_alpha), width=line_size),
                        name=name,
                        legendgroup=name,
                        showlegend=(i == 0),
                        visible=(i == 0),
                    )
                    traces.append(trace)
                    visibility_per_trace.append(i)
                    group_values.append(group_name)
            else:
                trace = go.Scatter(
                    x=group[date_column],
                    y=group[value_column],
                    mode='lines',
                    line=dict(color=hex_to_rgba(line_color, alpha=line_alpha), width=line_size),
                    name=group_name,
                    showlegend=False,
                    visible=(i == 0),
                )
                traces.append(trace)
                visibility_per_trace.append(i)
                group_values.append(group_name)
            
            # Add smooth line if applicable
            if smooth:
                # Assuming 'smooth' processing is done prior and stored in '__smooth' column
                trace = go.Scatter(
                    x=group[date_column],
                    y=group['__smooth'],
                    mode='lines',
                    line=dict(color=hex_to_rgba(smooth_color, alpha=smooth_alpha), width=smooth_size),
                    name='Smoother',
                    showlegend=False,
                    visible=(i == 0),
                )
                traces.append(trace)
                visibility_per_trace.append(i)
                group_values.append(group_name)
            
            # Collect shapes for y_intercept and x_intercept
            shapes = []
            if y_intercept is not None:
                if not isinstance(y_intercept, list):
                    y_intercept = [y_intercept]
                for y in y_intercept:
                    shape = dict(
                        type="line",
                        y0=y, y1=y,
                        x0=group[date_column].min(), x1=group[date_column].max(),
                        line=dict(color=y_intercept_color, width=1),
                    )
                    shapes.append(shape)
            if x_intercept is not None:
                if not isinstance(x_intercept, list):
                    x_intercept = [x_intercept]
                for x in x_intercept:
                    shape = dict(
                        type="line",
                        x0=x, x1=x,
                        y0=group[value_column].min(), y1=group[value_column].max(),
                        line=dict(color=x_intercept_color, width=1),
                    )
                    shapes.append(shape)
            shapes_per_group[group_name] = shapes
        
        # Create visibility settings for each group
        visibility_lists = []
        for group_name in unique_group_names:
            visibility = [gv == group_name for gv in group_values]
            visibility_lists.append(visibility)
        
        # Create dropdown buttons
        dropdown_buttons = []
        for idx, group_name in enumerate(unique_group_names):
            button = dict(
                label=group_name,
                method='update',
                args=[
                    {'visible': visibility_lists[idx]},
                    {
                        'title': f"{title} - {group_name}",
                        'shapes': shapes_per_group.get(group_name, []),
                    }
                ]
            )
            dropdown_buttons.append(button)
        
        # Create figure and add traces
        fig = go.Figure(data=traces)
        
        # Update layout with dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    active=0,
                    buttons=dropdown_buttons,
                    x=plotly_dropdown_x,
                    xanchor="left",
                    y=plotly_dropdown_y,
                    yanchor="top",
                    showactive=True,
                    font=dict(
                        size=10  # Adjust the font size to make the dropdown text smaller
                    ),
                    pad=dict(
                        r=5,  # Reduce right padding
                        t=5,  # Reduce top padding
                        l=5,  # Reduce left padding
                        b=5,  # Reduce bottom padding
                    ),
                    # bgcolor='rgba(255, 255, 255, 0.5)',  # Set a transparent background for a sleek look
                    # bordercolor='gray',  # Optionally add a border color
                    borderwidth=1 
                )
            ],
            title=f"{title} - {unique_group_names[0]}",
            xaxis_title=x_lab,
            yaxis_title=y_lab,
            legend_title_text=color_lab,
            xaxis=dict(tickformat=x_axis_date_labels),
        )
        
        # Apply styling and layout updates
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=40))
        fig.update_layout(
            template="plotly_white", 
            font=dict(size=base_size),
            title_font=dict(size=base_size*1.2),
            legend_title_font=dict(size=base_size*0.8),
            legend_font=dict(size=base_size*0.8),
        )
        fig.update_xaxes(tickfont=dict(size=base_size*0.8))
        fig.update_yaxes(tickfont=dict(size=base_size*0.8))
        fig.update_layout(
            autosize=True, 
            width=width,
            height=height,
        )
        fig.update_traces(hoverlabel=dict(font_size=base_size*0.8))
        
        if not legend_show:
            fig.update_layout(showlegend=False)
        
        return fig
    
    else:
        # Existing functionality when plotly_dropdown is False
        # SETUP SUBPLOTS ----
        subplot_titles = []
        if group_names is not None:
            if not isinstance(group_names, list):
                group_names = [group_names]
            
            data[group_names] = data[group_names].astype(str)
            group_lookup_df = data[group_names].drop_duplicates().reset_index(drop=True)
            group_lookup_df["_group_names"] =  group_lookup_df[group_names].agg(" | ".join, axis=1)
            grouped = data.groupby(group_names, sort=False, group_keys=False)
            num_groups = len(grouped)
            facet_nrow = -(-num_groups // facet_ncol)  # Ceil division
            subplot_titles = [" | ".join(map(str, name)) if isinstance(name, tuple) else str(name) for name in grouped.groups.keys()]
            if color_column is not None:
                colors = color_palette * 1_000_000
                colors = colors[:num_groups]
            else: 
                colors = [line_color] * num_groups
        else:
            facet_nrow = 1
            num_groups = 1
        
        fig = make_subplots(
            rows=facet_nrow, 
            cols=facet_ncol, 
            subplot_titles=subplot_titles,
            shared_xaxes=True if facet_scales == "free_y" else False,
            shared_yaxes=True if facet_scales == "free_x" else False,
        )
        
        # ADD TRACES -----
        if group_names is not None:
            for i, (name, group) in enumerate(grouped):
                row = i // facet_ncol + 1
                col = i % facet_ncol + 1
                if color_column is not None:
                    color_group = group.merge(color_df, on=color_column, how="left")
                    color_group = color_group.merge(group_lookup_df, on=group_names, how="left")
                    for j, (name, color_group) in enumerate(color_group.groupby("_color_group_names")):
                        color = color_group['_color'].unique()[0]
                        grp_nm = color_group['_group_names'].unique()[0]
                        name = color_group['_color_group_names'].unique()[0]
                        color = name_to_hex(color)
                        trace = go.Scatter(
                            x=color_group[date_column], 
                            y=color_group[value_column], 
                            mode='lines',
                            line=dict(color=hex_to_rgba(color, alpha=line_alpha), width=line_size), 
                            name=name, 
                            legendgroup=name,
                        )
                        fig.add_trace(trace, row=row, col=col)
                        fig.layout.annotations[i].update(text=grp_nm)
                    # Remove duplicate legends in each legend group
                    seen_legendgroups = set()
                    for trace in fig.data:
                        legendgroup = trace.legendgroup
                        if legendgroup in seen_legendgroups:
                            trace.showlegend = False
                        else:
                            seen_legendgroups.add(legendgroup)
                else:
                    group_group = group.merge(group_lookup_df, on=group_names, how="left")
                    grp_nm = group_group['_group_names'].unique()[0]
                    trace = go.Scatter(
                        x=group_group[date_column], 
                        y=group_group[value_column], 
                        mode='lines', 
                        line=dict(color=hex_to_rgba(line_color, alpha=line_alpha), width=line_size), 
                        showlegend=False, 
                        name=group_group[value_column].name
                    )
                    fig.add_trace(trace, row=row, col=col)
                    fig.layout.annotations[i].update(text=grp_nm)
                if smooth:
                    trace = go.Scatter(
                        x=group[date_column], 
                        y=group['__smooth'], 
                        mode='lines', 
                        line=dict(color=hex_to_rgba(smooth_color, alpha=smooth_alpha), width=smooth_size), 
                        showlegend=False, 
                        name="Smoother"
                    )
                    fig.add_trace(trace, row=row, col=col)
                if y_intercept is not None:
                    if not isinstance(y_intercept, list):
                        y_intercept = [y_intercept]
                    for y in y_intercept:
                        fig.add_shape(
                            go.layout.Shape(
                                type="line", y0=y, y1=y, x0=group[date_column].min(), x1=group[date_column].max(), 
                                line=dict(color=y_intercept_color, width=1)
                            ), 
                            row=row, col=col
                        )
                if x_intercept is not None:
                    if not isinstance(x_intercept, list):
                        x_intercept = [x_intercept]
                    for x in x_intercept:
                        fig.add_shape(
                            go.layout.Shape(
                                type="line", x0=x, x1=x, y0=group[value_column].min(), y1=group[value_column].max(), 
                                line=dict(color=x_intercept_color, width=1)
                            ), 
                            row=row, col=col
                        )
        else:
            if color_column is not None:
                color_group = data.merge(color_df, on=color_column, how="left")
                for j, (name, color_group) in enumerate(color_group.groupby("_color")):
                    color = color_group['_color'].unique()[0]
                    name = color_group['_color_group_names'].unique()[0]
                    color = name_to_hex(color)
                    trace = go.Scatter(
                        x=color_group[date_column], 
                        y=color_group[value_column], 
                        mode='lines',
                        line=dict(color=hex_to_rgba(color, alpha=line_alpha), width=line_size), 
                        name=name, 
                        legendgroup=name,
                    )
                    fig.add_trace(trace)
                # Remove duplicate legends in each legend group
                seen_legendgroups = set()
                for trace in fig.data:
                    legendgroup = trace.legendgroup
                    if legendgroup in seen_legendgroups:
                        trace.showlegend = False
                    else:
                        seen_legendgroups.add(legendgroup)
            else:
                trace = go.Scatter(
                    x=data[date_column], 
                    y=data[value_column], 
                    mode='lines', 
                    line=dict(color=hex_to_rgba(line_color, alpha=line_alpha), width=line_size), 
                    showlegend=False, 
                    name="Time Series"
                )
                fig.add_trace(trace)
            if smooth:
                trace = go.Scatter(
                    x=data[date_column], 
                    y=data['__smooth'], 
                    mode='lines', 
                    line=dict(color=hex_to_rgba(smooth_color, alpha=smooth_alpha), width=smooth_size), 
                    showlegend=False, 
                    name="Smoother"
                )
                fig.add_trace(trace)
            if y_intercept is not None:
                if not isinstance(y_intercept, list):
                    y_intercept = [y_intercept]
                for y in y_intercept:
                    fig.add_shape(
                        go.layout.Shape(
                            type="line", y0=y, y1=y, x0=data[date_column].min(), x1=data[date_column].max(), 
                            line=dict(color=y_intercept_color, width=1)
                        )
                    )
            if x_intercept is not None:
                if not isinstance(x_intercept, list):
                    x_intercept = [x_intercept]
                for x in x_intercept:
                    fig.add_shape(
                        go.layout.Shape(
                            type="line", x0=x, x1=x, y0=data[value_column].min(), y1=data[value_column].max(), 
                            line=dict(color=x_intercept_color, width=1)
                        )
                    )
        # Finalize the plotly plot
        fig.update_layout(
            title=title,
            xaxis_title=x_lab,
            yaxis_title=y_lab,
            legend_title_text=color_lab,
            xaxis=dict(tickformat=x_axis_date_labels),
        )
        fig.update_xaxes(
            matches=None, showticklabels=True, visible=True, 
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=40))
        fig.update_layout(
            template="plotly_white", 
            font=dict(size=base_size),
            title_font=dict(size=base_size*1.2),
            legend_title_font=dict(size=base_size*0.8),
            legend_font=dict(size=base_size*0.8),
        )
        fig.update_xaxes(tickfont=dict(size=base_size*0.8))
        fig.update_yaxes(tickfont=dict(size=base_size*0.8))
        fig.update_annotations(font_size=base_size*0.8)
        fig.update_layout(
            autosize=True, 
            width=width,
            height=height,
        )
        fig.update_traces(hoverlabel=dict(font_size=base_size*0.8))
        
        if not legend_show:
            fig.update_layout(showlegend=False)
       
        return fig


def _plot_timeseries_plotnine(
    data: pd.DataFrame,
    date_column,
    value_column,
    
    color_column = None,
    color_palette = None,
    
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
    smooth_size = 0.3,
    smooth_alpha = 1,
    
    legend_show = True,
    
    title = "Time Series Plot",
    x_lab = "",
    y_lab = "",
    color_lab = "Legend",
    
    x_axis_date_labels = "%b %Y",
    base_size = 11,
    
    
    width = None,
    height = None,
):
    """This function is not intended to be called directly. It is used by the `plot_timeseries` function."""
    
    # Data Setup
    
    if color_column is not None:
        
        if not isinstance(color_column, list):
            color_column = [color_column]
        
        data[color_column] = data[color_column].astype(str) 
        
        data["_color_group_names"] =  data[color_column].agg(" | ".join, axis=1)
        
        data["_color_group_names"] = pd.Categorical(data["_color_group_names"], categories=data["_color_group_names"].unique(), ordered=True)
        
    if group_names is not None:
        
        if not isinstance(group_names, list):
            group_names = [group_names]
        
        data[group_names] = data[group_names].astype(str) 
        
        data["_group_names"] =  data[group_names].agg(" | ".join, axis=1)
        
        data['_group_names'] = pd.Categorical(data['_group_names'], categories=data['_group_names'].unique(), ordered=True)
    
    
    # Plot setup
    if group_names is not None:
        if color_column is not None:
            g = ggplot(
                data = data,
                mapping = aes(
                    x = date_column,
                    y = value_column,
                    group = "_color_group_names",
                    color = "_color_group_names"
                )
            )
        else:
            g = ggplot(
                data = data,
                mapping = aes(
                    x = date_column,
                    y = value_column,
                    group = "_group_names",
                )
            )
    else:
        g = ggplot(
            data = data,
            mapping = aes(
                x = date_column,
                y = value_column,
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
                        color = "_color_group_names"
                    ),
                    size     = line_size,
                    linetype = line_type,
                    alpha    = line_alpha
                ) \
            + scale_color_manual(
                values=color_palette
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
            "_group_names",
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
                    group = "_color_group_names",
                ),
                color = smooth_color,
                size = smooth_size,
                alpha = smooth_alpha
            )

    # Add theme
    g = g + \
        theme_timetk(base_size=base_size, width = width, height = height)
        
    if not legend_show:
        g = g + theme(legend_position='none')
    
    return g
    

    
    
    
    
    
    

import pandas as pd
import pandas_flavor as pf

from typing import Union, Optional

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_anomalize_data


@pf.register_dataframe_method
def plot_anomalies_decomp(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    
    line_color: str = "#2c3e50",
    line_size: Optional[float] = None,
    line_type: str = 'solid',
    line_alpha: float = 1.0,
    
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    x_intercept: Optional[str] = None,
    x_intercept_color: str = "#2c3e50",
    
    title: str = "Anomaly Decomposition Plot",
    x_lab: str = "",
    y_lab: str = "",
    
    x_axis_date_labels: str = "%b %Y",
    base_size: float = 11,
    width: Optional[int] = None,
    height: Optional[int] = None,

    engine: str = 'plotly',    
):
    '''
    The `plot_anomalies_decomp` function takes in data from the `anomalize()` 
    function, and returns a plot of the anomaly decomposition.
    
    Parameters
    ----------
    data : Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]
        The input data for the plot from `anomalize`. It can be either a pandas 
        DataFrame or a pandas DataFrameGroupBy object.
    date_column : str
        The name of the column in the data that contains the dates.
    line_color : str, optional
        The color of the line in the plot. It is specified as a hexadecimal 
        color code. The default value is "#2c3e50".
    line_size : Optional[float]
        The line_size parameter determines the thickness of the lines in the 
        plot. It is an optional parameter, so if you don't specify a value, the 
        default line size will be used.
    line_type : str, optional
        The `line_type` parameter specifies the type of line to be used in the 
        plot. It can take the following values:
        - "solid" (default): a solid line
        - "dashed": a dashed line
    line_alpha : float
        The `line_alpha` parameter controls the transparency of the lines in the 
        plot. It accepts a float value between 0 and 1, where 0 means completely 
        transparent and 1 means completely opaque.
    y_intercept : Optional[float]
        The `y_intercept` parameter is an optional float value that specifies 
        the y-coordinate of a horizontal line to be plotted on the graph. This 
        line can be used to indicate a specific threshold or reference value. If 
        not specified, no horizontal line will be plotted.
    y_intercept_color : str, optional
        The `y_intercept_color` parameter is used to specify the color of the 
        y-intercept line on the plot. By default, it is set to `"#2c3e50"`, 
        which is a dark blue color. You can change this parameter to any valid 
        color code or name to change the color of the line.
    x_intercept : Optional[str]
        The `x_intercept` parameter is used to specify the value on the x-axis 
        where you want to draw a vertical line. This can be useful for 
        highlighting a specific point or event in the data.
    x_intercept_color : str, optional
        The `x_intercept_color` parameter is used to specify the color of the 
        vertical line representing the x-intercept on the plot. By default, it 
        is set to "#2c3e50", which is a dark blue color. You can change this 
        parameter to any valid color code or name to change the color of the line.
    title : str, optional
        The title of the plot. It is set to "Anomaly Decomposition Plot" by default.
    x_lab : str
        The x_lab parameter is used to specify the label for the x-axis of the 
        plot. It is a string that represents the label text.
    y_lab : str
        The `y_lab` parameter is used to specify the label for the y-axis of the 
        plot. It is a string that represents the label text.
    x_axis_date_labels : str, optional
        The `x_axis_date_labels` parameter is used to specify the format of the 
        date labels on the x-axis of the plot. It accepts a string representing 
        the format of the date labels. For example, "%b %Y" would display the 
        month abbreviation and year (e.g., Jan 2019).
    base_size : float, optional
        The `base_size` parameter determines the base font size for the plot. It
        is used to control the size of the text elements in the plot, such as 
        axis labels, titles, and tick labels. The default value is 11, but you 
        can adjust it to make the text larger or smaller
    width : Optional[int]
        The width parameter determines the width of the plot in pixels. It is an 
        optional parameter, so if you don't specify a value, the plot will be 
        displayed with the default width.
    height : Optional[int]
        The height parameter determines the height of the plot in pixels. It is 
        an optional parameter, so if you don't specify a value, the plot will be 
        displayed with a default height.
    engine : str, optional
        The `engine` parameter specifies the plotting engine to use. It can be 
        set to either "plotly", "plotnine", or "matplotlib".
    
    Returns
    -------
        A plotly, plotnine, or matplotlib plot.
    
    See Also
    --------
    1. anomalize : Function that calculates the anomalies and formats the data 
    for visualization.
    2. plot_anomalies : Function that plots the anomalies.
    
    Examples
    --------
    
    ``` {python}
    # EXAMPLE 1: SINGLE TIME SERIES
    import pytimetk as tk
    import pandas as pd
    import numpy as np

    # Create a date range
    date_rng = pd.date_range(start='2021-01-01', end='2024-01-01', freq='MS')

    # Generate some random data with a few outliers
    np.random.seed(42)
    data = np.random.randn(len(date_rng)) * 10 + 25  
    data[3] = 100  # outlier

    # Create a DataFrame
    df = pd.DataFrame(date_rng, columns=['date'])
    df['value'] = data
    
    # Anomalize the data
    anomalize_df = tk.anomalize(
        df, "date", "value",
        method = "twitter", 
        iqr_alpha = 0.10, 
        clean_alpha = 0.75,
        clean = "min_max",
        verbose = True,
    )
    
    # Visualize the results, plotly
    anomalize_df.plot_anomalies_decomp("date", engine = 'plotly')
    ```
    
    ```{python}
    # Visualize the results, plotnine
    anomalize_df.plot_anomalies_decomp("date", engine = "plotnine")
    ```
    
    ``` {python}
    # EXAMPLE 2: MULTIPLE TIME SERIES
    import pytimetk as tk
    import pandas as pd
    
    df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"])[["id", "Date", "Weekly_Sales"]]
    
    anomalize_df = (
        df
            .groupby('id') 
            .anomalize(
                "Date", "Weekly_Sales", 
                period = 52, 
                trend = 52, 
                threads = 1
            ) 
    )
    
    # Visualize the decomposition results, plotly
    (
        anomalize_df
            .groupby("id")
            .plot_anomalies_decomp(
                date_column = "Date",
                line_color = "steelblue",
                width = 1200,
                height = 800,
                x_axis_date_labels = "%y",
                engine = 'plotly',                
            )
    )
    ```
    
    ```{python}
    # Visualize the decomposition results, plotnine
    
    (
        anomalize_df
            .groupby("id")
            .plot_anomalies_decomp(
                date_column = "Date",
                line_color = "steelblue",
                width = 1200,
                height = 800,
                x_axis_date_labels = "%y",
                engine = 'plotnine',                
            )
    )
    ```
    '''
    
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    check_anomalize_data(data)
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.copy()
        data = data[[*group_names, date_column, 'observed', 'seasonal', 'trend', 'remainder']]
        facet_ncol = data[group_names].drop_duplicates().shape[0]
    else:
        group_names = None
        data = data.copy()
        data = data[[date_column, 'observed', 'seasonal', 'trend', 'remainder']]
        facet_ncol = 1
        
        
    data_prepared = data.melt(
        id_vars = [date_column] if group_names is None else group_names + [date_column], 
        var_name='_var',
        value_name='_val',
    )
    
    fig = data_prepared.groupby(
        ['_var'] if group_names is None else group_names + ['_var']
    ).plot_timeseries(
        date_column = date_column, 
        value_column = '_val',
        
        facet_ncol = facet_ncol, 
        facet_dir = 'h',
        
        line_color = line_color,
        line_size = line_size,
        line_type = line_type,
        line_alpha = line_alpha,
        
        smooth = False,
        
        y_intercept = y_intercept,
        y_intercept_color = y_intercept_color,
        x_intercept = x_intercept,
        x_intercept_color = x_intercept_color,
        
        title = title,
        x_lab = x_lab,
        y_lab = y_lab,
        
        x_axis_date_labels = x_axis_date_labels,
        base_size = base_size,
        
        width = width,
        height = height,
        
        engine = engine,
    )
    
    return fig
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_anomalies_decomp = plot_anomalies_decomp


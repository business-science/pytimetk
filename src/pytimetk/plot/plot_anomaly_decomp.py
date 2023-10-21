import pandas as pd
import pandas_flavor as pf

from typing import Union, Optional

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_anomalize_data


@pf.register_dataframe_method
def plot_anomaly_decomp(
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
pd.core.groupby.generic.DataFrameGroupBy.plot_anomaly_decomp = plot_anomaly_decomp


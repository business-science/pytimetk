import pandas as pd
import numpy as np
import pandas_flavor as pf

from plotnine import *

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from mizani.breaks import date_breaks
from mizani.formatters import date_format

from statsmodels.nonparametric.smoothers_lowess import lowess

from pytimetk.plot.theme import theme_timetk, palette_timetk
from pytimetk.utils.plot_helpers import hex_to_rgba, rgba_to_hex

from typing import Union, Optional

from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column, check_value_column

@pf.register_dataframe_method
def plot_anomalies(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,
    
    color_column: Optional[str] = None,
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
    
    title: str = "Anomaly Plot",
    x_lab: str = "",
    y_lab: str = "",
    color_lab: str = "Legend",
    
    x_axis_date_labels: str = "%b %Y",
    base_size: float = 11,
    width: Optional[int] = None,
    height: Optional[int] = None,

    engine: str = 'plotly'
):
    
    group_names = None
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.copy()
    else:
        data = data.copy()
    
    if group_names is None:
        1+1
    else:
        data[group_names] = data[group_names].astype(str)
    
        fig = (
            data[[*group_names, date_column, "observed", "recomposed_l1", "recomposed_l2"]] 
                .melt(
                    id_vars = [*group_names, date_column], value_name='_val',
                    var_name = '_var',
                ) 
                .groupby(group_names) 
                .plot_timeseries(
                    date_column, "_val", 
                    color_column = "_var",
                    color_palette = ["#2c3e50", '#64646433', '#64646433'],
                    facet_ncol = 2, 
                    smooth = False,
                    width = 800,
                    height = 800,
                )
        )
        
        # 2.0 Add Ribbon
        
        # Create lists to store the indices of the traces
        indices_l1 = [i for i, trace in enumerate(fig.data) if trace.name == 'recomposed_l1']
        indices_l2 = [i for i, trace in enumerate(fig.data) if trace.name == 'recomposed_l2']

        # Iterate through the pairs of indices and update the fig.data list
        for i_l1, i_l2 in zip(indices_l1, indices_l2):
            # Ensure 'recomposed_l1' comes before 'recomposed_l2' in fig.data
            if i_l1 > i_l2:
                i_l1, i_l2 = i_l2, i_l1  # Swap the indices
                fig.data[i_l1], fig.data[i_l2] = fig.data[i_l2], fig.data[i_l1]  # Swap the traces

            # Update the 'recomposed_l2' trace to fill towards 'recomposed_l1'
            fig.data[i_l2].update(fill='tonexty', fillcolor=hex_to_rgba('#64646433'))  # Adjust fill color as needed

        # 3.0 Add Red dots
        
        # Assuming fig.data contains your trace data and you know the number of subplots
        num_subplots = len(fig.layout.annotations)  # Assuming each subplot has one annotation
        
        def generate_triples(n):
            result = []
            for i in range(1, 3*n+1, 3):
                result.append([i, i+1, i+2])
            return result
        
        tripples = generate_triples(num_subplots)

        for tripple in tripples:
            
            observed_data = None
            recomposed_l2_data = None
            recomposed_l1_data = None
            
            for i in tripple:
                i-=1
                print(i)
                
                
                # Extract data for each subplot
                
                print(fig.data[i].name)
                
                if fig.data[i].name == 'observed':
                    observed_data = fig.data[i]
                elif fig.data[i].name == 'recomposed_l1':
                    recomposed_l1_data = fig.data[i]
                elif fig.data[i].name == 'recomposed_l2':
                    recomposed_l2_data = fig.data[i]
            
                print("observed_data")
                print(observed_data)
                
                print("recomposed_l1_data")
                print(recomposed_l1_data)
                
                print("recomposed_l2_data")
                print(recomposed_l2_data)

            # Ensure we have the data
            if observed_data and recomposed_l2_data and recomposed_l1_data:
                # Identify points where condition is met
                x_values = []
                y_values = []
                for x, y, y_l2, y_l1 in zip(observed_data.x, observed_data.y, recomposed_l2_data.y, recomposed_l1_data.y):
                    if y > y_l2 or y < y_l1:
                        x_values.append(x)
                        y_values.append(y)
                
                print(x_values)
                print(y_values)
                print("---")
                
                # Add scatter plot with identified points to the correct subplot
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='markers',
                        marker=dict(color='red', size=6),
                        name=f'anomalies',
                        legendgroup='anomalies',
                        xaxis=observed_data.xaxis,
                        yaxis=observed_data.yaxis
                    )
                )
                
        return fig
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_anomalies = plot_anomalies




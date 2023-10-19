import pandas as pd
import pandas_flavor as pf
from typing import Union, Optional
from functools import partial
import plotly.graph_objects as go

from plotnine import *

from pytimetk.plot.plot_timeseries import plot_timeseries
from pytimetk.utils.plot_helpers import hex_to_rgba, rgba_to_hex, parse_rgba
from pytimetk.utils.checks import check_dataframe_or_groupby, check_date_column
from pytimetk.plot.theme import theme_timetk


@pf.register_dataframe_method
def plot_anomalies(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,

    facet_ncol: int = 1,
    facet_nrow: Optional[int] = None,
    facet_scales: str = "free_y",
    facet_dir: str = "h", 

    line_color: str = "#2c3e50",
    line_size: Optional[float] = None,
    line_type: str = 'solid',
    line_alpha: float = 1.0,
    
    anom_color: str = "#E31A1C",
    anom_alpha: float = 1.0,
    anom_size: Optional[float] = None,
    
    ribbon_fill: str = "#646464",
    ribbon_alpha: float = 0.2,
    
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    x_intercept: Optional[str] = None,
    x_intercept_color: str = "#2c3e50",
    
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
    
    # Check data
    check_dataframe_or_groupby(data)
    check_date_column(data, date_column)
    
    # Handle line_size
    if line_size is None:
        if engine == 'plotnine':
            line_size = 0.35
        elif engine == 'matplotlib':
            line_size = 0.35
        elif engine == 'plotly':
            line_size = 0.65
    
    if engine == 'plotly':
        
        if anom_size is None:
            anom_size = 6.0
        
        
        fig = _plot_anomalies_plotly(
            data = data,
            date_column = date_column,

            facet_ncol = facet_ncol,
            facet_nrow = facet_nrow,
            facet_scales = facet_scales,
            facet_dir = facet_dir, 

            line_color = line_color,
            line_size = line_size,
            line_type = line_type,
            line_alpha = line_alpha,
            
            anom_color = anom_color,
            anom_alpha = anom_alpha,
            anom_size = anom_size,
            
            ribbon_fill = ribbon_fill,
            ribbon_alpha = ribbon_alpha,
            
            y_intercept = y_intercept,
            y_intercept_color = y_intercept_color,
            x_intercept = x_intercept,
            x_intercept_color = x_intercept_color,
            
            legend_show = legend_show,
            
            title = title,
            x_lab = x_lab,
            y_lab = y_lab,
            color_lab = color_lab,
            
            x_axis_date_labels = x_axis_date_labels,
            base_size = base_size,
            width = width,
            height = height,
        )
        
    else:
        
        if anom_size is None:
            anom_size = 1.0
            
        fig = _plot_anomalies_plotnine(
            data = data,
            date_column = date_column,

            facet_ncol = facet_ncol,
            facet_nrow = facet_nrow,
            facet_scales = facet_scales,
            facet_dir = facet_dir, 

            line_color = line_color,
            line_size = line_size,
            line_type = line_type,
            line_alpha = line_alpha,
            
            anom_color = anom_color,
            anom_alpha = anom_alpha,
            anom_size = anom_size,
            
            ribbon_fill = ribbon_fill,
            ribbon_alpha = ribbon_alpha,
            
            y_intercept = y_intercept,
            y_intercept_color = y_intercept_color,
            x_intercept = x_intercept,
            x_intercept_color = x_intercept_color,
            
            legend_show = legend_show,
            
            title = title,
            x_lab = x_lab,
            y_lab = y_lab,
            color_lab = color_lab,
            
            x_axis_date_labels = x_axis_date_labels,
            base_size = base_size,
            width = width,
            height = height,
        )
        
        if engine == 'matplotlib':
            if width == None:
                width_size = 800 # in pixels for compat with plotly
            else: 
                width_size = width

            if height == None:
                height_size = 600 # in pixels for compat with plotly
            else:
                height_size = height
            fig = fig + theme_timetk(height=height_size, width=width_size) # setting default figure size to prevent matplotlib sizing error
            fig = fig.draw()
     
    return fig
    
# Monkey patch the method to pandas groupby objects
pd.core.groupby.generic.DataFrameGroupBy.plot_anomalies = plot_anomalies


def _plot_anomalies_plotly(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,

    facet_ncol: int = 1,
    facet_nrow: Optional[int] = None,
    facet_scales: str = "free_y",
    facet_dir: str = "h", 

    line_color: str = "#2c3e50",
    line_size: float = None,
    line_type: str = 'solid',
    line_alpha: float = 1.0,
    
    anom_color: str = "#E31A1C",
    anom_alpha: float = 1.0,
    anom_size: Optional[float] = None,
    
    ribbon_fill: str = "#646464",
    ribbon_alpha: float = 0.2,
    
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    x_intercept: Optional[str] = None,
    x_intercept_color: str = "#2c3e50",
    
    legend_show: bool = True,
    
    title: str = "Anomaly Plot",
    x_lab: str = "",
    y_lab: str = "",
    color_lab: str = "Legend",
    
    x_axis_date_labels: str = "%b %Y",
    base_size: float = 11,
    width: Optional[int] = None,
    height: Optional[int] = None,
    
):
    
    # Plot Setup
    ribbon_color_rgba = hex_to_rgba(ribbon_fill, ribbon_alpha)
    ribbon_color_rgba_list = parse_rgba(ribbon_color_rgba)
    ribbon_color_hex = rgba_to_hex(*ribbon_color_rgba_list)
    
    preload_plot_timeseries = partial(
        plot_timeseries,
                    
        date_column = date_column,
        value_column = "_val", 
        color_column = "_var",
        
        color_palette = [
            line_color, 
            ribbon_color_hex, 
            ribbon_color_hex
        ],
        
        facet_ncol = facet_ncol,
        facet_nrow = facet_nrow,
        facet_scales = facet_scales,
        facet_dir = facet_dir,
        
        line_size = line_size,
        line_type = line_type,
        line_alpha = line_alpha,
        
        y_intercept = y_intercept,
        y_intercept_color = y_intercept_color,
        x_intercept = x_intercept,
        x_intercept_color = x_intercept_color,
        
        smooth = False,
        
        legend_show = legend_show,
        
        title = title,
        x_lab = x_lab,
        y_lab = y_lab,
        color_lab = color_lab,
        
        x_axis_date_labels = x_axis_date_labels,
        
        base_size = base_size,
        
        width = width,
        height = height,
    )
    
    # 1.0 Create Main Plot
    
    is_grouped = False
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        group_names = data.grouper.names
        data = data.obj.copy()
        is_grouped = True
    else:
        data = data.copy()
        data["_grp"] = "Time Series"
        group_names = ["_grp"]
    
    
    data[group_names] = data[group_names].astype(str)

    data_prepared = (
        data[[*group_names, date_column, "observed", "recomposed_l1", "recomposed_l2"]] 
            .melt(
                id_vars = [*group_names, date_column], 
                value_name='_val',
                var_name = '_var',
            ) 
            .groupby(group_names) 
    )
    
    fig = preload_plot_timeseries(data_prepared)
                
        
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
        fig.data[i_l2].update(fill='tonexty', fillcolor=ribbon_color_rgba)  # Adjust fill color as needed

    # 3.0 Add Anomaly Dots
    
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
            
            # Extract data for each subplot
            
            if fig.data[i].name == 'observed':
                observed_data = fig.data[i]
            elif fig.data[i].name == 'recomposed_l1':
                recomposed_l1_data = fig.data[i]
                fig.data[i].legendgroup = "bands"
                fig.data[i].showlegend = False
            elif fig.data[i].name == 'recomposed_l2':
                recomposed_l2_data = fig.data[i]
                fig.data[i].legendgroup = "bands"
                fig.data[i].showlegend = False

        # Ensure we have the data
        if observed_data and recomposed_l2_data and recomposed_l1_data:
            # Identify points where condition is met
            x_values = []
            y_values = []
            for x, y, y_l2, y_l1 in zip(observed_data.x, observed_data.y, recomposed_l2_data.y, recomposed_l1_data.y):
                if y > y_l2 or y < y_l1:
                    x_values.append(x)
                    y_values.append(y)
            
            # Add scatter plot with identified points to the correct subplot
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        color=anom_color, 
                        size=anom_size,
                        opacity=anom_alpha,
                    ),
                    name='anomalies',
                    legendgroup='anomalies',
                    xaxis=observed_data.xaxis,
                    yaxis=observed_data.yaxis
                )
            )
    
    # Remove duplicate legends in each legend group
    seen_legendgroups = set()
    for trace in fig.data:
        legendgroup = trace.legendgroup
        if legendgroup in seen_legendgroups:
            trace.showlegend = False
        else:
            seen_legendgroups.add(legendgroup)
    
    # Cleanup annotations if ungrouped
    if not is_grouped:
        fig['layout']['annotations'] = []
    
    # Apply Styling
    fig.update_annotations(font_size=base_size*0.8)     
    fig.update_traces(hoverlabel=dict(font_size=base_size*0.8))
            
    return fig


def _plot_anomalies_plotnine(
    data: Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy],
    date_column: str,

    facet_ncol: int = 1,
    facet_nrow: Optional[int] = None,
    facet_scales: str = "free_y",
    facet_dir: str = "h", 

    line_color: str = "#2c3e50",
    line_size: float = None,
    line_type: str = 'solid',
    line_alpha: float = 1.0,
    
    anom_color: str = "#E31A1C",
    anom_alpha: float = 1.0,
    anom_size: Optional[float] = None,
    
    ribbon_fill: str = "#646464",
    ribbon_alpha: float = 0.2,
    
    y_intercept: Optional[float] = None,
    y_intercept_color: str = "#2c3e50",
    x_intercept: Optional[str] = None,
    x_intercept_color: str = "#2c3e50",
    
    legend_show: bool = True,
    
    title: str = "Anomaly Plot",
    x_lab: str = "",
    y_lab: str = "",
    color_lab: str = "Legend",
    
    x_axis_date_labels: str = "%b %Y",
    base_size: float = 11,
    width: Optional[int] = None,
    height: Optional[int] = None,
    
):
    
    if isinstance(data, pd.core.groupby.generic.DataFrameGroupBy):
        
        group_names = data.grouper.names
        
        data = data.obj.copy()
        
        if not isinstance(group_names, list):
            group_names = [group_names]
        
        data[group_names] = data[group_names].astype(str) 
        
        data["_group_names"] =  data[group_names].agg(" | ".join, axis=1)
        
    else:
        group_names = None
        data = data.copy()
    
    # 1.0 Canvas
    if group_names is not None:
        g = ggplot(
                data = data,
                mapping = aes(
                    x = date_column,
                    y = "observed",
                    group = "_group_names",
                )
            )
    
    else:
        g = ggplot(
            data = data,
            mapping = aes(
                x = date_column,
                y = "observed",
            )
        )
        
    # 2.0 Add facets
    if group_names is not None:
       g = g + facet_wrap(
            "_group_names",
            ncol = facet_ncol,
            nrow = facet_nrow, 
            scales = facet_scales, 
            dir = facet_dir, 
            shrink = True
        )
    
    # 3.0 Add Ribbons
    g = g + geom_ribbon(
        mapping = aes(
            ymin = "recomposed_l1",
            ymax = "recomposed_l2",
        ),
        alpha = ribbon_alpha,
        fill  = ribbon_fill,
    )
    
    # 4.0 Add Lines
    g = g + \
        geom_line(
            color    = line_color,
            size     = line_size,
            linetype = line_type,
            alpha    = line_alpha
        )
    
    # 5.0 Add Anomalies
    if anom_size is None:
        anom_size = 2.0
    
    g = g + \
        geom_point(
            mapping = aes(
                x = date_column,
                y = "observed",
            ),
            color = anom_color,
            size = anom_size,
            alpha = anom_alpha,
            data = data.query("observed > recomposed_l2 | observed < recomposed_l1"),
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
    
    # Add theme
    g = g + \
        theme_timetk(base_size=base_size, width = width, height = height)
        
    if not legend_show:
        g = g + theme(legend_position='none')
    
    return g
    
    
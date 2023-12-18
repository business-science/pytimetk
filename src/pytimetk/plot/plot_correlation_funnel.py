import pandas as pd
import numpy as np
import pandas_flavor as pf

import plotly.express as px

from plotnine import ggplot, aes, geom_vline, geom_point, geom_text, labs, theme_minimal, xlim

from pytimetk.plot.theme import theme_timetk, palette_timetk

@pf.register_dataframe_method
def plot_correlation_funnel(
    data, 
    limits=(-1, 1), 
    alpha=1,
    title = "Correlation Funnel Plot",
    x_lab = "Correlation",
    y_lab = "Feature",
    base_size = 11,
    width = None,
    height = None,
    interactive=True, 
):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("plot_correlation_funnel(): Object is not of class `pd.DataFrame`.")
    
    if interactive:
          
        fig = px.scatter(
            data, 
            x='correlation', 
            y='feature', 
            hover_data={'correlation':':.3f', 'feature':True, 'bin':True},
            range_x=limits, 
            title='Correlation Funnel'
        )
        
        # Finalize the plotly plot
    
        fig.update_layout(
            title=title,
            xaxis_title=x_lab,
            yaxis_title=y_lab,
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
        fig.update_yaxes(
            tickfont=dict(size=base_size*0.8), 
            autorange="reversed"
        )
        fig.update_annotations(font_size=base_size*0.8)
        fig.update_layout(
            autosize=True, 
            width=width,
            height=height,
            showlegend=False,
        )
        fig.update_traces(
            marker=dict(color='#2c3e50', opacity=alpha), 
            selector=dict(mode='markers'),
            text = data['bin'],
            hoverlabel=dict(font_size=base_size*0.8)
        )
        
        return fig

    else:
        
        data['feature'] = pd.Categorical(data['feature'], categories=data['feature'].unique()[::-1], ordered=True)
        
        p = (
            ggplot(data)
            + aes(x='correlation', y='feature', label='bin')
            + geom_vline(xintercept=0, linetype='dashed', color='red')
            + geom_point(color='#2c3e50', alpha=alpha)
            + geom_text(size=base_size*0.8, color='#2c3e50', nudge_y=0.3)
            + labs(title=title, x=x_lab, y=y_lab)
            + xlim(limits[0], limits[1])
            + theme_minimal()
        )
        p = p + theme_timetk(base_size=base_size, width = width, height = height)
        
        return p
        

import pandas as pd
import numpy as np
import pandas_flavor as pf

# from plotnine import ggplot, aes, geom_vline, geom_point, geom_text, labs, theme_minimal, theme, element_text
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

@pf.register_dataframe_method
def plot_correlation_funnel(
    data, 
    interactive=False, 
    limits=(-1, 1), 
    alpha=1
):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("plot_correlation_funnel(): Object is not of class `pd.DataFrame`.")
    
    if interactive:
        data['label_text'] = data.apply(lambda row: f"{row['feature']}\nCorrelation: {row['correlation']:.3f}", axis=1)
        
        fig = px.scatter(
            data, 
            x='correlation', 
            y='feature', 
            text='label_text',
            range_x=limits, 
            title='Correlation Funnel'
        )
        
        fig.update_traces(marker=dict(color='#2c3e50', opacity=alpha), selector=dict(mode='markers'))
        fig.update_layout(shapes=[dict(type='line', x0=0, x1=0, y0=0, y1=1, yref='paper', line=dict(color='red', dash='dash'))])
        fig.update_xaxes(title_text="Correlation")
        fig.update_yaxes(title_text="Feature")
        fig.update_layout(showlegend=False)
        
        return fig

    else:
        fig, ax = plt.subplots()
        ax.scatter(data['correlation'], data['feature'], c='#2c3e50', alpha=alpha)
        
        for i, row in data.iterrows():
            ax.text(row['correlation'], row['feature'], size=12, color='#2c3e50')
        
        ax.axvline(x=0, linestyle='--', color='red')
        ax.set_xlim(limits)
        ax.set_xlabel('Correlation')
        ax.set_ylabel('Feature')
        ax.set_title('Correlation Funnel')
        
        return plt.show()

# def _plot_correlation_funnel_plotnine(data, interactive=False, limits=(-1, 1), alpha=1):
#     if not isinstance(data, pd.DataFrame):
#         raise ValueError("plot_correlation_funnel(): Object is not of class `pd.DataFrame`.")
    
#     if interactive:
#         data['label_text'] = data.apply(lambda row: f"{row['feature']}\nCorrelation: {row['correlation']:.3f}", axis=1)

#         p = (
#             ggplot(data)
#             + aes(x='correlation', y='feature', text='label_text')
#             + geom_vline(xintercept=0, linetype='dashed', color='red')
#             + geom_point(color='#2c3e50', alpha=alpha)
#             + labs(title='Correlation Funnel')
#             + theme_minimal()
#         )
#         p = p + theme(axis_text_x=element_text(size=12))
        
#         return p

#     else:
#         p = (
#             ggplot(data)
#             + aes(x='correlation', y='feature', label='feature')
#             + geom_vline(xintercept=0, linetype='dashed', color='red')
#             + geom_point(color='#2c3e50', alpha=alpha)
#             + geom_text(size=12, color='#2c3e50')
#             + labs(title='Correlation Funnel')
#             + theme_minimal()
#         )
#         p = p + theme(axis_text_x=element_text(size=12))
        
#         return p
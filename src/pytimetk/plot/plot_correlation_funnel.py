import pandas as pd
import numpy as np
import pandas_flavor as pf

from typing import Optional

import plotly.express as px

from plotnine import (
    ggplot,
    aes,
    geom_vline,
    geom_point,
    geom_segment,
    geom_text,
    labs,
    scale_y_continuous,
    xlim,
)

from pytimetk.plot.theme import theme_timetk


@pf.register_dataframe_method
def plot_correlation_funnel(
    data: pd.DataFrame,
    limits: tuple = (-1, 1),
    alpha: float = 1.0,
    title: str = "Correlation Funnel Plot",
    x_lab: str = "Correlation",
    y_lab: str = "Feature",
    base_size: float = 11,
    point_color: str = "#2c3e50",
    width: Optional[int] = None,
    height: Optional[int] = None,
    engine: str = "plotly",
):
    """The `plot_correlation_funnel` function generates a correlation funnel plot using either Plotly or
    plotnine in Python.

    Parameters
    ----------
    data : pd.DataFrame
        The `data` parameter is a DataFrame that contains the correlation values and corresponding
        features. It should have two columns: 'correlation' and 'feature'.
    limits : tuple
        The `limits` parameter is a tuple that specifies the lower and upper limits of the x-axis in the
        correlation funnel plot. By default, the limits are set to (-1, 1), which means the x-axis will
        range from -1 to 1.
    alpha : float
        The `alpha` parameter determines the transparency of the data points in the plot. A value of 1.0
        means the points are fully opaque, while a value less than 1.0  makes the points more transparent.
    title : str, optional
        The title of the plot.
    x_lab : str, optional
        The `x_lab` parameter is used to specify the label for the x-axis of the plot. It represents the
        label for the correlation values.
    y_lab : str, optional
        The `y_lab` parameter is used to specify the label for the y-axis in the correlation funnel plot.
        It represents the name or description of the feature being plotted.
    base_size : float, optional
        The `base_size` parameter is used to set the base font size for the plot. It is multiplied by
        different factors to determine the font sizes for various elements of the plot, such as the title,
        axis labels, tick labels, legend, and annotations.
    width : Optional[int]
        The `width` parameter is used to specify the width of the plot in pixels. It determines the
        horizontal size of the plot.
    height : Optional[int]
        The `height` parameter is used to specify the height of the plot in pixels. It determines the
        vertical size of the plot when it is rendered. Static plots grow with
        the number of features when no height is supplied.
    engine : str, optional
        The `engine` parameter determines the plotting engine to be used. It can be set to either "plotly"
        or "plotnine". If set to "plotly", the function will generate an interactive plot using the
        Plotly library. If set to "plotnine", it will generate a static
        plot using the plotnine library. The default value is "plotly".

    Returns
    -------
        The function `plot_correlation_funnel` returns a plotly figure object if the `engine` parameter is
        set to 'plotly', and a plotnine object if the `engine` parameter is set to 'plotnine'.
        When called via the polars `.tk` accessor the data is converted to pandas internally before plotting.

    See Also
    --------
    - `binarize()`: Binarize the dataset into 1's and 0's.
    - `correlate()`: Calculate the correlation between features in a DataFrame.

    Examples
    --------
    ``` {python}
    # NON-TIMESERIES EXAMPLE ----

    import pandas as pd
    import numpy as np
    import pytimetk as tk

    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of rows for your DataFrame
    num_rows = 200

    # Create fake data for the columns
    data = {
        'Age': np.random.randint(18, 65, size=num_rows),
        'Gender': np.random.choice(['Male', 'Female'], size=num_rows),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], size=num_rows),
        'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], size=num_rows),
        'Years_Playing': np.random.randint(0, 30, size=num_rows),
        'Average_Income': np.random.randint(20000, 100000, size=num_rows),
        'Member_Status': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], size=num_rows),
        'Number_Children': np.random.randint(0, 5, size=num_rows),
        'Own_House_Flag': np.random.choice([True, False], size=num_rows),
        'Own_Car_Count': np.random.randint(0, 3, size=num_rows),
        'PersonId': range(1, num_rows + 1),  # Add a PersonId column as a row count
        'Client': np.random.choice(['A', 'B'], size=num_rows)  # Add a Client column with random values 'A' or 'B'
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # Binarize the data
    df_binarized = df.binarize(n_bins=4, thresh_infreq=0.01, name_infreq="-OTHER", one_hot=True)

    df_binarized.glimpse()
    ```

    ``` {python}
    df_correlated = df_binarized.correlate(target='Member_Status__Platinum')
    df_correlated.head(10)
    ```

    ``` {python}
    # Interactive
    df_correlated.plot_correlation_funnel(
        engine='plotly',
        height=600
    )
    ```

    ``` {python}
    # Static
    df_correlated.plot_correlation_funnel(
        engine ='plotnine',
        height = 900
    )
    ```

    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            "plot_correlation_funnel(): Object is not of class `pd.DataFrame`."
        )

    if engine == "plotly":
        fig = px.scatter(
            data,
            x="correlation",
            y="feature",
            hover_data={"correlation": ":.3f", "feature": True, "bin": True},
            range_x=limits,
            title="Correlation Funnel",
        )

        # Finalize the plotly plot

        fig.update_layout(
            title=title,
            xaxis_title=x_lab,
            yaxis_title=y_lab,
        )

        fig.update_xaxes(
            matches=None,
            showticklabels=True,
            visible=True,
        )

        fig.update_layout(margin=dict(l=10, r=10, t=40, b=40))
        fig.update_layout(
            template="plotly_white",
            font=dict(size=base_size),
            title_font=dict(size=base_size * 1.2),
            legend_title_font=dict(size=base_size * 0.8),
            legend_font=dict(size=base_size * 0.8),
        )
        fig.update_xaxes(tickfont=dict(size=base_size * 0.8))
        fig.update_yaxes(tickfont=dict(size=base_size * 0.8), autorange="reversed")
        fig.update_annotations(font_size=base_size * 0.8)
        fig.update_layout(
            autosize=True,
            width=width,
            height=height,
            showlegend=False,
        )
        fig.update_traces(
            marker=dict(color=point_color, opacity=alpha),
            selector=dict(mode="markers"),
            text=data["bin"],
            hoverlabel=dict(font_size=base_size * 0.8),
        )

        return fig

    else:
        plot_data = data.copy().reset_index(drop=True)
        feature_labels = list(plot_data["feature"].drop_duplicates())[::-1]
        feature_positions = {
            feature: index + 1 for index, feature in enumerate(feature_labels)
        }
        plot_data["_feature_y"] = (
            plot_data["feature"].map(feature_positions).astype(float)
        )
        plot_data["_label_side"] = np.where(
            plot_data["correlation"] < 0, "left", "right"
        )
        for column in ("_label_x", "_label_y", "_segment_x", "_segment_y"):
            plot_data[column] = np.nan
        x_span = limits[1] - limits[0]
        x_center = (limits[0] + limits[1]) / 2

        # Give each label its own vertical slot on its side of the feature row.
        # This keeps connectors attached to their points without relying on
        # adjustText, which can move labels across rows and obscure the arrows.
        for (_, side), group in plot_data.groupby(["_feature_y", "_label_side"]):
            group = group.sort_values("correlation")
            count = len(group)
            if count == 1:
                offsets = np.array([0.2])
            else:
                extent = min(0.36, 0.13 * (count - 1))
                offsets = np.linspace(-extent, extent, count)

            if side == "left":
                label_x = max(
                    limits[0] + 0.14 * x_span,
                    min(
                        x_center - 0.06 * x_span,
                        group["correlation"].min() - 0.03 * x_span,
                    ),
                )
                segment_x = label_x + 0.01 * x_span
            else:
                label_x = min(
                    limits[1] - 0.14 * x_span,
                    max(
                        x_center + 0.06 * x_span,
                        group["correlation"].max() + 0.03 * x_span,
                    ),
                )
                segment_x = label_x - 0.01 * x_span

            plot_data.loc[group.index, "_label_x"] = label_x
            plot_data.loc[group.index, "_label_y"] = group["_feature_y"] + offsets
            plot_data.loc[group.index, "_segment_x"] = segment_x
            plot_data.loc[group.index, "_segment_y"] = (
                group["_feature_y"] + offsets * 0.75
            )

        if height is None:
            height = max(500, 70 * len(feature_labels) + 80)

        p = (
            ggplot(plot_data, aes(x="correlation", y="_feature_y"))
            + geom_vline(xintercept=0, linetype="dashed", color="red")
            + geom_point(color=point_color, alpha=alpha)
            + geom_segment(
                aes(xend="_segment_x", yend="_segment_y"),
                color=point_color,
                alpha=alpha,
                size=0.25,
            )
            + geom_text(
                aes(x="_label_x", y="_label_y", label="bin"),
                data=plot_data[plot_data["_label_side"] == "left"],
                size=base_size * 0.8,
                color=point_color,
                ha="right",
            )
            + geom_text(
                aes(x="_label_x", y="_label_y", label="bin"),
                data=plot_data[plot_data["_label_side"] == "right"],
                size=base_size * 0.8,
                color=point_color,
                ha="left",
            )
            + labs(title=title, x=x_lab, y=y_lab)
            + xlim(limits[0], limits[1])
            + scale_y_continuous(
                breaks=list(range(1, len(feature_labels) + 1)),
                labels=feature_labels,
            )
        )
        p = p + theme_timetk(base_size=base_size, width=width, height=height)

        return p

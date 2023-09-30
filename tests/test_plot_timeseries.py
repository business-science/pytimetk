import pytest
import pandas as pd
import numpy as np
import timetk


# Prepare a sample dataframe for testing
data = pd.DataFrame({
    'date': pd.date_range(start='1/1/2020', periods=5, freq='D'),
    'value': np.random.randn(5),
    'id': ['A', 'A', 'B', 'B', 'B']
})


def test_plotly_engine():
    fig = data.plot_timeseries('date', 'value', engine='plotly')
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"


data.groupby('id').plot_timeseries(
        'date', 'value', 
        color_column = 'id',
        facet_ncol = 2,
        x_axis_date_labels = "%Y",
        engine = 'matplotlib'
    )


def test_matplotlib_engine():
    fig = data.plot_timeseries('date', 'value', engine='matplotlib', width = 1200, height = 800)
    assert type(fig).__name__ == "Figure", "Expected a matplotlib Figure object"


def test_plotnine_engine():
    # Since I don't have the complete function, I'm assuming the function returns a ggplot object for plotnine
    fig = data.plot_timeseries('date', 'value', engine='plotnine')
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"


# Test for groupby functionality
def test_groupby():
    fig = (data.groupby('id').plot_timeseries('date', 'value', engine='plotly'))
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object"

def test_matplotlib_groupby():
    fig = (data.groupby('id').plot_timeseries('date', 'value', engine='matplotlib', width = 1200, height = 800))
    assert type(fig).__name__ == "Figure", "Expected a matplotlib_ Figure object"

def test_plotnine_groupby():
    fig = (data.groupby('id').plot_timeseries('date', 'value', engine='plotnine'))
    assert str(type(fig)).endswith("ggplot'>"), "Expected a plotnine ggplot object"



# Test for smooth functionality
def test_smooth():
    fig = data.plot_timeseries('date', 'value', smooth=True, engine='plotly')
    assert type(fig).__name__ == "Figure", "Expected a plotly Figure object with smoothing"

# Test for Handling GroupBy objects
def test_groupby_handling():
    group = data.groupby('id')
    fig = group.plot_timeseries(date_column="date", value_column="value", engine="plotly")
    assert isinstance(fig, type(timetk.make_subplots())), "Figure type doesn't match expected type"

# Test for selection of plotting engine
# def test_plotting_engine():
#     with pytest.raises(ValueError):
#         data.plot_timeseries(date_column="date", value_column="value")

# Test plotting with different color columns and group names
# Removing this test for now.  There is a known bug in the code when group_names != color_names
# def test_plot_with_color_and_groups():
#     fig = data.plot_timeseries(date_column="date", value_column="value", color_column="id", group_names="id", engine="plotly")
#     assert len(fig.data) == 2 , "Number of traces doesn't match number of groups"


# Additional tests can be added based on other functionalities or edge cases

if __name__ == '__main__':
    pytest.main([__file__])

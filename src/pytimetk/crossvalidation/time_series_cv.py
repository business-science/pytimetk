import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from timebasedcv import TimeBasedSplit
from timebasedcv.splitstate import SplitState
from timebasedcv.utils._types import ModeType
from timebasedcv.utils._types import TensorLike
from timebasedcv.utils._types import NullableDatetime
from timebasedcv.utils._types import SeriesLike
from timebasedcv.utils._types import DateTimeLike

from typing import Generator
from typing import Literal
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import TypeVar
from typing import Optional, Union

from pytimetk.plot.theme import palette_timetk

TL = TypeVar("TL", bound=TensorLike)

class TimeSeriesCV(TimeBasedSplit):
    """
    `TimeSeriesCV` is a subclass of `TimeBasedSplit` with default mode set to 'backward' 
    and an optional `split_limit` to return the first `n` slices of time series cross-validation sets.
    
    Parameters
    ----------
    frequency: str
        The frequency (or time unit) of the time series. Must be one of "days", "seconds", "microseconds",
        "milliseconds", "minutes", "hours", "weeks". These are the only valid values for the `unit` argument of
        `timedelta` from python `datetime` standard library.
    train_size: int
        Defines the minimum number of time units required to be in the train set.
    forecast_horizon: int
        Specifies the number of time units to forecast.
    gap: int
        Sets the number of time units to skip between the end of the train set and the start of the forecast set.
    stride: int 
        How many time unit to move forward after each split. If `None` (or set to 0), the stride is equal to the
        `forecast_horizon` quantity.
    window: 
        The type of window to use, either "rolling" or "expanding".
    mode: ModeType, optional
        The mode to use for cross-validation. Default is 'backward'.
    split_limit: int, optional
        The maximum number of splits to return. If not provided, all splits are returned.
    
    Raises:
    ----------   
    ValueError:
        - If `frequency` is not one of "days", "seconds", "microseconds", "milliseconds", "minutes", "hours",
        "weeks".
        - If `window` is not one of "rolling" or "expanding".
        - If `mode` is not one of "forward" or "backward"
        - If `train_size`, `forecast_horizon`, `gap` or `stride` are not strictly positive.
        
    TypeError: 
        If `train_size`, `forecast_horizon`, `gap` or `stride` are not of type `int`.
    
    Examples:
    ---------
    
    ``` {python}
    import pandas as pd
    import numpy as np
    from pytimetk import TimeSeriesCV

    RNG = np.random.default_rng(seed=42)

    dates = pd.Series(pd.date_range("2023-01-01", "2023-01-31", freq="D"))
    size = len(dates)

    df = (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "time": pd.date_range(start, end, periods=_size, inclusive="left"),
                        "a": RNG.normal(size=_size - 1),
                        "b": RNG.normal(size=_size - 1),
                    }
                )
                for start, end, _size in zip(dates[:-1], dates[1:], RNG.integers(2, 24, size - 1))
            ]
        )
        .reset_index(drop=True)
        .assign(y=lambda t: t[["a", "b"]].sum(axis=1) + RNG.normal(size=t.shape[0]) / 25)
    )

    df.set_index("time", inplace=True)

    # Now let's run split the data with the provided `TimeSeriesCV` instance

    # Initialize TimeSeriesCV with desired parameters
    tscv = TimeSeriesCV(
        frequency="days",
        train_size=10,
        forecast_horizon=5,
        gap=0,
        stride=0,
        split_limit=3  # Limiting to 3 splits
    )

    X, y = df.loc[:, ["a", "b"]], df["y"]

    # Creates a split generator
    splits = tscv.split(X, y)

    for X_train, X_forecast, y_train, y_forecast in splits:
        print(X_train)
        print(X_forecast)
    ```

    ``` {python}
    # Also, you can use `glimpse()` to print summary information about the splits

    tscv.glimpse(y)
    ```

    ``` {python}
    # You can also plot the splits by calling `plot()` on the `TimeSeriesCV` instance with the `y` Pandas series

    tscv.plot(y)
    ```
    """

    def __init__(
        self,
        frequency: str,
        train_size: int,
        forecast_horizon: int,
        gap: int,
        stride: int = 0,
        window: str = "rolling",
        mode: ModeType = "backward",
        split_limit: int = None,
        **kwargs
    ):
        # Initialize the parent class
        super().__init__(
            frequency = frequency,
            train_size = train_size,
            forecast_horizon = forecast_horizon,
            gap = gap,
            stride = stride,
            window = window,
            mode=mode, 
            **kwargs
        )
        
        self.split_limit = split_limit

        # Assign the parameters to the class
        

    def split(
        self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike] = None,
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: bool = False,
    ) -> Generator[Union[Tuple[TL, ...], Tuple[Tuple[TL, ...], SplitState]], None, None]:
        """Returns a generator of split arrays with an optional `split_limit`.

        Arguments:
            *arrays: 
                The arrays to split. Must have the same length as `time_series`.
            time_series: 
                The time series used to create boolean masks for splits. If not provided, the method will try 
                to use the index of the first array (if it is a DataFrame or Series) as the time series.
            start_dt: 
                The start of the time period. If provided, it is used in place of `time_series.min()`.
            end_dt: 
                The end of the time period. If provided, it is used in place of `time_series.max()`.
            return_splitstate: 
                Whether to return the `SplitState` instance for each split.

        Yields:
            A generator of tuples of arrays containing the training and forecast data. If `split_limit` is set, 
            yields only up to `split_limit` splits.
        """
        # If time_series is not provided, attempt to extract it from the index of the first array
        if time_series is None:
            if isinstance(arrays[0], (pd.DataFrame, pd.Series)):
                time_series = arrays[0].index
            else:
                raise ValueError("time_series must be provided if arrays do not have a time-based index.")

        # Ensure the time_series is compatible with Narwhals
        if isinstance(time_series, pd.Index):
            time_series = pd.Series(time_series, index=time_series)

        split_generator = super().split(
            *arrays, time_series=time_series, start_dt=start_dt, end_dt=end_dt, return_splitstate=return_splitstate
        )

        if self.split_limit is not None:
            for i, split in enumerate(split_generator):
                if i >= self.split_limit:
                    break
                yield split
        else:
            yield from split_generator

    def glimpse(self, *arrays: TL, time_series: SeriesLike[DateTimeLike] = None):
        """Prints summary information about the splits, focusing on the first two arrays.

        Arguments:
            *arrays: 
                The arrays to split. Only the first two will be used for summary information.
            time_series: 
                The time series used for splitting. If not provided, the index of the first array is used. Default is None.
        """

        # Use only the first array for splitting and summary
        X = arrays[0]

        if time_series is None:
            if isinstance(X, (pd.DataFrame, pd.Series)):
                time_series = X.index
            else:
                raise ValueError("time_series must be provided if the first array does not have a time-based index.")

        # If the time_series is an index, convert it to a Series for easier handling
        if isinstance(time_series, pd.Index):
            time_series = pd.Series(time_series, index=time_series)

        # Iterate through the splits and print summary information
        for split_number, (X_train, X_forecast) in enumerate(self.split(X, time_series=time_series), start=1):
            # Get the start and end dates for the training and forecast periods
            train_start_date = time_series[X_train.index[0]]
            train_end_date = time_series[X_train.index[-1]]
            forecast_start_date = time_series[X_forecast.index[0]]
            forecast_end_date = time_series[X_forecast.index[-1]]

            # Print summary information
            print(f"Split Number: {split_number}")
            print(f"Train Shape: {X_train.shape}, Forecast Shape: {X_forecast.shape}")
            print(f"Train Period: {train_start_date} to {train_end_date}")
            print(f"Forecast Period: {forecast_start_date} to {forecast_end_date}\n")
            
    def plot(
        self, 
        y: pd.Series, 
        time_series: pd.Series = None,
        color_palette: Optional[Union[dict, list, str]] = None,
        facet_ncol: int = 1,
        facet_nrow: Optional[int] = None,
        facet_scales: str = "fixed",  # "fixed" ensures y-axis is globally consistent
        facet_dir: str = "h",
        line_color: str = "#2c3e50",
        line_size: float = 1.0,  # Line size for the plot
        line_alpha: float = 1.0,
        legend_show: bool = True,
        title: str = "Time Series Cross Validation Plot",
        x_lab: str = "",
        y_lab: str = "",
        color_lab: str = "Legend",
        x_axis_date_labels: str = "%b %Y",
        base_size: float = 11,
        width: Optional[int] = None,
        height: Optional[int] = None,
        engine: str = "plotly"
    ):
        """Plots the cross-validation sets using Plotly with a custom theme and formatting.

        Arguments:
            y: The Pandas series of target values to plot.
            time_series: The time series used for the x-axis. If not provided, the index of `y` will be used.
            (Additional arguments for customizing the theme and layout)
        """
        # Handle color palette
        if color_palette is None:
            color_palette = list(palette_timetk().values())  # Default colors
        elif isinstance(color_palette, str):
            color_palette = [color_palette]  # Convert single color to a list
        elif isinstance(color_palette, dict):
            # Convert dictionary to a list of colors for train and forecast
            color_palette = [
                color_palette.get("train", palette_timetk().values()[0]),
                color_palette.get("forecast", palette_timetk().values()[1]),
            ]
        elif not isinstance(color_palette, list):
            raise ValueError("Invalid `color_palette` parameter. It must be a dictionary, list, or string.")

        # Use the index of y if time_series is not provided
        if time_series is None:
            if isinstance(y, pd.Series):
                time_series = y.index
            else:
                raise ValueError("time_series must be provided if y does not have a time-based index.")

        # Ensure time_series is a Pandas Index
        if not isinstance(time_series, pd.Index):
            raise ValueError("time_series must be a Pandas Index or convertible to one.")

        # Determine the number of folds
        splits = list(self.split(y, time_series=time_series, return_splitstate=True))
        num_folds = len(splits)

        # Create subplots with shared y-axis
        fig = make_subplots(
            rows=num_folds, cols=1,
            shared_xaxes=True,
            shared_yaxes=True,  # Ensure y-axis is shared globally
            subplot_titles=[f"Fold {i+1}" for i in range(num_folds)]
        )

        # Enumerate through the splits and add traces to each subplot
        for fold, (train_forecast, split_state) in enumerate(splits, start=1):
            train, forecast = train_forecast

            ts = split_state.train_start
            te = split_state.train_end
            fs = split_state.forecast_start
            fe = split_state.forecast_end

            # Add train set trace as lines to the current subplot
            fig.add_trace(
                go.Scatter(
                    x=time_series[(time_series >= ts) & (time_series < te)],
                    y=train,
                    name=f"Train Fold {fold}",
                    mode="lines",  # Use lines for plotting
                    line=dict(
                        color=color_palette[0], 
                        width=line_size
                    ),
                    opacity=line_alpha
                ),
                row=fold, col=1
            )

            # Add forecast set trace as lines to the current subplot
            fig.add_trace(
                go.Scatter(
                    x=time_series[(time_series >= fs) & (time_series < fe)],
                    y=forecast,
                    name=f"Forecast Fold {fold}",
                    mode="lines",  # Use lines for plotting
                    line=dict(
                        color=color_palette[1], 
                        width=line_size
                    ),
                    opacity=line_alpha
                ),
                row=fold, col=1
            )

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_lab,
            yaxis_title=y_lab,
            legend_title_text=color_lab,
            xaxis=dict(tickformat=x_axis_date_labels),
            template="plotly_white",
            font=dict(size=base_size),
            title_font=dict(size=base_size * 1.2),
            legend_title_font=dict(size=base_size * 0.8),
            legend_font=dict(size=base_size * 0.8),
            margin=dict(l=10, r=10, t=40, b=40),
            autosize=True,
            width=width,
            height=height,
        )

        # Update axes
        fig.update_xaxes(tickfont=dict(size=base_size * 0.8), matches=None, showticklabels=True, visible=True)
        fig.update_yaxes(tickfont=dict(size=base_size * 0.8), range=[y.min(), y.max()])  # Keep y-axis range consistent

        # Update hover labels
        fig.update_traces(hoverlabel=dict(font_size=base_size * 0.8))

        # Show or hide legend
        if not legend_show:
            fig.update_layout(showlegend=False)

        return fig




# class TimeSeriesCV:
#     """Generates tuples of train_idx, test_idx pairs
#     Assumes the MultiIndex contains levels 'symbol' and 'date'
#     purges overlapping outcomes. Includes a shift for each test set."""

#     def __init__(
#         self,
#         n_splits=3,
#         train_period_length=126,
#         test_period_length=21,
#         lookahead=None,
#         shift_length=0,  # New parameter to specify the shift length
#         date_idx='date',
#         shuffle=False,
#         seed=None,
#     ):
#         self.n_splits = n_splits
#         self.lookahead = lookahead
#         self.test_length = test_period_length
#         self.train_length = train_period_length
#         self.shift_length = shift_length  # Store the shift length
#         self.shuffle = shuffle
#         self.seed = seed
#         self.date_idx = date_idx

#     def split(self, X, y=None, groups=None):
#         unique_dates = X.index.get_level_values(self.date_idx).unique()
#         days = sorted(unique_dates, reverse=True)
        
#         splits = []
#         for i in range(self.n_splits):
#             # Adjust the end index for the test set to include the shift for subsequent splits
#             test_end_idx = i * self.test_length + i * self.shift_length
#             test_start_idx = test_end_idx + self.test_length
#             train_end_idx = test_start_idx + self.lookahead - 1
#             train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            
#             if train_start_idx >= len(days):
#                 break  # Break if the start index goes beyond the available data
            
#             dates = X.reset_index()[[self.date_idx]]
#             train_idx = dates[(dates[self.date_idx] > days[min(train_start_idx, len(days)-1)])
#                               & (dates[self.date_idx] <= days[min(train_end_idx, len(days)-1)])].index
#             test_idx = dates[(dates[self.date_idx] > days[min(test_start_idx, len(days)-1)])
#                              & (dates[self.date_idx] <= days[min(test_end_idx, len(days)-1)])].index
            
#             if self.shuffle:
#                 if self.seed is not None:
#                     np.random.seed(self.seed)
                
#                 train_idx_list = list(train_idx)
#                 np.random.shuffle(train_idx_list)
#                 train_idx = np.array(train_idx_list)
#             else:
#                 train_idx = train_idx.to_numpy()
                
#             test_idx = test_idx.to_numpy()
            
#             splits.append((train_idx, test_idx))
        
#         return splits

#     def get_n_splits(self, X=None, y=None, groups=None):
#         """Adjusts the number of splits if there's not enough data for the desired configuration."""
#         return self.n_splits
    
    
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

    # Set index
    df.set_index("time", inplace=True)
    
    # Create an X dataframeand y series
    X, y = df.loc[:, ["a", "b"]], df["y"]

    # Initialize TimeSeriesCV with desired parameters
    tscv = TimeSeriesCV(
        frequency="days",
        train_size=10,
        forecast_horizon=5,
        gap=0,
        stride=0,
        split_limit=3  # Limiting to 3 splits
    )
    
    tscv
    ```

    ``` {python}
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
                The arrays to split. Only the first one will be used for summary information.
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
        bar_height: float = 0.3,  
        title: str = "Time Series Cross-Validation Plot",
        x_lab: str = "",
        y_lab: str = "Fold",
        x_axis_date_labels: str = None, 
        base_size: float = 11,
        width: Optional[int] = None,
        height: Optional[int] = None,
        engine: str = "plotly"
    ):
        """Plots the cross-validation folds on a single plot with folds on the y-axis and dates on the x-axis using filled Scatter traces.

        Parameters
        ----------
        y : pd.Series
            The target time series as a pandas Series.
        time_series : pd.Series
            The time series used for splitting. If not provided, the index of `y` is used. Default is None.
        color_palette : Optional[Union[dict, list, str]]
            The color palette to use for the train and forecast. If not provided, the default colors are used.
        bar_height : float
            The height of each bar in the plot. Default is 0.3.
        title : str
            The title of the plot. Default is "Time Series Cross-Validation Plot".
        x_lab : str
            The label for the x-axis. Default is "".
        y_lab : str
            The label for the y-axis. Default is "Fold".
        x_axis_date_labels : str
            The format of the date labels on the x-axis. Default is None.
        base_size : float
            The base font size for the plot. Default is 11.
        width : Optional[int]
            The width of the plot in pixels. Default is None.
        height : Optional[int]
            The height of the plot in pixels. Default is None.
        engine : str
            The plotting engine to use. Default is "plotly".
        """
        # Handle color palette
        if color_palette is None:
            color_palette = list(palette_timetk().values())  # Default colors
        elif isinstance(color_palette, str):
            color_palette = [color_palette]  # Convert single color to a list
        elif isinstance(color_palette, dict):
            # Convert dictionary to a list of colors for train and forecast
            color_palette = [
                color_palette.get("train", list(palette_timetk().values())[0]),
                color_palette.get("forecast", list(palette_timetk().values())[1]),
            ]
        elif not isinstance(color_palette, list):
            raise ValueError("Invalid `color_palette` parameter. It must be a dictionary, list, or string.")

        # Use the index of y if time_series is not provided
        if time_series is None:
            if isinstance(y, pd.Series):
                time_series = y.index
            else:
                raise ValueError("time_series must be provided if y does not have a time-based index.")

        # Ensure time_series is a DatetimeIndex
        if not isinstance(time_series, pd.DatetimeIndex):
            time_series = pd.to_datetime(time_series)

        # Determine the number of folds
        splits = list(self.split(y, time_series=time_series, return_splitstate=True))
        num_folds = len(splits)

        # Create figure
        fig = go.Figure()

        # Calculate the vertical positions for each fold
        fold_positions = list(range(1, num_folds + 1))

        # Enumerate through the splits and add filled Scatter traces
        for fold, (train_forecast, split_state) in enumerate(splits, start=1):
            train_indices, forecast_indices = train_forecast

            ts = split_state.train_start
            te = split_state.train_end
            fs = split_state.forecast_start
            fe = split_state.forecast_end

            # Convert indices to datetime if necessary
            if isinstance(ts, int):
                ts_date = time_series[ts]
            else:
                ts_date = pd.to_datetime(ts)
            if isinstance(te, int):
                te_date = time_series[te]
            else:
                te_date = pd.to_datetime(te)
            if isinstance(fs, int):
                fs_date = time_series[fs]
            else:
                fs_date = pd.to_datetime(fs)
            if isinstance(fe, int):
                fe_date = time_series[fe]
            else:
                fe_date = pd.to_datetime(fe)

            # Calculate y-axis positions
            y0 = fold - bar_height / 2
            y1 = fold + bar_height / 2

            # Create coordinates for the training period rectangle
            x_train = [ts_date, te_date, te_date, ts_date, ts_date]
            y_train = [y0, y0, y1, y1, y0]

            # Add Scatter trace for the training period
            fig.add_trace(go.Scatter(
                x=x_train,
                y=y_train,
                mode='lines',
                fill='toself',
                fillcolor=color_palette[0],
                line=dict(width=0),
                hoverinfo='text',
                hoverlabel=dict(font_size=base_size * 0.8),
                text=f"Fold {fold}<br>Train Period<br>{ts_date.date()} to {te_date.date()}",
                showlegend=False,
            ))

            # Create coordinates for the forecast period rectangle
            x_forecast = [fs_date, fe_date, fe_date, fs_date, fs_date]
            y_forecast = [y0, y0, y1, y1, y0]

            # Add Scatter trace for the forecast period
            fig.add_trace(go.Scatter(
                x=x_forecast,
                y=y_forecast,
                mode='lines',
                fill='toself',
                fillcolor=color_palette[1],
                line=dict(width=0),
                hoverinfo='text',
                hoverlabel=dict(font_size=base_size * 0.8),
                text=f"Fold {fold}<br>Forecast Period<br>{fs_date.date()} to {fe_date.date()}",
                showlegend=False,
            ))

            # Optionally, add text annotations for each fold
            train_midpoint = ts_date + (te_date - ts_date) / 2

            fig.add_trace(go.Scatter(
                x=[train_midpoint],
                y=[fold],
                text=[f"Fold {fold}"],
                mode="text",
                showlegend=False,
                textposition="middle center",
                hoverinfo='skip',
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_lab,
            yaxis_title=y_lab,
            xaxis=dict(
                type='date',
                tickformat=x_axis_date_labels,
                tickfont=dict(size=base_size * 0.8),
                showgrid=True,
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=fold_positions,
                ticktext=[f'Fold {i}' for i in fold_positions],
                range=[0.5, num_folds + 0.5],
                autorange='reversed',  # Place Fold 1 at the top
                tickfont=dict(size=base_size * 0.8),
                showgrid=False,
            ),
            template="plotly_white",
            font=dict(size=base_size),
            title_font=dict(size=base_size * 1.2),
            margin=dict(l=10, r=10, t=40, b=40),
            autosize=True,
            width=width,
            height=height,
            showlegend=False,
        )

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
    
    
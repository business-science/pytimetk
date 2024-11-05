import pandas as pd
import numpy as np

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

TL = TypeVar("TL", bound=TensorLike)

class TimeSeriesCV(TimeBasedSplit):
    """`TimeSeriesCV` is a subclass of `TimeBasedSplit` with default mode set to 'backward' 
    and an optional `slice_limit` to return the first `n` slices of time series cross-validation sets.
    
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
    slice_limit: int, optional
        The maximum number of slices to return. If not provided, all slices are returned.
    
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
    
    ```python
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
        gap=1,
        stride=0,
        slice_limit=3  # Limiting to 3 slices
    )

    X, y = df.loc[:, ["a", "b"]], df["y"]

    # If `time_series` is not provided, it will use the index of `X` or `y` if available
    for X_train, X_forecast, y_train, y_forecast in tscv.split(X, y):
        
        # Get the start and end dates for the training and forecast periods
        train_start_date = min(X_train.index)
        train_end_date = max(X_train.index)
        forecast_start_date = min(X_forecast.index)
        forecast_end_date = max(X_forecast.index)

        print(f"Train: {X_train.shape}, Forecast: {X_forecast.shape}")
        print(f"Train Period: {train_start_date} to {train_end_date}")
        print(f"Forecast Period: {forecast_start_date} to {forecast_end_date}\n")
    ```
    """

    def __init__(self, *args, mode: ModeType = "backward", slice_limit: int = None, **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.slice_limit = slice_limit

    def split(
        self,
        *arrays: TL,
        time_series: SeriesLike[DateTimeLike] = None,
        start_dt: NullableDatetime = None,
        end_dt: NullableDatetime = None,
        return_splitstate: bool = False,
    ) -> Generator[Union[Tuple[TL, ...], Tuple[Tuple[TL, ...], SplitState]], None, None]:
        """Returns a generator of split arrays with an optional `slice_limit`.

        Arguments:
            *arrays: The arrays to split. Must have the same length as `time_series`.
            time_series: The time series used to create boolean masks for splits. If not provided, the method will try 
                to use the index of the first array (if it is a DataFrame or Series) as the time series.
            start_dt: The start of the time period. If provided, it is used in place of `time_series.min()`.
            end_dt: The end of the time period. If provided, it is used in place of `time_series.max()`.
            return_splitstate: Whether to return the `SplitState` instance for each split.

        Yields:
            A generator of tuples of arrays containing the training and forecast data. If `slice_limit` is set, 
            yields only up to `slice_limit` splits.
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

        if self.slice_limit is not None:
            for i, split in enumerate(split_generator):
                if i >= self.slice_limit:
                    break
                yield split
        else:
            yield from split_generator



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
    
    
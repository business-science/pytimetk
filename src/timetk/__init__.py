
# *** Import everything to make timetk a standalone package ***

from .core.summarize_by_time import *
from .core.timeseries_signature import *
from .core.future_timeseries import *
from .core.lags import *
from .core.pad import *

from .datasets.get_datasets import *

from .utils.datetime_helpers import *
from .utils.pandas_helpers import *

# *** Needed for quartodoc build important functions ***
from .core.summarize_by_time import (
    summarize_by_time
)
from .core.timeseries_signature import (
    get_timeseries_signature, augment_timeseries_signature
)
from .core.future_timeseries import (
    make_future_timeseries, future_frame
)
from .core.lags import (
    augment_lags
)
from .core.pad import (
    pad_by_time
)
from .datasets.get_datasets import (
    load_dataset, get_available_datasets
)
from .utils.datetime_helpers import (
    floor_date, week_of_month, 
)


__version__ = '0.0.0.9000'
__author__ = 'Matt Dancho (Business Science)'

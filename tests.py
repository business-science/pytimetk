from datetime import date
from importlib import reload

import pandas as pd
import numpy as np



from timetk.data_wrangling import summarize_by_time

reload(timetk.data)
from timetk.data import load_dataset


help(load_dataset)

m4_daily = load_dataset("m4_daily", parse_dates = ['date'])

m4_daily.info()

summarize_by_time(
    data = m4_daily,
    date_column='date',
    value_column='value',
    groups='id',
    by = "MS",
    agg_func=np.sum,
    wide_format=False
)

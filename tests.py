from importlib import reload
import pandas as pd
import numpy as np
import timetk

reload(timetk)
from timetk import (
    load_dataset, 
    floor_date,
    aggregate_by_time
)


# Datasets

# help(load_dataset)

m4_daily = load_dataset("m4_daily", parse_dates = ['date'])

m4_daily = pd.DataFrame(m4_daily)

m4_daily.groupby('id').agg(np.sum)


# Floor Date

m4_daily['date'].dt.to_period('W')

floor_date(m4_daily['date'], "M")

m4_daily['date'].floor_date('M')


m4_daily.assign(month_year = lambda x: floor_date(x.date, unit = "M"))

m4_daily.assign(date_test = lambda x: x.date.floor_date(unit = "W"))

# Aggregate by Time

aggregate_by_time(
    data = m4_daily,
    date_column='date',
    value_column='value',
    groups='id',
    by = "MS",
    agg_func=lambda x: np.sum(x),
    wide_format=False
)

m4_daily \
    .aggregate_by_time(
        date_column='date',
        value_column='value',
        groups='id',
        by = "MS",
        agg_func=lambda x: np.sum(x),
        wide_format=False
    )

# - Mult-Index
# summary_df_1_columns_old = summary_df_1.columns

# summary_df_1_columns_old[0]

# idx = summary_df_1_columns_old

# col_0 = idx[0]

# "_".join(idx[0]).rstrip("_")

# "_".join(idx[3]).rstrip("_")

# summary_df_1.columns = summary_df_1_columns_old.map(lambda x: "_".join(x).rstrip("_"))
# summary_df_1


# Summarize -----









import pandas as pd
import pytimetk as tk


def test_timeseries_unit_frequency_table_polars_matches_pandas():
    pandas_table = tk.timeseries_unit_frequency_table()
    polars_table = tk.timeseries_unit_frequency_table(engine="polars")

    pd.testing.assert_frame_equal(polars_table, pandas_table)


def test_time_scale_template_polars_matches_pandas():
    pandas_table = tk.time_scale_template()
    polars_table = tk.time_scale_template(engine="polars")

    pd.testing.assert_frame_equal(polars_table, pandas_table)

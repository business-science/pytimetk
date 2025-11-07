import numpy as np
import pandas as pd

from pytimetk.utils.datetime_helpers import parse_human_duration, resolve_lag_sequence


def test_parse_human_duration_numeric_and_calendar_units():
    td = parse_human_duration("90 seconds")
    assert isinstance(td, pd.Timedelta)
    assert td == pd.Timedelta(seconds=90)

    td_weeks = parse_human_duration("2 weeks")
    assert isinstance(td_weeks, pd.Timedelta)
    assert td_weeks == pd.Timedelta(weeks=2)

    offset = parse_human_duration("3 months")
    assert isinstance(offset, pd.DateOffset)
    assert offset.kwds.get("months") == 3 and offset.n == 1


def test_resolve_lag_sequence_from_int_and_sequence():
    idx = pd.date_range("2021-01-01", periods=10, freq="D")
    result = resolve_lag_sequence(5, idx)
    assert np.array_equal(result, np.arange(0, 6))

    result_seq = resolve_lag_sequence([0, 2, 4, 7], idx)
    assert np.array_equal(result_seq, np.array([0, 2, 4, 7]))


def test_resolve_lag_sequence_from_phrase_clamps_to_length():
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    result = resolve_lag_sequence("10 days", idx)
    assert result.max() == len(idx) - 1

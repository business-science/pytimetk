import pandas as pd

from pytimetk.utils.selection import (
    contains,
    ends_with,
    matches,
    resolve_column_selection,
    starts_with,
)


def _sample_df():
    return pd.DataFrame(columns=["value", "VALUE_DELTA", "id", "category"])


def test_contains_selector_case_insensitive():
    df = _sample_df()
    selector = contains("value", case=False)
    result = selector(df.columns)
    assert result == ["value", "VALUE_DELTA"]


def test_prefix_suffix_and_regex_selectors():
    df = _sample_df()
    assert starts_with("val")(df.columns) == ["value"]
    assert ends_with("id")(df.columns) == ["id"]
    regex_matches = matches(r"^VAL", flags=0)(pd.Index(["VAL_A", "VAL_B"]))
    assert regex_matches == ["VAL_A", "VAL_B"]


def test_resolve_column_selection_with_groupby():
    df = pd.DataFrame({"grp": [1, 1], "value": [10, 20], "category": ["a", "b"]})
    grouped = df.groupby("grp")

    resolved = resolve_column_selection(df, ["value", contains("cat")])
    assert resolved == ["value", "category"]

    resolved_group = resolve_column_selection(grouped, "value")
    assert resolved_group == ["value"]

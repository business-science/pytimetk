# Sorting Review

## Scope & Method
- Focused on time-series utilities where chronological order is required for correctness (decompositions, rolling/expanding windows, regime detection, future frame generation, and plotting).
- Verified each candidate by reading the implementation and cross-checking for explicit `sort_values`/`sort` calls or helper usage (`sort_dataframe`).
- Highlighted any gaps where sorting is expected but not enforced.

## Summary
| Function / Area | Why sorting is required | Evidence | Status |
| --- | --- | --- | --- |
| `augment_regime_detection` | HMM log-returns must be chronological for stable training/prediction. | Uses `sort_dataframe` before the pandas path and `frame.sort(sort_keys)` in the polars bridge (`src/pytimetk/finance/regime_detection.py:316-352, 494-527`). | ✅ Enforced |
| `stl_diagnostics` (used by `plot_stl_diagnostics`) | STL decomposition assumes ordered timestamps. | `_stl_diagnostics_single` sorts and resets index before fitting (`src/pytimetk/core/stl_diagnostics.py:91-149`); `plot_stl_diagnostics` delegates to it (`src/pytimetk/plot/plot_stl_diagnostics.py:226-239`). | ✅ Enforced |
| `augment_rolling` | Rolling windows require sequential order within each group. | Pandas path calls `sort_dataframe` up front, cudf/polars paths sort by group/date before rolling (`src/pytimetk/feature_engineering/rolling.py:268-306, 458-518`). | ✅ Enforced |
| `augment_rolling_apply` | Custom rolling apply expects deterministic window ordering. | Grouped frames sorted by group/date; concatenated result re-sorted to the original index (`src/pytimetk/feature_engineering/rolling_apply.py:282-349, 351-404`). | ✅ Enforced |
| `augment_expanding` | Expanding stats are cumulative over ordered observations. | Data sorted per group/date for pandas, cudf, and polars implementations (`src/pytimetk/feature_engineering/expanding.py:404-438, 478-487`). | ✅ Enforced |
| Shift-style helpers (`augment_diffs`, `augment_lags`, `augment_leads`) | Forward/backward differences/leads rely on sequential rows. | Each helper sorts by group/date, computes diffs/leads, then restores the original ordering (`src/pytimetk/feature_engineering/diffs.py:324-368`; similar patterns exist in `lags.py` and `leads.py`). | ✅ Enforced |
| `future_frame` | Frequency inference and extrapolation must examine ordered dates. | Both ungrouped and grouped branches sort timestamps before calling `get_frequency` and generating future rows (`src/pytimetk/core/future.py:386-440`). | ✅ Enforced |
| `plot_timeseries` (Plotly backend) | Line traces should traverse chronological x-values to avoid self-crossing paths. | Data is grouped with `sort=False` and fed to `go.Scatter` without sorting (see `src/pytimetk/plot/plot_timeseries.py:795-1076`). | ⚠️ Needs sorting |

## Detailed Notes

### `augment_regime_detection`
- Sorting occurs immediately after converting to the requested engine via `sort_dataframe`, ensuring per-group chronological order before computing log returns and fitting HMMs (`src/pytimetk/finance/regime_detection.py:316-333`).
- The polars path mirrors this by adding a temporary row id, sorting by `group + date`, running the pandas implementation, then restoring the original order (`src/pytimetk/finance/regime_detection.py:490-527`). No additional action needed.

### `stl_diagnostics` & `plot_stl_diagnostics`
- `_stl_diagnostics_single` sorts the frame, coerces numeric types, and only then infers periods and fits STL, so each group is chronologically consistent (`src/pytimetk/core/stl_diagnostics.py:91-149`).
- `plot_stl_diagnostics` trusts the sorted diagnostic output; no extra sorting is necessary so long as upstream behavior is preserved (`src/pytimetk/plot/plot_stl_diagnostics.py:226-239`).

### Rolling / Expanding Helpers
- `augment_rolling` and `_augment_rolling_cudf_dataframe`/polars equivalents sort data by `group + date` and, when necessary, use `row_id_column` to restore original ordering so downstream joins remain aligned (`src/pytimetk/feature_engineering/rolling.py:268-304, 458-518`).
- `augment_rolling_apply` sorts per group (or globally when ungrouped) before invoking custom window logic and reindexes to the original order (`src/pytimetk/feature_engineering/rolling_apply.py:282-349`).
- `augment_expanding` follows the same pattern for expanding statistics, guaranteeing deterministic cumulative calculations (`src/pytimetk/feature_engineering/expanding.py:404-438, 478-487`).

### Shift-Based Feature Generators
- Difference/lead/lag utilities explicitly sort by `group + date` before applying `diff`, `pct_change`, or `shift`, then resort by a synthetic row id so the returned frame matches the caller’s original layout (`src/pytimetk/feature_engineering/diffs.py:324-368`; identical scaffolding exists in `leads.py` and `lags.py`).

### `future_frame`
- Frequency detection (`get_frequency`) and the generation of future timestamps operate on `df[date_column].sort_values()` to avoid deriving cadence from unsorted inputs. Grouped inputs also infer frequency from a sorted reference group (`src/pytimetk/core/future.py:386-440`). This safeguards against irregular spacing artifacts.

### Shared Helper: `sort_dataframe`
- Many of the functions above rely on `sort_dataframe` to standardize ordering while preserving grouping metadata (`src/pytimetk/utils/pandas_helpers.py:130-194`). Keeping this helper authoritative simplifies auditing future call sites.

### Gap: `plot_timeseries`
- Neither the dropdown nor the faceted Plotly paths enforce sorting. Groups are created with `sort=False`, and `go.Scatter` traces consume whatever order the caller supplied (`src/pytimetk/plot/plot_timeseries.py:795-1076`). This can produce zig-zagging lines or incorrect annotations when the input frame is not pre-sorted.
- **Suggested fix:** before plotting, apply `group.sort_values(date_column)` (or use the existing `sort_dataframe` helper) so every trace sees monotonically increasing dates. Re-sort back to the original index if the function must return data to callers.

## Recommended Next Steps
1. Add an explicit sort (per group when grouping columns are provided, otherwise global) to `plot_timeseries` before constructing Plotly traces, along with a regression test that feeds unsorted data.
2. Continue routing new time-series utilities through `sort_dataframe` to keep the sorting contract centralized and easier to audit.

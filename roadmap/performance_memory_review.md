# pytimetk Performance & Memory Opportunities

- **Context**: Assessment focused on redundant DataFrame copies, engine conversion churn, and spots where a Polars-native path or `clone()` could shrink memory pressure without affecting behaviour.
- **Priority legend**: 🔴 high impact, 🟡 medium impact, 🟢 quick win.

## 🔴 Trim redundant pandas copies

- `src/pytimetk/feature_engineering/ewm.py:331` and peers (`expanding_apply.py:193`, `rolling_apply.py:220`, `feature_engineering/fourier.py:171`, `feature_engineering/timeseries_signature.py:153`, etc.) copy the full frame before sorting and augmenting. Because `sort_values` and groupby already materialise new blocks, these copies double memory. Consider:
  - switch to `data.sort_values(..., inplace=True)` after a shallow `copy(deep=False)` only when the caller passes a plain `DataFrame`;
  - or enable pandas Copy-on-Write globally per execution (e.g. in `convert_to_engine`) so downstream mutations automatically detach without explicit `.copy()`.
- `src/pytimetk/utils/pandas_helpers.py:176` in `sort_dataframe` copies prior to sorting, and the result feeds most feature engineering helpers; the extra copy cascades. Returning the sorted frame without an upfront copy and adding a shallow copy only when `keep_grouped_df` is true keeps semantics intact.
- `src/pytimetk/utils/memory_helpers.py:58` defensively clones input before downcasting. Expose an `inplace` flag (default `False`) so power users can avoid the additional allocation when they manage isolation themselves.

## 🔴 Streamline engine conversion plumbing

- `src/pytimetk/utils/dataframe_ops.py:240` inserts a synthetic row id column in pandas, then converts to Polars. Replacing this with `pl.from_pandas(data, include_index=True)` followed by `with_row_count` avoids touching the pandas frame and cuts a full-column copy.
- `src/pytimetk/utils/dataframe_ops.py:331` mirrors the same pattern for cudf; prefer generating the row index inside cudf using `cudf.Series.arange` or `DataFrame.insert` after conversion.
- `src/pytimetk/utils/dataframe_ops.py:410` sorts on the temporary row id to restore order. Sorting materialises another full frame; using `take`/`reindex` with the recorded positional indices avoids the sort and associated allocations.
- `src/pytimetk/utils/dataframe_ops.py:488` repeatedly converts Polars back to pandas when the original object was already pandas. Cache the pandas view inside `FrameConversion` (`pandas_cache: Optional[pd.DataFrame]`) to prevent redundant conversions on fallback paths.

## 🔴 Close Polars fallbacks

- `src/pytimetk/feature_engineering/ewm.py:179` forces Polars users through pandas, incurring two conversions and temporary copies per call. Implementing a Polars-native EWM via `pl.DataFrame.sort`, `group_by(...).agg(pl.col(col).ewm_mean(alpha=...))` removes the round-trips and unlocks zero-copy chaining.
- `src/pytimetk/core/pad.py:207` and `src/pytimetk/core/apply_by_time.py:214` both down-convert to pandas. Polars exposes `DataFrame.upsample` and resampling via `group_by_dynamic`; wiring those in would avoid serialisation and let large frames stay in Arrow buffers.
- Where we still need pandas fallbacks, call `prepared.clone()` before mutating inside the Polars branch so the original frame can stay cached without triggering copy-on-write churn. The clone cost is lower than repeated pandas conversions when users chain multiple helpers.

## 🟡 Feature store pipeline

- `_dataframe_fingerprint` in `src/pytimetk/feature_store/store.py:724` always converts to pandas, even when the source is already Polars. Leveraging `pl.DataFrame.hash_rows` (Polars ≥0.20) or Arrow `table = frame.to_arrow()` for hashing keeps data in its native buffer and avoids the duplicate pandas materialisation.
- `_coerce_return_frame` in the same module converts back to pandas and reassigns indexes. For large artifacts, offer a `return_engine="arrow"` option that hands off a zero-copy `pyarrow.Table`, giving callers full control over pandas vs Polars conversion.

## 🟡 Memory helper enhancements

- `reduce_memory_usage` currently casts floats to `float32` unconditionally when safe; exposing a `float_precision=("float32","float64")` argument and documenting the accuracy trade-off would help users tune allocations.
- Extend `reduce_memory_usage` to accept Polars frames by pattern matching numeric types and calling `.with_columns(pl.col(...).cast(...))`. This keeps type coercion in the Arrow domain and prevents `to_pandas` detours.

## 🟢 Observability & testing

- Add microbenchmarks (e.g. `zzz_local/benchmarks/augment_ewm_bench.py`) to measure time and peak RSS for common helper combinations under pandas vs Polars vs cudf. Guard PRs by comparing benchmark deltas in CI.
- Surface optional debug logging (env flag) to warn when a Polars input is forced through pandas; this gives end users actionable visibility and validates future native implementations.


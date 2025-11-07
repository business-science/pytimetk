# Tidy Selector Integration Roadmap

## Background
- pytimetk recently introduced reusable helpers for tidy-style column selection (`contains`, `starts_with`, `ends_with`, `matches`, `resolve_column_selection`) and duration parsing (`parse_human_duration`, `resolve_lag_sequence`).
- These utilities now power `plot_acf_diagnostics` and are exposed at the package root, providing a foundation for more expressive APIs that work across pandas and polars.
- The goal is to extend tidy selectors to other plotting and diagnostic functions where column selection patterns are common, while maintaining backwards compatibility.

## Guiding Principles
- **Opt-in flexibility:** Keep existing string/list arguments working; tidy selectors are additive enhancements.
- **Consistent resolution:** Centralize selector handling through `resolve_column_selection` to guarantee identical behaviour across pandas and polars inputs.
- **Documentation parity:** Each function supporting selectors should document examples for pandas and polars.
- **Incremental rollout:** Prioritize high-usage APIs and plotting helpers before lower-level utilities.

## Phase 1 — Plotting Enhancements
1. ✅ `plot_acf_diagnostics` accepts tidy selectors for `value_column` / `ccf_columns` and handles polars inputs.
2. ✅ `plot_timeseries` already supports tidy selectors; add docs/examples to highlight the capability.
3. In progress: extend tidy selectors beyond the visualization layer:
   - [ ] Core resampling helpers: `summarize_by_time`, `apply_by_time`, `pad_by_time`, `future_frame`.
   - ✅ Feature engineering ops (`augment_lags`, `augment_leads`, `augment_diffs`) now resolve selectors + duration specs.
   - ✅ Finance indicators (`augment_rsi`, `augment_macd`, `augment_atr`, `augment_adx`) now accept selectors for price columns (pandas + polars).
4. Update usage docs with selector-driven examples (pandas + polars) for each upgraded API.

## Phase 2 — Diagnostic & Feature APIs
1. `seasonal_diagnostics` / `stl_diagnostics`: allow selectors for value columns and optional facets.
2. Explore selector support in frequently used feature engineering utilities (e.g., `augment_lags`, `augment_pct_change`, `augment_rolling`).
3. Verify that `ts_features`, `ts_summary`, and other analytics helpers can optionally accept selectors for column subsets.

## Phase 3 — Namespace Consistency & Tests
1. Audit `pl_df.tk` namespace to ensure selector-enabled functions are available to polars users.
2. Expand unit tests to cover selector usage across pandas/polars variants and grouped data.
3. Add Quarto documentation snippets demonstrating tidy selection for each upgraded API.

## Phase 4 — Communication & Adoption
1. Document selector patterns in a dedicated “Tidy Helpers” reference section (✅ now listed in `_quarto.yml`).
2. Blog or changelog entry summarizing the selector rollout, highlighting cross-backend support.
3. Collect user feedback to identify additional APIs that would benefit from selectors.

## Open Questions
- Should selectors accept negation (e.g., `exclude` helpers) or will a positive-only API suffice?
- Do we need lazy evaluation for selectors when used with large polars LazyFrames?
- How far should selector support extend into finance/feature store modules versus focusing on core plotting/diagnostics?

## Next Steps
1. Prioritize selector integration for `plot_seasonal_diagnostics` and `plot_stl_diagnostics`.
2. Draft documentation updates for selector-enabled plotting functions.
3. Prepare additional tests (pandas + polars) for each API as selectors ship.

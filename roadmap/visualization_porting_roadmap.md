# pytimetk Visualization Porting Roadmap

## Background
- The R package `timetk` ships a suite of visualization helpers that streamline exploratory time series analysis.  
- The current Python port (`pytimetk`) lacks these plotters:  
  - `plot_time_series_boxplot()`  
  - `plot_time_series_regression()`  
  - `plot_acf_diagnostics()`  
  - `plot_seasonal_diagnostics()`  
  - `plot_stl_diagnostics()`  
  - `plot_time_series_cv_plan()`  
- The goal for `pytimetk` is to deliver feature parity for these functions while keeping APIs pythonic and interoperable with pandas and plotly.

## Guiding Principles
- **User experience parity:** Mirror the intent and ergonomics of the R functions but leverage Python-native defaults (pandas indexing, plotly express, etc.).
- **Consistent API surface:** Align signatures with existing `pytimetk` helpers (data frames in, tidy column selection, optional interactive output).
- **Composable outputs:** Favor returning plotly figure objects so they embed cleanly in notebooks, dashboards, or saved as static artifacts.
- **Testable logic:** Separate data transformation utilities from plotting layers to unit test computations independently of the visual layer.

## Phase 1 — Discovery & Design
1. Audit `timetk` R implementations to catalogue required inputs, optional parameters, and typical workflows.
2. Document dependencies (e.g., plotly, statsmodels) and confirm current `pyproject.toml` coverage.
3. Draft Pythonic function signatures and shared helper utilities (e.g., column validation, seasonal feature extraction).
4. Create design spikes for:
   - Interactive vs. static rendering strategy.
   - Handling grouped time series (facet wraps, subplot specs).
   - Localization of date labels and axis formatting.

## Phase 2 — Core Utility Layer
1. Implement reusable data prep utilities:
   - ACF/PACF/CCF computation wrapper around `statsmodels`.
   - STL decomposition helper using `statsmodels` or `prophet`-compatible pipelines.
   - Resampling + cross-validation frame generator mirroring `timetk::time_series_cv`.
2. Add unit tests for each helper to ensure parity with R outputs (regression tests using known datasets).
3. Update documentation on shared utilities (docstrings + `docs/` references).

## Phase 3 — Plotting Functions
1. Implement visualization functions in the following order (building on utilities):
   - [x] `plot_acf_diagnostics()`
   - [ ] `plot_stl_diagnostics()`
   - [x] `plot_seasonal_diagnostics()`
   - [x] `plot_time_series_boxplot()`
   - [ ] `plot_time_series_regression()`
   - [ ] `plot_time_series_cv_plan()`
2. Provide sensible defaults (palette, titles, hover labels) and expose optional kwargs for advanced styling.
3. Ensure each function returns a plotly `Figure` and optionally accepts `show=False` to defer rendering.
4. Add smoke tests that validate figure generation and key trace counts for canonical datasets.

## Phase 4 — Documentation & Examples
1. Write usage guides in `docs/` demonstrating interactive and static workflows.
2. Port representative notebooks from R `timetk` (where licensing allows) to Jupyter examples.
3. Update README feature matrix and changelog once each function stabilizes.
4. Showcase before/after visuals to communicate parity with the R package.

## Phase 5 — Release & Feedback
1. Publish beta builds (internal or to a feature branch) for feedback from core users.
2. Gather adoption metrics and bug reports; prioritize fixes before public release.
3. Prepare announcement (blog, release notes) highlighting visualization parity.

## Open Questions
- Should we offer Matplotlib exports alongside Plotly for static publication workflows?
- Do we need plugin hooks for alternative decomposition engines (e.g., `statsforecast`)?
- How should we handle large grouped datasets with respect to performance and memory?
- What level of theming customization do users expect relative to the R ggplot2-based output?

## Next Steps
1. Approve roadmap scope and sequencing.
2. Begin Phase 1 by reviewing the R source implementations and cataloging parameter expectations.

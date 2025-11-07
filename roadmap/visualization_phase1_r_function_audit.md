# Phase 1 Audit — timetk R Visualization Functions

This note catalogs the R implementations that pytimetk needs to port. Each section highlights public APIs, supporting helpers, and behavioral details that Python equivalents must replicate.

## Common Patterns
- All plotters accept data frames/tibbles and rely on tidy evaluation (`rlang::enquo`, `UseMethod`) to dispatch between `data.frame` and `grouped_df` methods.
- Plot construction is primarily `ggplot2` with optional `plotly::ggplotly(...)` conversion; trelliscope integration is available where faceting across large group sets is expected.
- Fig styling uses `theme_tq()` and `scale_color_tq()` from tidyquant.
- Data prep responsibilities are delegated to `tk_*` helpers (acf/seasonal/stl diagnostics, CV plan expansion) so plotting layers stay thin.
- Grouped methods ungroup, rewire facets, and forward to the corresponding `data.frame` implementation.

## `plot_time_series_boxplot()` (`R/plot-time_series_boxplot.R`)
- **Purpose:** Box-and-whisker visualization of values bucketed into a user-specified time aggregation (`.period`) with optional smoother overlay.
- **Key args:** `.period` (required; time window), `.color_var`, `.facet_vars`, `.smooth*` options, `.trelliscope`, `.interactive`, `.plotly_slider`.
- **Workflow:** mutate selected value/color expressions, derive `.box_group` via `lubridate::floor_date`, optionally collapse facets, smooth via `auto_smooth()` wrapper around `smooth_vec()` (`plot-time_series.R`), render with `geom_boxplot` + optional `geom_line` for smoother.
- **Dependencies:** `auto_smooth()`, `smooth_vec()`, `tk_get_trend()`, tidyverse, `trelliscopejs`, `plotly`.
- **Group support:** `grouped_df` method maps grouping columns into `.facet_vars` before delegating.

## `plot_time_series_regression()` (`R/plot-time_series_regression.R`)
- **Purpose:** Fit `stats::lm()` on a user-specified formula, then overlay fitted values alongside the original series using `plot_time_series()`.
- **Key args:** `.formula` (required), `.show_summary`, `...` forwarded to `plot_time_series()`.
- **Workflow:** run `lm`, optionally print `summary()`, predict, align fitted values with original order, pivot longer to two-column series (`value` vs. `fitted`), plot via `plot_time_series()` with `color_var = name`.
- **Dependencies:** `stats::lm`, `plot_time_series`, tidyverse reshape utilities.
- **Group support:** nests by group, fits models per group, optionally prints summaries, then unnests and reuses the base function.

## `plot_acf_diagnostics()` (`R/plot-acf_diagnostics.R`)
- **Purpose:** Plot ACF, PACF, and optional CCF traces for a value column (and lagged predictors).
- **Key args:** `.value`, `.ccf_vars`, `.lags` (accepts numeric or time-based spec), `.show_ccf_vars_only`, styling toggles for line/point/white-noise bands, `.interactive`.
- **Workflow:** call `tk_acf_diagnostics()` to compute autocorrelation metrics plus white-noise bounds, reshape with `pivot_longer`, plot with `geom_line`/`geom_point`, optionally add white-noise limit lines, convert to plotly if requested.
- **Dependencies:** `tk_acf_diagnostics()`, `scale_color_tq`, `theme_tq`, `plotly`.
- **Group support:** grouped variant consolidates group identifiers into a combined factor, lays out facets with `facet_grid`.

## `plot_seasonal_diagnostics()` (`R/plot-seasonal_diagnostics.R`)
- **Purpose:** Visualize value distributions across calendar-based seasonal features (hour, weekday, month, etc.).
- **Key args:** `.feature_set` ("auto" or explicit selections), `.geom` ("boxplot"/"violin"), `.facet_vars`, `.interactive`.
- **Workflow:** determine facets from tidyselect expressions, auto-select feature set via `get_seasonal_auto_features()`, compute seasonal stats via `tk_seasonal_diagnostics()`, reshape to `.group` vs `.group_value`, render with boxplot/violin, facet by seasonal group × optional facets, wrap with plotly when requested.
- **Dependencies:** `tk_seasonal_diagnostics()`, `get_seasonal_auto_features()`, tidyverse, `plotly`.
- **Group support:** grouped method promotes group vars into `.facet_vars` before delegating.

## `plot_stl_diagnostics()` (`R/plot-stl_diagnostics.R`)
- **Purpose:** Display STL decomposition components produced by `tk_stl_diagnostics()` for a series.
- **Key args:** `.feature_set` (subset of observed/season/trend/remainder/seasadj), `.frequency`, `.trend`, `.facet_vars`, `.facet_scales`.
- **Workflow:** optional facet collapsing like seasonal diagnostics, invoke `tk_stl_diagnostics()` with frequency/trend config, pivot the requested components to long form, draw line charts per component, facet wrap on component × optional groups, apply plotly if interactive.
- **Dependencies:** `tk_stl_diagnostics()`, `stats::stl` (inside helper), tidyverse, `plotly`.
- **Group support:** grouped method forwards group names into facet pipeline.

## `plot_time_series_cv_plan()` (`R/rsample-plot_time_series_cv_plan.R`)
- **Purpose:** Visualize rolling origin or time-series CV splits (analysis vs assessment windows) emitted by `time_series_cv()` or `rolling_origin()`.
- **Key args:** `.data` can be `rset` or pre-expanded tibble via `tk_time_series_cv_plan()`, `.smooth`, `.title`, plus `plot_time_series` passthrough arguments (`...`).
- **Workflow:** dispatch by object class, expand CV plan (either inside helper `plot_ts_cv_rset()` or `plot_ts_cv_dataframe()`), group by `.id`, call `plot_time_series()` using `.key` to color analysis vs assessment segments, optionally smooth.
- **Dependencies:** `tk_time_series_cv_plan()`, `plot_time_series()`, `rsample` structures, tidyverse.
- **Group support:** inherent in CV plan; plotting happens per split id.

## Implications for pytimetk
- **Helper parity:** Need Python equivalents for `tk_acf_diagnostics`, `tk_seasonal_diagnostics`, `tk_stl_diagnostics`, `tk_time_series_cv_plan`, and `auto_smooth/smooth_vec` behavior (or dedicated utilities in the Python codebase).
- **Plot engine:** Plotly should be the primary output; decide on direct plotly construction vs. building via seaborn/matplotlib then converting.
- **Tidyselect analogs:** Python API must provide ergonomic column selection; consistent with existing pytimetk patterns (likely string names or callables).
- **Group/facet strategy:** Many functions rely on collapsing groups into facets; need a reusable helper to transform grouped pandas DataFrames into stacked long-form with metadata suitable for facetting in Plotly (e.g., facet_col/facet_row or subplot specs).
- **Time-based parameters:** `.period`, `.lags`, `.frequency`, `.trend` accept natural language durations; Python side needs parsing (possibly via `pandas.Timedelta`, `dateutil`, or pendulum) and bridging to underlying computations.

## Python Baseline Snapshot
- `plot_timeseries` already handles smoothing (LOWESS), multiple engines, groupby inputs, and legend/theme utilities; we can borrow its internal helpers for boxplot/regression visual styling.
- Date utilities (`floor_date`, `ceil_date`, `freq_to_timedelta`, `get_frequency`, `get_seasonal_frequency`, `get_trend_frequency`) mirror many R helpers; they just need thin wrappers for the new diagnostics.
- `TimeSeriesCV.plot()` offers a Plotly visualization of resample windows; we can repurpose its layout logic for `plot_time_series_cv_plan`.
- Feature engineering (`augment_timeseries_signature`, `get_timeseries_signature`) provides the raw seasonal features required by `tk_seasonal_diagnostics`.
- `anomalize` (STL + decomposition) and statsmodels dependencies ensure STL calculations are already feasible on the Python side.

## Helper Gaps to Close
- **Diagnostics data prep**
  - ✅ Prototype `acf_diagnostics` helper with Plotly visual (`plot_acf_diagnostics`) now supports grouped faceting, configurable styling, tidy selectors for column arguments, and an optional dropdown UX for switching groups.
  - ✅ Implemented `seasonal_diagnostics` helper using `augment_timeseries_signature` with auto feature selection heuristics.
  - ✅ Implemented `stl_diagnostics` helper leveraging statsmodels STL with auto/override frequency & trend parsing.
  - ✅ Introduced `plot_seasonal_diagnostics` (Plotly) built atop the new helper with tidy selector support, dropdown faceting, and polars compatibility.
  - Build `time_series_cv_plan` helper that normalizes inputs from `TimeSeriesCV`, raw resample DataFrames, or iterables of splits into a tidy plotting frame.
- **Parsing & selection**
  - ✅ Human-friendly duration parsing (`parse_human_duration`, `resolve_lag_sequence`) now supports specs like `"30 days"` or `"3 months"`.
  - ✅ Flexible column selectors (`utils.selection`) add helpers such as `contains()`, `starts_with()`, and `resolve_column_selection()` to mimic tidyselect ergonomics.
- **Plot assembly**
  - Create reusable Plotly subplot/facet utility to arrange traces (boxplots, line diagnostics, violin/box combos) across groups consistently.
  - Decide on smoother abstraction (`auto_smooth`) so boxplot/regression charts can share logic with `plot_timeseries`.

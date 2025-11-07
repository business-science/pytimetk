## Datetime Alias & Duration Migration Plan

### Objective
Unify how pytimetk accepts and normalizes frequency specifications so users can provide:
1. Human-friendly durations (`"3 weeks"`, `"2 months"`, etc.).
2. Current pandas aliases (`"ME"`, `"QE"`, `"YE"`, `"SME"`, `"CBME"`, ...).
3. Legacy pandas aliases (`"M"`, `"Q"`, `"Y"`, `"SM"`, `"BM"`, `"BQ"`, `"BY"`, `"CBM"`) without deprecation warnings.

### Proposed Approach
1. **Alias Map:** Create a single dictionary that translates deprecated aliases to the new ones (per pandas 2.2 release notes). Apply it inside the common normalization helper used by `pad_by_time`, `future_frame`, rolling/expanding utilities, etc.
2. **Normalization Flow:**  
   - First detect human-friendly durations via `parse_human_duration`.  
   - If not a duration, uppercase the alias, remap via the dictionary, then call `pd.tseries.frequencies.to_offset`.
3. **Selective Warning:** Optionally emit a `FutureWarning` the first time a legacy alias is normalized so users are nudged toward the new forms.
4. **Docs & Docstrings:** Update API docs to mention the accepted formats and showcase “old vs new” examples.
5. **Tests:** Add parametrized tests covering:
   - Legacy aliases (`"M"`, `"Q"`, …) producing the same result as their new counterparts.
   - Human durations still passing through unchanged.
   - Mixed-case inputs (e.g., `"q"` → `"QE"`).

### Execution Steps
1. Implement the alias dictionary + normalization helper update.
2. Update affected modules to use the shared helper (if any bypass it today).
3. Add regression tests (core + feature-engineering suites) to confirm both formats work.
4. Refresh docs (reference + production guides) to highlight the supported formats.

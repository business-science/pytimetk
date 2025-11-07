## Ray Migration Notes

### Completed
- Converted `future_frame` (pandas backend) to dispatch grouped work via Ray when `threads != 1`.
- Converted `ts_features` grouped execution to Ray, eliminating the previous ProcessPool dependency.
- Added reusable helpers in `pytimetk.utils.ray_helpers` for lazy initialization, argument fan-out, and tqdm-friendly progress.
- Migrated `parallel_apply` plus all pandas-based rolling/expanding helpers (`augment_expanding`, `augment_expanding_apply`, `augment_rolling`, `augment_rolling_apply`) to the shared Ray helper with sequential fallbacks when Ray is absent.

### In Progress
- Update README/docs to highlight Ray as a core dependency, explain its role in threaded helpers, and describe how to disable parallel paths (e.g., set `threads=1`) when needed.
- Monitor remaining modules (pad/future variants, CV helpers) for additional opportunities once the current changes stabilize.

### Next Steps
1. Add troubleshooting guidance for Ray-related issues (worker startup, resource constraints) so users can self-diagnose.
2. Evaluate additional candidates (`ts_features` variants that operate per-series, CV helpers) once the current stack is verified in real workloads.
3. Consider profiling GPU/cudf paths to confirm Ray isn’t needed there or whether similar helpers would benefit those execution modes.

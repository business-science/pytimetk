<div align="center">
<img src="docs/logo-timetk.png" width="30%"/>
</div>

<div align="center">
  <a href="https://github.com/business-science/pytimetk/actions">
  <img alt="Github Actions" src="https://img.shields.io/github/actions/workflow/status/business-science/pytimetk/timetk-checks.yaml?style=for-the-badge"/>
  </a>
  <a href="https://pypi.python.org/pypi/pytimetk">
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/pytimetk.svg?style=for-the-badge"/>
  </a>
  <a href="https://github.com/business-science/pytimetk"><img src="https://img.shields.io/pypi/pyversions/pytimetk.svg?style=for-the-badge" alt="versions"></a>
  <a href="https://business-science.github.io/pytimetk/contributing.html">
  <a href="https://github.com/business-science/pytimetk/blob/main/LICENSE"><img src="https://img.shields.io/github/license/business-science/pytimetk.svg?style=for-the-badge" alt="license"></a>
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge"/>
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/business-science/pytimetk?style=for-the-badge">
  </a>
</div>

# pytimetk — the time-series toolkit for people who build stuff

> Time series easier, faster, more fun.

[**Please ⭐ us on GitHub (it takes 2‑seconds and makes a huge difference).**](https://github.com/business-science/pytimetk)

---

## Why pytimetk?

- **Single API, multiple engines.** Every helper works on `pandas` _and_ `Polars` (many run on NVIDIA cudf/GPU as well).
- **Productivity first.** Visualization, aggregation, feature engineering, anomaly detection, and regime modeling in a couple of lines.
- **Performance obsessed.** Vectorized Polars support, GPU acceleration (beta), and feature-store style caching.

### The toolkit at a glance

| Workflow | pytimetk API | Superpower | Docs |
| --- | --- | --- | --- |
| Visualization & diagnostics | `plot_timeseries`, `plot_stl_diagnostics`, `plot_time_series_boxplot`, `theme_plotly_timetk` | Interactive Plotly charts, STL faceting, distribution-aware plots, Plotly theming helper | [Visualization guide](https://business-science.github.io/pytimetk/guides/01_visualization.html) |
| Time-aware aggregations | `summarize_by_time`, `apply_by_time`, `pad_by_time(fillna=…)` | Resample, roll up, and now fill padded rows with a single scalar | [Selectors & periods guide](https://business-science.github.io/pytimetk/guides/08_selectors_dates.html) |
| Feature engineering | `augment_timeseries_signature`, `augment_rolling`, `augment_wavelet`, `feature_store` | Calendar signatures, GPU-ready rolling windows, wavelets, reusable feature sets | [Feature engineering reference](https://business-science.github.io/pytimetk/reference/index.html#feature-engineering) |
| Anomaly workflows | `anomalize`, `plot_anomalies`, `plot_anomalies_decomp`, `plot_anomalies_cleaned` | Detect → diagnose → visualize anomalies without switching libraries | [Anomaly docs](https://business-science.github.io/pytimetk/reference/anomalize.html) |
| Finance & regimes | `augment_regime_detection` (✨ `regime_backends` extra), `augment_macd`, … | HMM-based regime detection with hmmlearn or pomegranate, dozens of indicators | [Finance module](https://business-science.github.io/pytimetk/reference/index.html#finance) |
| Polars-native workflows | `.tk` accessor on `pl.DataFrame`, `engine="polars"` on heavy helpers | Plot, summarize, and engineer features without ever leaving Polars | [Polars guide](https://business-science.github.io/pytimetk/guides/07_polars.html) |
| Production extras (beta) | Feature store, MLflow integration, GPU acceleration | Cache expensive transforms, log metadata, or flip a switch for RAPIDS | [Production docs](https://business-science.github.io/pytimetk/production/02_gpu_acceleration.html) |

---


## Installation

Install the latest stable version of `pytimetk` using `pip`:

```bash
pip install pytimetk
```

Alternatively you can install the development version:

```bash
pip install --upgrade --force-reinstall git+https://github.com/business-science/pytimetk.git
```

---

## 60‑second tour

```python
import numpy as np
import pandas as pd
import pytimetk as tk
from pytimetk.utils.selection import contains

sales = tk.load_dataset("bike_sales_sample", parse_dates=["order_date"])

# 1. Summaries in one line (Polars engine for speed)
monthly = (
    sales.groupby("category_1")
    .summarize_by_time(
        date_column="order_date",
        value_column="total_price",
        freq="MS",
        agg_func=["sum", "mean"],
        engine="polars",
    )
)

# 2. Visualize straight from Polars/pandas
monthly.plot_timeseries(
    date_column="order_date",
    value_column=contains("sum"),
    color_column="category_1",
    title="Revenue by Category",
    plotly_dropdown=True,
)

# 3. Fill gaps + detect anomalies
hourly = (
    sales.groupby(["category_1", "order_date"], as_index=False)
    .agg(total_price=("total_price", "sum"))
    .groupby("category_1")
    .pad_by_time(date_column="order_date", freq="1H", fillna=0)
)

anomalies = (
    hourly.groupby("category_1")
    .anomalize("order_date", "total_price")
    .plot_anomalies(date_column="order_date", plotly_dropdown=True)
)
```

---

## Fresh in the latest releases

- **New data visualizations** Discover new time series plots like Time Series Box Plots, Regression Plots, Seasonal and Decomposition plots in our upgraded [Guide 01](https://business-science.github.io/pytimetk/guides/01_visualization.html).
- **Selectors + natural periods guide.** Learn how to point at columns with `contains()`/`starts_with()` and specify periods like `"2 weeks"` or `"45 minutes"`. → [Guide 08](guides/08_selectors_dates.html)
- **Polars everywhere.** Dedicated [Polars guide](https://business-science.github.io/pytimetk/guides/07_polars.html) plus `.tk` accessor coverage for plotting, feature engineering, and gap filling.
- **GPU + Feature Store (beta).** Run rolling stats using our [RAPIDS cudf guide](https://business-science.github.io/pytimetk/production/02_gpu_acceleration.html) or cache/track expensive feature sets with metadata and MLflow hooks in our new [Feature Store guide](https://business-science.github.io/pytimetk/production/01_feature_store.html).


---

## Guides & docs

| Topic | Why read it? |
| --- | --- |
| [Quick Start](https://business-science.github.io/pytimetk/getting-started/02_quick_start.html) | Load data, plot, summarize, and forecast-ready features in ~5 minutes. |
| [Visualization Guide](https://business-science.github.io/pytimetk/guides/01_visualization.html) | Deep dive into `plot_timeseries`, STL diagnostics, anomaly plots, and Plotly theming. |
| [Polars Guide](https://business-science.github.io/pytimetk/guides/07_polars.html) | How to keep data in Polars while still using pytimetk plotting/feature APIs. |
| [Selectors & Human Durations](https://business-science.github.io/pytimetk/guides/08_selectors_dates.html) | Column selectors, natural-language periods, and new padding/future-frame tricks. |
| [Production / GPU](https://business-science.github.io/pytimetk/production/02_gpu_acceleration.html) | Feature store beta, caching, MLflow logging, and NVIDIA RAPIDS setup. |
| [API Reference](https://business-science.github.io/pytimetk/reference/index.html) | Full catalogue of helpers by module. |

---

## Quickstart snippet

```python
import pandas as pd
import pytimetk as tk

df = tk.load_dataset("bike_sales_sample", parse_dates=["order_date"])

(df.groupby("category_2")
   .summarize_by_time(
       date_column="order_date",
       value_column="total_price",
       freq="MS",
       agg_func=["mean", "sum"],
       engine="polars",
   ))
```

---

## Feature Store & Caching (Beta)

> ⚠️ **Beta:** The Feature Store APIs and on-disk format may change before general availability. We’d love [feedback and bug reports](https://github.com/business-science/pytimetk/issues).

Persist expensive feature engineering steps once and reuse them everywhere. Register a transform, build it on a dataset, and reload it in any notebook or job with automatic versioning, metadata, and cache hits.

```python
import pandas as pd
import pytimetk as tk

df = tk.load_dataset("bike_sales_sample", parse_dates=["order_date"])

store = tk.FeatureStore()

store.register(
    "sales_signature",
    lambda data: tk.augment_timeseries_signature(
        data,
        date_column="order_date",
        engine="pandas",
    ),
    default_key_columns=("order_id",),
    description="Calendar signatures for sales orders.",
)

result = store.build("sales_signature", df)
print(result.from_cache)  # False first run, True on subsequent builds
```

- Supports local disk or any `pyarrow` filesystem (e.g., `s3://`, `gs://`) via the `artifact_uri` parameter, plus optional file-based locking for concurrent jobs.
- Optional MLflow helpers capture feature versions and artifacts with your experiments for reproducible pipelines.

# 🏆 More Coming Soon...

We are in the early stages of development. But it's obvious the potential for `pytimetk` now in Python. 🐍

- Please [⭐ us on GitHub](https://github.com/business-science/pytimetk) (it takes 2-seconds and means a lot). 
- To make requests, please see our [Project Roadmap GH Issue #2](https://github.com/business-science/pytimetk/issues/2). You can make requests there. 
- Want to contribute? [See our contributing guide here.](/contributing.html)

# ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=business-science/pytimetk&type=Date)](https://star-history.com/#business-science/pytimetk&Date)

[**Please ⭐ us on GitHub (it takes 2 seconds and means a lot).**](https://github.com/business-science/pytimetk)

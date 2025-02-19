<div align="center">
<img src="docs/logo-timetk.png" width="30%"/>
</div>

<div align="center">
  <a href="https://github.com/business-science/pytimetk/actions">
  <img alt="Github Actions" src="https://github.com/business-science/pytimetk/actions/workflows/timetk-checks.yaml/badge.svg"/>
  </a>
  <a href="https://pypi.python.org/pypi/pytimetk">
  <img alt="PyPI Version" src="https://img.shields.io/pypi/v/pytimetk.svg"/>
  </a>
  <a href="https://business-science.github.io/pytimetk/contributing.html">
  <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"/>
  </a>
</div>

# pytimetk

> Time series easier, faster, more fun. Pytimetk.

[**Please ⭐ us on GitHub (it takes 2-seconds and means a lot).**](https://github.com/business-science/pytimetk)

# Introducing pytimetk: Simplifying Time Series Analysis for Everyone

Time series analysis is fundamental in many fields, from business forecasting to scientific research. While the Python ecosystem offers tools like `pandas`, they sometimes can be verbose and not optimized for all operations, especially for complex time-based aggregations and visualizations.

Enter **pytimetk**. Crafted with a blend of ease-of-use and computational efficiency, `pytimetk` significantly simplifies the process of time series manipulation and visualization. By leveraging the `polars` backend, you can experience speed improvements ranging from 3X to a whopping 3500X. Let's dive into a comparative analysis.

| Features/Properties | **pytimetk**                  | **pandas (+matplotlib)**               |
|---------------------|-------------------------------|---------------------------------------|
| **Speed**           | 🚀 3X to 3500X Faster          | 🐢 Standard                           |
| **Code Simplicity** | 🎉 Concise, readable syntax    | 📜 Often verbose                      |
| `plot_timeseries()` | 🎨 2 lines, no customization  | 🎨 16 lines, customization needed    |
| `summarize_by_time()` | 🕐 2 lines, 13.4X faster     | 🕐 6 lines, 2 for-loops               |
| `pad_by_time()`     | ⛳ 2 lines, fills gaps in timeseries        | ❌ No equivalent    |
| `anomalize()`       | 📈 2 lines, detects and corrects anomalies  | ❌ No equivalent    |
| `augment_timeseries_signature()` | 📅 1 line, all calendar features    | 🕐 29 lines of `dt` extractors |
| `augment_rolling()` | 🏎️ 10X to 3500X faster     | 🐢 Slow Rolling Operations |

As evident from the table, **pytimetk** is not just about speed; it also simplifies your codebase. For example, `summarize_by_time()`, converts a 6-line, double for-loop routine in `pandas` into a concise 2-line operation. And with the `polars` engine, get results 13.4X faster than `pandas`!
  
Similarly, `plot_timeseries()` dramatically streamlines the plotting process, encapsulating what would typically require 16 lines of `matplotlib` code into a mere 2-line command in **pytimetk**, without sacrificing customization or quality. And with `plotly` and `plotnine` engines, you can create interactive plots and beautiful static visualizations with just a few lines of code.

For calendar features, **pytimetk** offers `augment_timeseries_signature()` which cuts down on over 30 lines of `pandas` dt extractions. For rolling features, **pytimetk** offers `augment_rolling()`, which is 10X to 3500X faster than `pandas`. It also offers `pad_by_time()` to fill gaps in your time series data, and `anomalize()` to detect and correct anomalies in your time series data.

Join the revolution in time series analysis. Reduce your code complexity, increase your productivity, and harness the speed that **pytimetk** brings to your workflows.

Explore more at our [pytimetk homepage](https://business-science.github.io/pytimetk/).

# Installation

Install the latest stable version of `pytimetk` using `pip`:

```bash
pip install pytimetk
```

Alternatively you can install the development version:

```bash
pip install --upgrade --force-reinstall git+https://github.com/business-science/pytimetk.git
```

# Quickstart:

This is a simple code to test the function `summarize_by_time`:

```python
import pytimetk as tk
import pandas as pd

df = tk.datasets.load_dataset('bike_sales_sample')
df['order_date'] = pd.to_datetime(df['order_date'])

df \
    .groupby("category_2") \
    .summarize_by_time(
        date_column='order_date', 
        value_column= 'total_price',
        freq = "MS",
        agg_func = ['mean', 'sum'],
        engine = "polars"
    )
```

# Documentation

Get started with the [pytimetk documentation](https://business-science.github.io/pytimetk/)

- [📈 Overview](https://business-science.github.io/pytimetk/)
- [🏁 Getting Started](https://business-science.github.io/pytimetk/getting-started/02_quick_start.html)
- [🗺️ Beginner Guides](https://business-science.github.io/pytimetk/guides/01_visualization.html)
- [📘Applied Data Science Tutorials with PyTimeTK](https://business-science.github.io/pytimetk/tutorials/01_sales_crm.html)

- [📄 API Reference](https://business-science.github.io/pytimetk/reference/)

# Developers (Contributors): Installation

To install `pytimetk` using [Poetry](https://python-poetry.org/), follow these steps:

### 1. Prerequisites

Make sure you have Python 3.9 or later installed on your system.

### 2. Install Poetry

To install Poetry, you can use the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)  provided by Poetry. Do not use pip.

### 3. Clone the Repository

Clone the `pytimetk` repository from GitHub:

```bash
git clone https://github.com/business-science/pytimetk
```

### 4. Install Dependencies

Use Poetry to install the package and its dependencies:

```bash
poetry install
```

or you can create a virtualenv with poetry and install the dependencies

```bash
poetry shell
poetry install
```

# 🏆 More Coming Soon...

We are in the early stages of development. But it's obvious the potential for `pytimetk` now in Python. 🐍

- Please [⭐ us on GitHub](https://github.com/business-science/pytimetk) (it takes 2-seconds and means a lot). 
- To make requests, please see our [Project Roadmap GH Issue #2](https://github.com/business-science/pytimetk/issues/2). You can make requests there. 
- Want to contribute? [See our contributing guide here.](/contributing.html) 

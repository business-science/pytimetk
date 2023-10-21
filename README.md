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

> The time series toolkit for Python

**Please ⭐ us on GitHub (it takes 2-seconds and means a lot).**

# Introducing pytimetk: Simplifying Time Series Analysis for Everyone

Time series analysis is fundamental in many fields, from business forecasting to scientific research. While the Python ecosystem offers tools like `pandas`, they sometimes can be verbose and not optimized for all operations, especially for complex time-based aggregations and visualizations.

Enter **pytimetk**. Crafted with a blend of ease-of-use and computational efficiency, `pytimetk` significantly simplifies the process of time series manipulation and visualization. By leveraging the `polars` backend, you can experience speed improvements ranging from 3X to a whopping 30X. Let's dive into a comparative analysis.

| Features/Properties | **pytimetk**                  | **pandas (+matplotlib)**               |
|---------------------|-------------------------------|---------------------------------------|
| **Speed**           | 🚀 3X to 30X Faster            | 🐢 Standard                           |
| **Code Simplicity** | 🎉 Concise, readable syntax    | 📜 Often verbose                      |
| `summarize_by_time()` | 🕐 2 lines, 13.4X faster     | 🕐 6 lines, 2 for-loops               |
| `plot_timeseries()`  | 🎨 2 lines, no customization  | 🎨 16 lines, customization needed    |

As evident from the table:

- `summarize_by_time()` in **pytimetk** is not just about speed; it also simplifies your codebase, converting a 6-line, double for-loop routine in `pandas` into a concise 2-line operation.
  
- Similarly, `plot_timeseries()` dramatically streamlines the plotting process, encapsulating what would typically require 16 lines of `matplotlib` code into a mere 2-line command in **pytimetk**, without sacrificing customization or quality.

Join the revolution in time series analysis. Reduce your code complexity, increase your productivity, and harness the speed that **pytimetk** brings to your workflows.

Explore more at [pytimetk homepage](https://business-science.github.io/pytimetk/).

# Installation

Install the latest stable version of `pytimetk` using `pip`:

```bash
pip install pytimetk
```

Alternatively you can install the development version:

```bash
pip install git+https://github.com/business-science/pytimetk.git
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
        agg_func = ['mean', 'sum']
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
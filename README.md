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
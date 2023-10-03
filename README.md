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



## Developers (Contributors): Installation

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
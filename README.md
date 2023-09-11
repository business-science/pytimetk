

## TimeTK

> The time series toolkit for Python

This library is currently under development and is not intended for general usage yet. Functionality is experimental until release 0.1.0.

## Installation

To install `timetk` using [Poetry](https://python-poetry.org/), follow these steps:

### 1. Prerequisites

Make sure you have Python 3.9 or later installed on your system.

### 2. Install Poetry

To install Poetry, you can use the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)  provided by Poetry. Do not use pip.

### 3. Clone the Repository

Clone the `timetk` repository from GitHub:

```
git clone https://github.com/business-science/pytimetk
```

### 4. Install Dependencies

Use Poetry to install the package and its dependencies:

```
install poetry
```

or you can create a virtualenv with poetry and install the dependencies

```
poetry shell
poetry install
```

# Usage

This is a simple code to test the function `summarize_by_time`:

```
import timetk
import pandas as pd

df = timetk.data.load_dataset('bikes_sales_sample')
df['order_date'] = pd.to_datetime(df['order_date'])

df \
    .summarize_by_time(
        date_column='order_date', 
        value_column= 'total_price',
        groups = "category_2",
        rule = "M",
        kind = 'timestamp',
        agg_func = ['mean', 'sum']
    )
    

```
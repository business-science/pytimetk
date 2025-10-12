import polars as pl
import pytimetk as tk


def test_augment_hilbert_polars_groupby():
    df = tk.load_dataset("walmart_sales_weekly", parse_dates=["Date"]).head(40)

    result = (
        pl.from_pandas(df)
        .group_by("id")
        .tk.augment_hilbert(
            date_column="Date",
            value_column=["Weekly_Sales"],
        )
    )

    assert isinstance(result, pl.DataFrame)
    assert result.height == len(df)
    assert "Weekly_Sales_hilbert_real" in result.columns
    assert "Weekly_Sales_hilbert_imag" in result.columns

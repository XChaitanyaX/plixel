import matplotlib.pyplot as plt
import pandas as pd
import pytest

from plixel import SheetAnalyser

data = {
    "Business Unit": ["Software", "Software", "Advertising", "Advertising"],
    "Jan": [1e5, 1e6, 1e7, 1e8],
    "Feb": [1e6, 1e7, 1e8, 1e9],
    "Year": [2020, 2020, 2021, 2021],
}

df = pd.DataFrame(data)
global_sa = SheetAnalyser(df=df)


def test_init() -> None:

    assert global_sa.df is not None


def test_plot_correlation_heatmap():
    global global_sa

    plot = global_sa.plot_correlation_heatmap()

    assert plt.get_fignums() != 0

    with pytest.raises(ValueError):
        err_data = {
            "Business Unit": ["Software", "Advertising"],
            "Jan": ["error", "data"],
        }

        err_df = pd.DataFrame(err_data)
        err_sa = SheetAnalyser(df=err_df)
        err_sa.plot_correlation_heatmap()


def test_get_trends():
    global global_sa

    trends = global_sa.get_trends()
    assert trends is not None

    with pytest.raises(ValueError):
        global_sa.get_trends(metric="error")


def test_plot_histogram():
    global global_sa

    plot = global_sa.plot_histogram(["Jan"])
    assert plt.get_fignums() != 0

    with pytest.raises(ValueError):
        err_data = {
            "Business Unit": ["Software", "Advertising"],
            "Jan": ["error", "data"],
        }
        err_df = pd.DataFrame(err_data)
        err_sa = SheetAnalyser(df=err_df)

        err_sa.plot_histogram(columns=["Jan", "Feb"])


def test_plot_business_units_over_years():
    global global_sa

    plot = global_sa.plot_business_units_over_years(
        business_col="Business Unit", business_unit="Software"
    )
    assert plt.get_fignums() != 0

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from plixel import SheetAnalyser
from openpyxl import Workbook

data = {
    "Business Unit": ["Software", "Software", "Advertising", "Advertising"],
    "Jan": [1e5, 1e6, 1e7, 1e8],
    "Feb": [1e6, 1e7, 1e8, 1e9],
    "Year": [2020, 2020, 2021, 2021],
}

df = pd.DataFrame(data)
global_sa1 = SheetAnalyser(df=df)

data1 = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Age": [20, 30, 40, 50],
    "Occupation": ["Engineer", "Doctor", "Artist", "Lawyer"],
    "Salary": [30000, 40000, 50000, 60000],
}

global_sa2 = SheetAnalyser(df=pd.DataFrame(data1))


def get_random_workbook() -> Workbook:
    return Workbook()


def test_init() -> None:

    assert global_sa1.df is not None

    with pytest.raises(ValueError):
        err_sa = SheetAnalyser()

    with pytest.raises(ValueError):
        err_sa = SheetAnalyser(file_path="error.xlsx")

    err_sa = SheetAnalyser(file_path="sample_files/business_data.xlsx")
    assert err_sa.df is not None


def test_plot_correlation_heatmap() -> None:
    global global_sa1

    plot = global_sa1.plot_correlation_heatmap()

    assert plt.get_fignums() != 0
    del plot

    with pytest.raises(ValueError):
        err_data = {
            "Business Unit": ["Software", "Advertising"],
            "Jan": ["error", "data"],
        }

        err_df = pd.DataFrame(err_data)
        err_sa = SheetAnalyser(df=err_df)
        err_sa.plot_correlation_heatmap()


def test_get_trends() -> None:
    global global_sa1

    trends = global_sa1.get_trends()
    assert trends is not None

    with pytest.raises(ValueError):
        global_sa1.get_trends(metric="error")


def test_plot_histogram() -> None:
    global global_sa1

    plot = global_sa1.plot_histogram(["Jan"])
    assert plt.get_fignums() != 0
    del plot

    with pytest.raises(ValueError):
        err_data = {
            "Business Unit": ["Software", "Advertising"],
            "Jan": ["error", "data"],
        }
        err_df = pd.DataFrame(err_data)
        err_sa = SheetAnalyser(df=err_df)

        err_sa.plot_histogram(columns=["Jan", "Feb"])


def test_plot_business_units_over_years() -> None:
    global global_sa1

    plot = global_sa1.plot_business_units_over_years(
        business_col="Business Unit", business_unit="Software"
    )
    assert plt.get_fignums() != 0

    with pytest.raises(ValueError):
        global_sa1.plot_business_units_over_years(
            business_col="Unit", business_unit="Software"
        )

    with pytest.raises(ValueError):
        global_sa1.plot_business_units_over_years(
            business_col="Business Unit", business_unit="Softwares"
        )

    with pytest.raises(ValueError):
        test_df = df.drop(columns=["Year"])
        test_sa = SheetAnalyser(df=test_df)

        test_sa.plot_business_units_over_years(
            business_col="Business Unit", business_unit="Software"
        )


def test_plot_barchart_for_each_month() -> None:
    global global_sa1

    plot = global_sa1.plot_barchart_for_each_month(
        business_col="Business Unit", business_unit="Software", year=2020
    )

    assert plt.get_fignums() != 0
    assert type(plot) == matplotlib.figure.Figure

    with pytest.raises(ValueError):
        global_sa1.plot_barchart_for_each_month(
            business_col="Unit", business_unit="Software", year=2020
        )

    with pytest.raises(ValueError):
        global_sa1.plot_barchart_for_each_month(
            business_col="Business Unit", business_unit="Softwares", year=2020
        )

    with pytest.raises(ValueError):
        test_df = df.drop(columns=["Year"])
        test_sa = SheetAnalyser(df=test_df)
        test_sa.plot_barchart_for_each_month(
            business_col="Business Unit", business_unit="Software", year=2024
        )

    with pytest.raises(ValueError):
        global_sa1.plot_barchart_for_each_month(
            business_col="Business Unit", business_unit="Software", year=9999
        )

    with pytest.raises(ValueError):
        global_sa1.plot_barchart_for_each_month(
            metric="error",
            business_col="Business Unit",
            business_unit="Software",
            year=2020,
        )


def test_get_no_of_employees() -> None:
    global global_sa2

    assert (
        global_sa2.get_no_of_employees(
            employee_col="Occupation", employee_type="Engineer"
        )
        == 1
    )

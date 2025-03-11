import calendar

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openpyxl import Workbook


class SheetAnalyser:
    def __init__(
        self,
        *,
        file_path: str | None = None,
        workbook: Workbook = None,
        names: str | None = None,
    ):
        if workbook is not None:
            self.df = pd.read_excel(workbook, sheet_name=names)
            self.active_sheet = workbook.active
        elif file_path is not None and file_path.endswith((".xlsx", ".xls")):
            self.active_sheet = pd.read_excel(file_path, sheet_name=names)
            if not isinstance(self.active_sheet, pd.DataFrame):
                self.sheets = list(self.active_sheet.keys())
                self.df = self.active_sheet[self.sheets[0]]
            else:
                self.df = self.active_sheet
        else:
            raise ValueError("Invalid file path or workbook")

    def get_trends(self, metric="mean"):
        """
        Returns the trend of the selected metric for all numeric columns in the DataFrame.

        Args:
            metric (str, optional): the trend of the given metric. Defaults to "mean".

        Raises:
            ValueError: if trend is not supported.

        Returns:
            dict: Trend of the selected metric for all numeric columns
        """
        numeric_cols = self.df.select_dtypes(include="number").columns

        metrics = {
            "mean": lambda col: self.df[col].mean(),
            "median": lambda col: self.df[col].median(),
            "max": lambda col: self.df[col].max(),
            "min": lambda col: self.df[col].min(),
            "std": lambda col: self.df[col].std(),
            "var": lambda col: self.df[col].var(),
        }

        if metric not in metrics:
            raise ValueError(f"Unsupported metric: {metric}")

        return {col: metrics[metric](col) for col in numeric_cols}

    def missing_values(self):
        return self.df.isnull().sum().to_dict()

    def correlation_matrix(self):
        return self.df.corr(numeric_only=True).to_dict()

    def duplicate_rows(self):
        return self.df[self.df.duplicated()].to_dict(orient="records")

    def unique_values(self):
        return {col: self.df[col].unique().tolist() for col in self.df.columns}

    def value_counts(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")
        return self.df[column].value_counts().to_dict()

    def plot_histogram(self, columns: list):
        """
        Plots histograms for the selected columns.

        Args:
            columns (list): List of columns to plot histograms for.
        Raises:
            ValueError: if any of the columns are not found in the DataFrame.

        Returns:
            Figure: Histograms for the selected columns

        """
        if not all(col in self.df.columns for col in columns):
            missing_cols = [col for col in columns if col not in self.df.columns]
            raise ValueError(f"Columns not found in the DataFrame: {missing_cols}")

        plt.figure(figsize=(10, 6))

        for col in columns:
            sns.histplot(self.df[col], kde=True, label=col, alpha=0.5)

        plt.title("Histograms for Selected Columns")
        plt.xlabel("Values")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()

        return plt.gcf()

    def plot_correlation_heatmap(self):
        """
        Plots a heatmap of the correlation matrix for the numeric columns in the DataFrame.

        Returns:
            Figure: Correlation Heatmap for the DataFrame
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")

        return fig

    def plot_business_units_over_years(
        self, *, business_col: str, business_unit: str, year: int
    ):
        """
        Plots the sales trend for a given business unit over the years.

        Args:
            business_col (str): business column name
            business_unit (str): business unit name
            year (int): year to plot the trend for

        Raises:
            ValueError: if business_col, business_unit, or year are not found in the DataFrame.

        Returns:
            Figure: Sales Trend for the given business unit over the years
        """
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")

        if business_unit not in self.df[business_col].unique():
            raise ValueError(
                f"Business unit '{business_unit}' not found in the DataFrame"
            )

        if "Year" not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")

        if year not in self.df["Year"].unique():
            raise ValueError(f"Year '{year}' not found in the DataFrame")

        plt.figure(figsize=(12, 8))
        months = tuple(calendar.month_abbr[1:])

        yearly_data = self.df[self.df["Year"] == year]

        for month in months:
            monthly_data = yearly_data[month].values
            plt.plot(monthly_data, label=month)

        plt.title(f"Sales Trend for {business_unit} in {year}")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()

        return plt.gcf()

    def plot_barchart_for_each_month(
        self, *, metric: str = "mean", business_col: str, business_unit: str, year: int
    ):
        """
        Plots the average sales for each month in a given year for a given business unit.

        Args:
            business_col (str): business column name
            business_unit (str): business unit name
            year (int): year to plot the trend for

        Raises:
            ValueError: if business_col, business_unit, or year are not found in the DataFrame.

        Returns:
            Figure: Average Sales for the given business unit in the given year
        """
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")

        if business_unit not in self.df[business_col].unique():
            raise ValueError(
                f"Business unit '{business_unit}' not found in the DataFrame"
            )

        if "Year" not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")

        if year not in self.df["Year"].unique():
            raise ValueError(f"Year '{year}' not found in the DataFrame")

        plt.figure(figsize=(12, 8))
        months = tuple(calendar.month_abbr[1:])

        yearly_data = self.df[self.df["Year"] == year]
        metrics = ("mean", "median", "max", "min", "std", "var")

        metric_functions = {
            "mean": lambda month: yearly_data[month].mean(),
            "median": lambda month: yearly_data[month].median(),
            "max": lambda month: yearly_data[month].max(),
            "min": lambda month: yearly_data[month].min(),
            "std": lambda month: yearly_data[month].std(),
            "var": lambda month: yearly_data[month].var(),
        }

        for month in months:
            monthly_data_avg = None
            if metric in metrics:
                monthly_data_avg = metric_functions[metric](month)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            plt.bar(month, monthly_data_avg, label=month)

        plt.title(f"Average Sales for {business_unit} in {year}")
        plt.xlabel("Month")
        plt.ylabel(f"{metric.capitalize()} Sales")
        plt.legend()
        plt.tight_layout()

        return plt.gcf()

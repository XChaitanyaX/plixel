import calendar
import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from openpyxl import Workbook


class SheetAnalyser:
    """A class to analyse Excel sheets.

    Attributes
    -----------
    df : DataFrame
        The DataFrame containing the data from the Excel sheet.
    active_sheet : DataFrame
        The active sheet in the Excel workbook.
    sheets : list[str]
        List of sheet names in the Excel workbook.

    Methods
    --------
    get_trends(metric: str = "mean") -> dict:
        Returns the trend of the selected metric for all numeric columns in the DataFrame.

    plot_histogram(columns: list) -> plt.Figure:
        Plots histograms for the selected columns.

    plot_correlation_heatmap() -> plt.Figure:
        Plots a heatmap of the correlation matrix for the numeric columns in the DataFrame.

    Most of the methods are derived from pd.DataFrame class so that there is no need to type more code.

    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 3, 4, 5],
    ...     "B": [6, 7, 8, 9, 10],
    ...     "C": [11, 12, 13, 14, 15]
    ... })
    >>> sheet_analyser = SheetAnalyser(df=df)
    """

    def __init__(
        self,
        *,
        file_path: str | None = None,
        df: pd.DataFrame | None = None,
        workbook: Workbook | None = None,
    ) -> None:
        """Initializes the SheetAnalyser class.

        Args:
            file_path (str | None, optional): path of the xl file. Defaults to None.
            df (pd.DataFrame | None, optional): DataFrame representing the xl sheet. Defaults to None.
            workbook (Workbook, optional): a WorkBook from openpyxl. Defaults to None.
            names (str | None, optional): names of the columns. Mention only if column names are not present in the provided sheet. Defaults to None.
        Raises:
            ValueError: if none of the arguments are provided or given file_path does not exist.
        >>> sheet = SheetAnalyser(file_path="sample_files/business_data.xlsx")
        >>> sheet = SheetAnalyser(df=pd.DataFrame())
        >>> sheet = SheetAnalyser()
        Traceback (most recent call last):
            ...
        ValueError: Invalid file path or workbook
        """
        if workbook is not None:
            self.df = pd.read_excel(workbook)
            self.active_sheet = workbook.active

        elif file_path is not None and file_path.endswith((".xlsx", ".xls")):
            if not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")

            self.df = pd.read_excel(file_path)

        elif df is not None and isinstance(df, pd.DataFrame):
            self.df = df

        else:
            raise ValueError("Invalid file path or workbook")

        self.neutralize_cols()
        self.neutralize_rows()

    def get_columns(self, dtype) -> list[str]:
        return self.df.select_dtypes(include=dtype).columns.tolist()

    def neutralize_cols(self) -> None:
        self.df.columns = pd.Index([col.strip() for col in self.df.columns])

    def neutralize_rows(self) -> None:
        for i in self.get_columns(object):
            self.df[i] = self.df[i].apply(lambda x: x.strip())

    def get_trends(self, metric="mean") -> dict[str | Any, pd.Series | Any]:
        """
        Returns the trend of the selected metric for all numeric columns in the DataFrame.

        Args:
            metric (str, optional): the trend of the given metric. Defaults to "mean".

        Raises:
            ValueError: if trend is not supported.

        Returns:
            dict: Trend of the selected metric for all numeric columns

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3, 4, 5],
        ...     "B": [6, 7, 8, 9, 10],
        ...     "C": [11, 12, 13, 14, 15]
        ... })
        >>> sheet_analyser = SheetAnalyser(df=df)
        >>> sheet_analyser.get_trends()
        {'A': np.float64(3.0), 'B': np.float64(8.0), 'C': np.float64(13.0)}
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

    def plot_histogram(self, columns: list) -> plt.Figure:  # need to change
        """
        Plots histograms for the selected columns.

        Args:
            columns (list): List of columns to plot histograms for.
        Raises:
            ValueError: if any of the columns are not found in the DataFrame.

        Returns:
            Figure: Histograms for the selected columns

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3, 4, 5],
        ...     "B": [6, 7, 8, 9, 10],
        ...     "C": [11, 12, 13, 14, 15]
        ... })
        >>> sheet_analyser = SheetAnalyser(df=df)
        >>> plot = sheet_analyser.plot_histogram(columns=["A", "B", "C"])
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        >>> assert plt.get_fignums() != 0
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

    def plot_correlation_heatmap(self) -> plt.Figure:
        """
        Plots a heatmap of the correlation matrix for the numeric columns in the DataFrame.

        Returns:
            Figure: Correlation Heatmap for the DataFrame

        >>> sheet = SheetAnalyser(file_path="sample_files/business_data.xlsx")
        >>> plot = sheet.plot_correlation_heatmap()
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        >>> assert plt.get_fignums() != 0
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df.corr(numeric_only=True)
        if corr_matrix.empty:
            raise ValueError("No numeric columns found in the DataFrame")
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")

        return fig

    def plot_business_units_over_years(
        self, *, business_col: str, business_unit: str
    ) -> plt.Figure:
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

        >>> sheet = SheetAnalyser(file_path="sample_files/business_data.xlsx")
        >>> plot = sheet.plot_business_units_over_years(business_col="Businees Unit", business_unit="Software")
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        >>> assert plt.get_fignums() != 0
        """
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")

        if business_unit not in self.df[business_col].unique():
            raise ValueError(
                f"Business unit '{business_unit}' not found in the DataFrame"
            )

        if "Year" not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")

        plt.figure(figsize=(12, 8))
        unique_years = self.df["Year"].unique()
        unique_years.sort()

        months = tuple(calendar.month_abbr[1:])

        yearly_expenses = []
        for year in unique_years:
            yearly_data = self.df[self.df["Year"] == year]
            yearly_data_for_business_unit = yearly_data[
                yearly_data[business_col] == business_unit
            ]
            yearly_expenses = sum(
                yearly_data_for_business_unit[month].sum()
                for month in months
                if month in yearly_data_for_business_unit.columns
            )
            plt.bar(year, yearly_expenses, label=year)

        plt.title(f"Sales Trend for {business_unit} in {year}")
        plt.xlabel("Year")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()

        return plt.gcf()

    def plot_barchart_for_each_month(
        self, *, metric: str = "mean", business_col: str, business_unit: str, year: int
    ) -> plt.Figure:
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

        >>> sheet = SheetAnalyser(file_path="sample_files/business_data.xlsx")
        >>> plot = sheet.plot_barchart_for_each_month(business_col="Businees Unit", business_unit="Software", year=2012)
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        >>> assert plt.get_fignums() != 0
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
            if month not in yearly_data.columns:
                continue

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

    def get_no_of_employees(self, *, employee_col: str, employee_type: str) -> int:
        """
        Returns the number of employees in each department.

        Args:
            employee_col (str): column name containing the department names

        Raises:
            ValueError: if employee_col is not found in the DataFrame.

        Returns:
            dict: Number of employees in each department

        >>> sheet = SheetAnalyser(file_path="sample_files/employee_data.xlsx")
        >>> sheet.get_no_of_employees(employee_col="Occupation", employee_type="Engineer")
        6
        """
        if employee_col not in self.df.columns:
            raise ValueError(f"Column '{employee_col}' not found in the DataFrame")

        if employee_type not in self.df[employee_col].unique():
            raise ValueError(
                f"Employee type '{employee_type}' not found in the DataFrame"
            )

        return sum(1 for _ in self.df[employee_col] if _ == employee_type)

    def no_of_employees_above(self, threshold: int, *, salary_col: str) -> int:
        """
        Returns the number of employees with a salary above a certain threshold.

        Args:
            salary (int): the salary threshold
            salary_col (str): column name containing the salaries

        Raises:
            ValueError: if salary_col is not found in the DataFrame.

        Returns:
            int: Number of employees with a salary above the threshold
        """
        if salary_col not in self.df.columns:
            raise ValueError(f"Column '{salary_col}' not found in the DataFrame")

        return sum(1 for _ in self.df[salary_col] if _ > threshold)

    def no_of_employees_in_each_dept(self) -> dict[str, int]:
        """
        Returns the number of employees in each department.

        Returns:
            dict: Number of employees in each department
        """
        return self.df["Occupation"].value_counts().to_dict()

    def get_active_emp(self, emp_type: str, *, emp_col: str) -> int:
        """
        Returns the number of active employees of a given type.

        Args:
            emp_type (str): the type of employee to filter by
            emp_col (str): the column name containing the employee types

        Returns:
            int: Number of active employees of the given type
        """
        return self.df[self.df[emp_col] == emp_type]["Status"].value_counts()["Active"]

    def get_active_employees(self, *, emp_col: str) -> int:
        """
        Returns the number of active employees.

        Args:
            emp_col (str): the column name containing the employee types

        Returns:
            int: Number of active employees
        """
        return self.get_employee_status(emp_col=emp_col)["Active"]

    def get_inactive_employees(self, *, emp_col: str) -> int:
        """
        Returns the number of inactive employees.

        Args:
            emp_col (str): the column name containing the employee types

        Returns:
            int: Number of inactive employees
        """
        return self.get_employee_status(emp_col=emp_col)["Inactive"]

    def get_employee_status(self, *, emp_col: str) -> dict[str, int]:
        """
        Returns the number of employees in each status.

        Args:
            emp_col (str): the column name containing the employee types

        Returns:
            dict: Number of employees in each status
        """
        return self.df[emp_col].value_counts().to_dict()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os


class CsvAnalyser:
    """
    A class that represents a CSV file analyser.

    Attributes
    ----------
    file_path : str
        The path to the CSV file.
    df : pd.DataFrame
        The DataFrame representation of the CSV file.

    Methods
    -------
    get_summary()
        Returns a summary of the DataFrame.

    get_trends(metric: str)
        Returns the trends of the DataFrame.

    filter_rows(column: str, value)
        Filters the rows of the DataFrame based on the given column and value.

    plot_correlation()
        Plots the correlation matrix of the DataFrame.

    merge_csv(file_path: str)
        Merges the DataFrame with another CSV file.

    merge_dataframes(df2: pd.DataFrame)
        Merges the DataFrame with another DataFrame.

    ...

    Rather than having to type:
        CsvAnalyser.df.describe()

    You can just type:
        CsvAnalyser.describe()

    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 3],
    ...     "B": [4, 5, 6]
    ... })
    >>> analyser = CsvAnalyser(df=df)
    >>> analyser.get_trends()
    {'A': np.float64(2.0), 'B': np.float64(5.0)}

    """

    def __init__(
        self, *, df: pd.DataFrame | None = None, file_path: str | None = None
    ):
        """
        Args:
            df (DataFrame): the df to Analyse
            file_path (str): location of the csv, .data, or excel file

        Raises:
            ValueError: If none of the arguments are provided or invalid file type
            FileNotFoundError: If the file does not exist

        """
        if df is not None:
            self._df = df
            self.df = df

        elif file_path is not None:
            # Check file extension first
            if not file_path.endswith((".csv", ".data", ".xlsx", ".xls")):
                raise ValueError("Must provide a .csv, .data, .xlsx, or .xls file")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")

            self._path = file_path
            if file_path.endswith((".csv", ".data")):
                self._df = pd.read_csv(file_path)
                self.df = pd.read_csv(file_path)
            else:  # .xlsx or .xls
                self._df = pd.read_excel(file_path)
                self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Must provide atleast one argument")

    def get_trends(self, metric="mean"):
        """
        Returns the trends of the DataFrame for numeric columns.
        Discarding non-numeric columns.

        Args:
            metric (str, optional): the statistic metric. Defaults to "mean".

        Raises:
            ValueError: If the metric is not supported

        Returns:
            dict: A dictionary containing the trends of the DataFrame

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> analyser.get_trends()
        {'A': np.float64(2.0), 'B': np.float64(5.0)}

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

    def filter_rows(self, column: str, value):
        """
        Filters the rows of the DataFrame based on the given column and value.

        Args:
            column (str): column name to filter
            value (int | str): value to filter

        Raises:
            ValueError: If the column is not found in the DataFrame

        Returns:
            df.DataFrame: DataFrame with filtered rows

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> analyser.filter_rows("A", 2)
           A  B
        1  2  5
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        return self.df[self.df[column] == value]

    def plot_correlation(self):
        """
        Plot the correlation matrix of the DataFrame.
        Excludes null and N/A values, only uses numeric columns

        Returns:
            Figure: the plot of the correlation matrix

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> plot = analyser.plot_correlation()
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>

        """
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include="number")

        plt.figure(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True)
        plt.title("Correlation Matrix")

        return plt.gcf()

    def merge_csv(self, file_path: str):
        """

        Args:
            file_path (str): path to the csv file to merge

        Raises:
            ValueError: if the file is not a .csv file
            FileNotFoundError: if the file does not exist

        Returns:
            pd.DataFrame: the merged DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found")
        if file_path.endswith((".csv", ".data")):
            df2 = pd.read_csv(file_path)
        else:
            raise ValueError("Only .csv files are supported")

        return pd.concat([self.df, df2], axis=0, ignore_index=True)

    def merge_dataframes(self, df2: pd.DataFrame):
        """
        merge the DataFrame with another DataFrame.

        Args:
            df2 (pd.DataFrame): the DataFrame to merge

        Returns:
            pd.DataFrame: the merged DataFrame

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> df2 = pd.DataFrame({
        ...     "A": [7, 8, 9],
        ...     "B": [10, 11, 12]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> analyser.merge_dataframes(df2)
           A   B
        0  1   4
        1  2   5
        2  3   6
        3  7  10
        4  8  11
        5  9  12
        """
        return pd.concat([self.df, df2], axis=0, ignore_index=True)

    def change_to_init_state(self):
        """
        Changes the DataFrame to the initial state.

        Similar to reinitialising the object to the original DataFrame.

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> analyser.df = pd.DataFrame({
        ...     "A": [7, 8, 9],
        ...     "B": [10, 11, 12]
        ... })
        >>> analyser.change_to_init_state()
        >>> analyser.df.equals(df)
        True
        """
        self.df = self._df

    def to_csv(self, file_path: str):
        self.df.to_csv(file_path, index=False)

    def to_excel(self, file_path: str):
        self.df.to_excel(file_path, index=False)

    def to_json(self, file_path: str):
        self.df.to_json(file_path, orient="records")

    def standardise_headers(self):
        """
        Standardises the headers of the DataFrame.

        lowercases the columns of the df and replaces spaces with underscores.

        >>> header = ["First Name", "Last Name", "Age"]
        >>> df = pd.DataFrame(columns=header)
        >>> analyser = CsvAnalyser(df=df)
        >>> analyser.standardise_headers()
        >>> df.columns
        Index(['first_name', 'last_name', 'age'], dtype='object')
        """
        self.df.columns = [
            col.lower().replace(" ", "_") for col in self.df.columns
        ]

    def plot_column(self, column: str, plot_type: str):
        """

        Args:
            column (str): column name to plot
            plot_type (str): type of the plot

        Raises:
            ValueError: if column not found in the DataFrame or
            unsupported plot type

        Returns:
            Figure: the plot of the column

        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> plot = analyser.plot_column("A", "histogram")
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        """

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        if plot_type == "histogram":
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column])
            plt.title(f"Histogram of {column}")
        elif plot_type == "boxplot":
            plt.figure(figsize=(10, 6))
            sns.boxplot(self.df[column])
            plt.title(f"Boxplot of {column}")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        return plt.gcf()

    def plot_correlation_heatmap(self):
        """
        Alias for plot_correlation() for backward compatibility.
        Plot the correlation matrix of the DataFrame.

        Returns:
            Figure: the plot of the correlation matrix

        Raises:
            ValueError: If there are no numeric columns to correlate
        """
        numeric_cols = self.df.select_dtypes(include="number").columns
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation")

        return self.plot_correlation()

    def plot_histogram(self, columns: list):
        """
        Plot histograms for the specified columns.

        Args:
            columns (list): list of column names to plot

        Raises:
            ValueError: if any column is not found or not numeric

        Returns:
            Figure: the plot with histograms
        """
        for column in columns:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame")
            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' is not numeric")

        fig, axes = plt.subplots(1, len(columns), figsize=(10 * len(columns), 6))
        if len(columns) == 1:
            axes = [axes]

        for i, column in enumerate(columns):
            sns.histplot(self.df[column], ax=axes[i])
            axes[i].set_title(f"Histogram of {column}")

        plt.tight_layout()
        return plt.gcf()

    def plot_business_units_over_years(self, business_col: str, business_unit: str):
        """
        Plot business unit data over years.

        Args:
            business_col (str): column name containing business units
            business_unit (str): specific business unit to plot

        Raises:
            ValueError: if columns not found or data invalid

        Returns:
            Figure: the plot showing trends over years
        """
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")

        if business_unit not in self.df[business_col].values:
            raise ValueError(f"Business unit '{business_unit}' not found in column '{business_col}'")

        if "Year" not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")

        filtered_df = self.df[self.df[business_col] == business_unit]

        plt.figure(figsize=(10, 6))

        # Get numeric columns excluding Year
        numeric_cols = [col for col in filtered_df.select_dtypes(include="number").columns if col != "Year"]

        for col in numeric_cols:
            yearly_data = filtered_df.groupby("Year")[col].sum()
            plt.plot(yearly_data.index, yearly_data.values, marker='o', label=col)

        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.title(f"{business_unit} - Trends Over Years")
        plt.legend()
        plt.grid(True)

        return plt.gcf()

    def plot_barchart_for_each_month(
        self, business_col: str, business_unit: str, year: int, metric: str = "sum"
    ):
        """
        Plot bar chart for each month for a specific business unit and year.

        Args:
            business_col (str): column name containing business units
            business_unit (str): specific business unit to plot
            year (int): year to filter data
            metric (str): aggregation metric (default: "sum")

        Raises:
            ValueError: if columns not found, data invalid, or unsupported metric

        Returns:
            Figure: the bar chart for monthly data
        """
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")

        if business_unit not in self.df[business_col].values:
            raise ValueError(f"Business unit '{business_unit}' not found in column '{business_col}'")

        if "Year" not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")

        if year not in self.df["Year"].values:
            raise ValueError(f"Year {year} not found in the DataFrame")

        supported_metrics = ["sum", "mean", "median", "max", "min"]
        if metric not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metric}. Supported: {supported_metrics}")

        filtered_df = self.df[
            (self.df[business_col] == business_unit) & (self.df["Year"] == year)
        ]

        # Get month columns (common month names)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        available_months = [m for m in months if m in filtered_df.columns]

        if not available_months:
            raise ValueError("No month columns found in the DataFrame")

        # Aggregate monthly data based on metric
        if metric == "sum":
            monthly_values = [filtered_df[month].sum() for month in available_months]
        elif metric == "mean":
            monthly_values = [filtered_df[month].mean() for month in available_months]
        elif metric == "median":
            monthly_values = [filtered_df[month].median() for month in available_months]
        elif metric == "max":
            monthly_values = [filtered_df[month].max() for month in available_months]
        elif metric == "min":
            monthly_values = [filtered_df[month].min() for month in available_months]

        plt.figure(figsize=(12, 6))
        plt.bar(available_months, monthly_values)
        plt.xlabel("Month")
        plt.ylabel(f"Value ({metric})")
        plt.title(f"{business_unit} - Monthly Data for {year}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt.gcf()

    def get_no_of_employees(self, employee_col: str, employee_type: str):
        """
        Get the number of employees of a specific type.

        Args:
            employee_col (str): column name containing employee types
            employee_type (str): specific employee type to count

        Raises:
            ValueError: if column not found

        Returns:
            int: count of employees of the specified type
        """
        if employee_col not in self.df.columns:
            raise ValueError(f"Column '{employee_col}' not found in the DataFrame")

        return len(self.df[self.df[employee_col] == employee_type])

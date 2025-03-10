import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
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

    Most of these functions are exact copies of the functions present in pandas.DataFrame.

    They are just provided here for convenience of not typing too much.

    Rather than having to type:
        CsvAnalyser.df.describe()

    You can just type:
        CsvAnalyser.describe()

    >>> df = pd.DataFrame({
    ...     "A": [1, 2, 3],
    ...     "B": [4, 5, 6]
    ... })
    >>> analyser = CsvAnalyser(df=df)
    >>> analyser.get_summary()
    {'columns': ['A', 'B'], 'row_count': 3, 'stats': {'A': {'count': 3.0, 'mean': 2.0, 'std': 1.0, 'min': 1.0, '25%': 1.5, '50%': 2.0, '75%': 2.5, 'max': 3.0}, 'B': {'count': 3.0, 'mean': 5.0, 'std': 1.0, 'min': 4.0, '25%': 4.5, '50%': 5.0, '75%': 5.5, 'max': 6.0}}}
    >>> analyser.get_trends()
    {'A': np.float64(2.0), 'B': np.float64(5.0)}

    """

    def __init__(self, *, df: pd.DataFrame = None, file_path: str | None = None):
        """

        Args:
            df (DataFrame): the df to Analyse
            file_path (str): location of the csv or .data file

        Raises:
            ValueError: If none of the arguments are provided
            FileNotFoundError: If the file does not exist

        """
        if df is not None:
            self._df = df
            self.df = df

        elif file_path is not None and file_path.endswith((".csv", ".data")):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")
            self._path = file_path
            self._df = pd.read_csv(file_path)
            self.df = pd.read_csv(file_path)
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
        Excludes null and N/A values

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
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True)
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
        self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]

    def remove_duplicates(self):
        return self.df.drop_duplicates()

    def plot_column(self, column: str, plot_type: str):
        """

        Args:
            column (str): column name to plot
            plot_type (str): type of the plot

        Raises:
            ValueError: if column not found in the DataFrame or unsupported plot type

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

    def get_row(self, index: int):
        if index >= len(self.df):
            raise IndexError(f"Index {index} out of bounds")
        return self.df.iloc[index]

    def get_column(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        return self.df[column]

    def check_missing_values(self):
        return self.df.isnull().sum().to_dict()

    def fill_missing_values(self, value):
        return self.df.fillna(value)

    def drop_missing_values(self):
        self.df.dropna(inplace=True)
        return self.df

    def drop_column(self, column: str | list[str]):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        return self.df.drop(columns=column)

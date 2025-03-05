from typing import override
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
    """

    def __init__(self, *, df: pd.DataFrame = None, file_path: str = None):
        """

        Args:
            file_path (str): location of the csv or .data file

        Raises:
            ValueError: If none of the arguments are provided
            FileNotFoundError: If the file does not exist
            
        """
        if df:
            self.df = df
        
        elif file_path.endswith((".csv", ".data")):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found")
            self._path = file_path
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Must provide atleast one argument")
        
    
    def get_summary(self):
        """
        Returns a summary of the DataFrame.
        
        lists the columns, row count and statistics of the DataFrame
        
        Returns: A dictionary containing the columns, row count and statistics of the DataFrame
        
        """
        return {
            "columns": list(self.df.columns),
            "row_count": len(self.df),
            "stats": self.df.describe(include="all").to_dict(),
        }

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
            value (_type_): value to filter

        Raises:
            ValueError: If the column is not found in the DataFrame

        Returns:
            df.DataFrame: DataFrame with filtered rows
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        return self.df[self.df[column] == value]

    def plot_correlation(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.corr(), annot=True)
        plt.title("Correlation Matrix")

        return plt.gcf()

    def merge_csv(self, file_path: str):
        if file_path.endswith((".csv", ".data")):
            df2 = pd.read_csv(file_path)
        else:
            raise ValueError("Only .csv files are supported")

        return pd.concat([self.df, df2], axis=0, ignore_index=True)

    def merge_dataframes(self, df2: pd.DataFrame):
        return pd.concat([self.df, df2], axis=0, ignore_index=True)

    def change_to_init_state(self):
        self.df = pd.read_csv(self._path)

    def to_csv(self, file_path: str):
        self.df.to_csv(file_path, index=False)

    def to_excel(self, file_path: str):
        self.df.to_excel(file_path, index=False)

    def to_json(self, file_path: str):
        self.df.to_json(file_path, orient="records")

    def standardise_headers(self):
        self.df.columns = [col.lower().replace(" ", "_") for col in self.df.columns]

    def remove_duplicates(self):
        return self.df.drop_duplicates()

    def plot_column(self, column: str, plot_type: str):
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
    
    
    
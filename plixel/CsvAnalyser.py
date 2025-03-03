import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


class CsvAnalyser:

    def __init__(self, file_path: str):
        if file_path.endswith((".csv", ".data")):
            self._path = file_path
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError("Only .csv files are supported")

    def get_summary(self):
        return {
            "columns": list(self.df.columns),
            "row_count": len(self.df),
            "stats": self.df.describe(include="all").to_dict(),
        }

    def get_trends(self, metric="mean"):
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
    
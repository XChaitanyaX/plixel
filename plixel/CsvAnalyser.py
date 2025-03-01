import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class CsvAnalyser:

    def __init__(self, file_path: str):
        if file_path.endswith(('.csv', '.data')):
            self.df = pd.read_csv(file_path)
        else:
            raise ValueError('Only .csv files are supported')
        
        
    def get_summary(self):
        return {
            'columns': list(self.df.columns),
            'row_count': len(self.df),
            'stats': self.df.describe(include="all").to_dict()
        }
        
    def get_trends(self, metric="mean"):
        numeric_cols = self.df.select_dtypes(include='number').columns
        
        metrics = {
            'mean': lambda col: self.df[col].mean(),
            'median': lambda col: self.df[col].median(),
            'max': lambda col: self.df[col].max(),
            'min': lambda col: self.df[col].min(),
            'std': lambda col: self.df[col].std(),
            'var': lambda col: self.df[col].var()
        }

        if metric not in metrics:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return {col: metrics[metric](col) for col in numeric_cols}
    
    def plot_column(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[column])
        plt.title(f'Histogram of {column}')
        
        
        return plt.gcf()
            
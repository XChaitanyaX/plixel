import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class SheetAnalyser:
    def __init__(self, file_path: str):
        if file_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError('Only .xlsx files are supported')
        
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
    
    def missing_values(self):
        return self.df.isnull().sum().to_dict()

    def correlation_matrix(self):
        return self.df.corr(numeric_only=True).to_dict()

    def duplicate_rows(self):
        return self.df[self.df.duplicated()].to_dict(orient='records')

    def unique_values(self):
        return {col: self.df[col].unique().tolist() for col in self.df.columns}

    def value_counts(self, column: str):
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")
        return self.df[column].value_counts().to_dict()
    
    def plot_histogram(self, columns: list):
        if not all(col in self.df.columns for col in columns):
            missing_cols = [col for col in columns if col not in self.df.columns]
            raise ValueError(f"Columns not found in the DataFrame: {missing_cols}")
        
        plt.figure(figsize=(10, 6))
        
        for col in columns:
            sns.histplot(self.df[col], kde=True, label=col, alpha=0.5)
        
        plt.title('Histograms for Selected Columns')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()

    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = self.df.corr(numeric_only=True)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Heatmap')

        return fig

    def plot_business_units_over_years(self, *, business_col: str, business_unit: str, year: int):
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")
        
        if business_unit not in self.df[business_col].unique():
            raise ValueError(f"Business unit '{business_unit}' not found in the DataFrame")
        
        if 'Year' not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")
        
        if year not in self.df['Year'].unique():
            raise ValueError(f"Year '{year}' not found in the DataFrame")
        
        plt.figure(figsize=(12, 8))
        months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        
        yearly_data = self.df[self.df['Year'] == year]
        
        for month in months:
            monthly_data = yearly_data[month].values
            plt.plot(monthly_data, label=month)
            
        plt.title(f"Sales Trend for {business_unit} in {year}")
        plt.xlabel('Month')
        plt.ylabel('Sales')
        plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_barchart_for_each_month(self, *, business_col: str, business_unit: str, year: int):
        if business_col not in self.df.columns:
            raise ValueError(f"Column '{business_col}' not found in the DataFrame")
        
        if business_unit not in self.df[business_col].unique():
            raise ValueError(f"Business unit '{business_unit}' not found in the DataFrame")
        
        if 'Year' not in self.df.columns:
            raise ValueError("Column 'Year' not found in the DataFrame")
        
        if year not in self.df['Year'].unique():
            raise ValueError(f"Year '{year}' not found in the DataFrame")
        
        plt.figure(figsize=(12, 8))
        months = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
        
        yearly_data = self.df[self.df['Year'] == year]
        
        for month in months:
            monthly_data_avg = yearly_data[month].mean()
            plt.bar(month, monthly_data_avg, label=month)
            
        plt.title(f"Average Sales for {business_unit} in {year}")
        plt.xlabel('Month')
        plt.ylabel('Average Sales')
        # plt.legend()
        plt.tight_layout()
        
        return plt.gcf()
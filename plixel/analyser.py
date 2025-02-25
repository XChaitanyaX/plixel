import pandas as pd

class SheetAnalyser:
    def __init__(self, file_path):
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        else:
            raise NotImplementedError('File type not supported')
        
    def get_summary(self):
        return self.df.describe()
    
    def get_trends(self):
        return self.df.corr()
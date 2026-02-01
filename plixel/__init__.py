from .CsvAnalyser import CsvAnalyser
from .CsvPlotter import CsvPlotter

# Alias for backward compatibility
SheetAnalyser = CsvAnalyser

__version__ = "0.1.0"
__all__ = ["CsvAnalyser", "CsvPlotter", "SheetAnalyser"]

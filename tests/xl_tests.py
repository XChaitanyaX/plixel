import matplotlib.pyplot as plt
import pandas as pd

from plixel import SheetAnalyser

data = {
    "Business Unit": ["Software", "Advertising"],
    "Jan": [1e5, 1e6],
    "Feb": [1e6, 1e7],
}

df = pd.DataFrame(data)
global_sa = SheetAnalyser(df=df)


def test_init() -> None:
    
    assert global_sa.df is not None


def test_correlation_heatmap():
    global global_sa

    plot = global_sa.plot_correlation_heatmap()

    assert plt.get_fignums() != 0

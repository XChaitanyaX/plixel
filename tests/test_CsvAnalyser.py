import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from plixel import CsvAnalyser, CsvPlotter

matplotlib.use("agg")


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_csv_analyser_init():
    """Test CsvAnalyser initialization"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    assert analyser.df is not None
    assert isinstance(analyser.plotter, CsvPlotter)
    
    with pytest.raises(ValueError):
        err_sa = CsvAnalyser()


def test_get_trends():
    """Test get_trends method returns correct statistics"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    trends = analyser.get_trends()
    assert trends is not None
    assert "A" in trends
    assert "B" in trends
    assert trends["A"] == 2.0
    assert trends["B"] == 5.0
    
    with pytest.raises(ValueError):
        analyser.get_trends(metric="error")


def test_filter_rows():
    """Test filter_rows method"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    filtered = analyser.filter_rows("A", 2)
    assert len(filtered) == 1
    assert filtered["A"].values[0] == 2
    
    with pytest.raises(ValueError):
        analyser.filter_rows("NonExistent", 1)


def test_plot_correlation():
    """Test plot_correlation delegates to plotter"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    plot = analyser.plot_correlation()
    assert plot is not None
    assert isinstance(plot, matplotlib.figure.Figure)
    assert plt.get_fignums() != 0


def test_plot_column():
    """Test plot_column delegates to plotter"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    plot = analyser.plot_column("A", "histogram")
    assert plot is not None
    assert isinstance(plot, matplotlib.figure.Figure)
    
    with pytest.raises(ValueError):
        analyser.plot_column("NonExistent", "histogram")
    
    with pytest.raises(ValueError):
        analyser.plot_column("A", "unsupported_type")


def test_csv_plotter_direct():
    """Test CsvPlotter can be used independently"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    plotter = CsvPlotter(analyser)
    
    plot = plotter.plot_correlation()
    assert plot is not None
    assert isinstance(plot, matplotlib.figure.Figure)


def test_merge_dataframes():
    """Test merge_dataframes method"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
    analyser = CsvAnalyser(df=df)
    
    merged = analyser.merge_dataframes(df2)
    assert len(merged) == 6
    assert merged["A"].tolist() == [1, 2, 3, 7, 8, 9]


def test_standardise_headers():
    """Test standardise_headers method"""
    df = pd.DataFrame(columns=["First Name", "Last Name", "Age"])
    analyser = CsvAnalyser(df=df)
    
    analyser.standardise_headers()
    assert "first_name" in analyser.df.columns
    assert "last_name" in analyser.df.columns
    assert "age" in analyser.df.columns


def test_change_to_init_state():
    """Test change_to_init_state method"""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    analyser = CsvAnalyser(df=df)
    
    analyser.df = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
    analyser.change_to_init_state()
    
    assert analyser.df.equals(df)

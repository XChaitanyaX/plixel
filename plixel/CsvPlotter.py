import matplotlib.pyplot as plt
import seaborn as sns


class CsvPlotter:
    """
    A class that handles plotting functionality for CSV data.

    Attributes
    ----------
    analyser : CsvAnalyser
        The CsvAnalyser instance to plot data from.

    Methods
    -------
    plot_correlation()
        Plots the correlation matrix of the DataFrame.

    plot_column(column: str, plot_type: str)
        Plots a specific column with the specified plot type.
    """

    def __init__(self, analyser):
        """
        Args:
            analyser: The CsvAnalyser instance containing the data to plot.
        """
        self.analyser = analyser

    def plot_correlation(self):
        """
        Plot the correlation matrix of the DataFrame.
        Excludes null and N/A values

        Returns:
            Figure: the plot of the correlation matrix

        >>> import pandas as pd
        >>> from plixel import CsvAnalyser
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> plotter = CsvPlotter(analyser)
        >>> plot = plotter.plot_correlation()
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>

        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.analyser.df.corr(), annot=True)
        plt.title("Correlation Matrix")

        return plt.gcf()

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

        >>> import pandas as pd
        >>> from plixel import CsvAnalyser
        >>> df = pd.DataFrame({
        ...     "A": [1, 2, 3],
        ...     "B": [4, 5, 6]
        ... })
        >>> analyser = CsvAnalyser(df=df)
        >>> plotter = CsvPlotter(analyser)
        >>> plot = plotter.plot_column("A", "histogram")
        >>> type(plot)
        <class 'matplotlib.figure.Figure'>
        """

        if column not in self.analyser.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame")

        if plot_type == "histogram":
            plt.figure(figsize=(10, 6))
            sns.histplot(self.analyser.df[column])
            plt.title(f"Histogram of {column}")
        elif plot_type == "boxplot":
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=self.analyser.df[column])
            plt.title(f"Boxplot of {column}")
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        return plt.gcf()

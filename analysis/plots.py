"""
Plotting utilities for cross-sectional equity signal project.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_returns(pnl_df: pd.DataFrame, column: str = "pnl_net", title: str = "Cumulative PnL"):
    """
    Plots cumulative PnL from a DataFrame.

    Parameters
    ----------
    pnl_df : pd.DataFrame
        Must contain a column with daily PnL
    column : str
        Column name for daily returns / PnL
    title : str
        Plot title
    """
    if column not in pnl_df.columns:
        raise ValueError(f"{column} not found in DataFrame")

    cumulative = pnl_df[column].cumsum()
    
    plt.figure(figsize=(10,5))
    plt.plot(cumulative.index, cumulative.values, label=column)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    plt.grid(True)
    plt.show()

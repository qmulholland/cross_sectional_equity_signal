"""
Target Module

Defines the target variable for cross-sectional equity prediction.
Typically, this is the forward return over a fixed horizon, optionally
excess return vs. a benchmark.
"""

import pandas as pd
import numpy as np


def compute_forward_returns(df: pd.DataFrame, horizon: int = 5, price_col='close') -> pd.DataFrame:
    """
    Compute forward returns for each ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', and price_col
    horizon : int
        Number of trading days to look ahead
    price_col : str
        Column name for price (e.g., 'close')

    Returns
    -------
    pd.DataFrame
        Same df with new column 'ret_{horizon}d'
    """
    df = df.copy()

    df = df.sort_values(['ticker', 'date'])
    df[f'ret_{horizon}d'] = df.groupby('ticker')[price_col].shift(-horizon) / df[price_col] - 1

    return df


def compute_excess_forward_returns(df: pd.DataFrame, horizon: int = 5, price_col='close', benchmark_col='benchmark') -> pd.DataFrame:
    """
    Compute forward returns relative to a benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', price_col, and benchmark_col
    horizon : int
        Forward return horizon in days
    price_col : str
        Column name for stock prices
    benchmark_col : str
        Column name for benchmark prices

    Returns
    -------
    pd.DataFrame
        Same df with new column 'ret_{horizon}d_excess'
    """
    df = df.copy()

    # Compute forward returns for stock and benchmark
    df = compute_forward_returns(df, horizon=horizon, price_col=price_col)
    df = df.sort_values(['date'])

    df['benchmark_ret'] = df.groupby('ticker')[benchmark_col].shift(-horizon) / df[benchmark_col] - 1

    df[f'ret_{horizon}d_excess'] = df[f'ret_{horizon}d'] - df['benchmark_ret']

    df.drop(columns=['benchmark_ret'], inplace=True)

    return df

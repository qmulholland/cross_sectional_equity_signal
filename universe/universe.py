"""
Universe Module

Defines and manages the stock universe for the cross-sectional equity strategy.
Allows filtering by sector, market cap, liquidity, or a fixed list of tickers.
"""

import pandas as pd


def fixed_tickers(tickers: list[str]) -> list[str]:
    """
    Use a fixed list of tickers as the universe.

    Parameters
    ----------
    tickers : list[str]
        List of stock tickers to include

    Returns
    -------
    list[str]
        Universe tickers
    """
    return tickers


def filter_by_sector(df: pd.DataFrame, sector_col='sector', sectors: list[str] | None = None) -> pd.DataFrame:
    """
    Filter stocks by sector.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain sector_col
    sector_col : str
        Column name for sector
    sectors : list[str], optional
        List of sectors to include. If None, include all.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    if sectors is None:
        return df
    return df[df[sector_col].isin(sectors)]


def filter_by_market_cap(df: pd.DataFrame, market_cap_col='market_cap', min_cap: float = 1e9, max_cap: float = 1e12) -> pd.DataFrame:
    """
    Filter stocks by market capitalization.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain market_cap_col
    market_cap_col : str
        Column name for market capitalization
    min_cap : float
        Minimum market cap (inclusive)
    max_cap : float
        Maximum market cap (inclusive)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    return df[(df[market_cap_col] >= min_cap) & (df[market_cap_col] <= max_cap)]


def filter_by_liquidity(df: pd.DataFrame, volume_col='volume', min_avg_volume: float = 1e5, window: int = 21) -> pd.DataFrame:
    """
    Filter stocks by liquidity (average daily volume).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ticker', 'date', and volume_col
    volume_col : str
        Column name for trading volume
    min_avg_volume : float
        Minimum average daily volume
    window : int
        Rolling window for average volume calculation

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    df = df.copy()
    df['avg_volume'] = df.groupby('ticker')[volume_col].transform(lambda x: x.rolling(window).mean())
    return df[df['avg_volume'] >= min_avg_volume].drop(columns=['avg_volume'])

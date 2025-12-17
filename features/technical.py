"""
Technical (price-based) feature engineering.

Design principles:
- Operates on long-format OHLCV data
- One row per (date, ticker)
- No data loading or I/O
- No execution logic
- Strictly avoids look-ahead bias
"""

import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns per ticker.

    Parameters
    ----------
    prices : pd.DataFrame
        Long-format OHLCV data with required columns:
        ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']

    Returns
    -------
    pd.DataFrame
        Same DataFrame with added column:
        - ret_1d
    """

    required_cols = {"date", "ticker", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = prices.copy()

    # Ensure proper ordering
    df = df.sort_values(["ticker", "date"])

    # Log returns (robust, standard in quant finance)
    df["ret_1d"] = (
        df.groupby("ticker")["close"]
        .apply(lambda x: x.pct_change())
    )

    return df

def compute_momentum(df: pd.DataFrame, windows: list[int] = [5, 10, 21]) -> pd.DataFrame:
    """
    Compute rolling momentum features for each ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ticker', 'date', and 'ret_1d'
    windows : list[int]
        List of rolling window sizes in days

    Returns
    -------
    pd.DataFrame
        Original df with new columns added:
        - mom_5, mom_10, mom_21, etc.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    for w in windows:
        # Momentum = cumulative return over past w days
        col_name = f"mom_{w}"
        df[col_name] = (
            df.groupby("ticker")["ret_1d"]
            .apply(lambda x: (1 + x).rolling(window=w, min_periods=1).apply(lambda r: r.prod() - 1))
        )

    return df

def compute_volatility(df: pd.DataFrame, windows: list[int] = [5, 10, 21]) -> pd.DataFrame:
    """
    Compute rolling volatility features for each ticker.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ticker', 'date', and 'ret_1d'
    windows : list[int]
        List of rolling window sizes in days

    Returns
    -------
    pd.DataFrame
        Original df with new columns added:
        - vol_5, vol_10, vol_21, etc.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    for w in windows:
        col_name = f"vol_{w}"
        df[col_name] = df.groupby("ticker")["ret_1d"].rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)

    return df

def cross_sectional_zscore(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute cross-sectional z-scores for specified features on each date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date' and 'ticker' columns
    feature_cols : list[str]
        List of feature column names to normalize

    Returns
    -------
    pd.DataFrame
        Original df with new columns:
        - original feature name + '_z' (e.g., 'mom_5_z', 'vol_10_z')
    """
    df = df.copy()
    df = df.sort_values(["date", "ticker"])

    for col in feature_cols:
        z_col = f"{col}_z"
        # Cross-sectional z-score per date
        df[z_col] = df.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0)
        )

    return df

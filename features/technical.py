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

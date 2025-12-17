"""
Data loading utilities for cross-sectional equity research.
"""

from typing import List
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: List[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download historical OHLCV data using yfinance.
    """

    print(f"Downloading data for {len(tickers)} tickers...")

    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if df.empty:
        raise ValueError("Downloaded price data is empty.")

    return df


def clean_price_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance MultiIndex columns into long-format DataFrame
    with one row per (date, ticker).
    """

    # Ensure MultiIndex columns
    if not isinstance(raw_df.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns from yfinance.")

    # Swap levels so ticker is first
    df = raw_df.copy()
    df.columns = df.columns.swaplevel(0, 1)

    # Sort for consistency
    df = df.sort_index(axis=1)

    # Convert columns -> rows
    df = df.stack(level=0).reset_index()

    # Rename columns cleanly
    df = df.rename(
        columns={
            "level_1": "ticker",
            "Date": "date",
        }
    )

    # Normalize column names
    df.columns = [c.lower() for c in df.columns]

    # Sort for time-series safety
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Sanity checks
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_prices(
    tickers: List[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    High-level wrapper: download + clean price data.
    """

    raw = download_prices(tickers, start_date, end_date)
    clean = clean_price_data(raw)

    print(
        f"Downloaded data for {clean['ticker'].nunique()} tickers. "
        f"Shape: {clean.shape}"
    )

    return clean


if __name__ == "__main__":
    TEST_TICKERS = ["AAPL", "MSFT", "GOOG"]

    prices = load_prices(
        tickers=TEST_TICKERS,
        start_date="2020-01-01",
    )

    print(prices.head())

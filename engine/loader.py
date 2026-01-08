"""
The loader.py module serves as the entry point for the entire pipeline.
It handles the automated retrieval for the ticker universe using the 
Yahoo Finance API. By centralizing the data ingestion and cleaning process,
it ensures that all subsequent strategy and backtesting modules receive
a standardized, 'analysis-ready' dataset. The modules transforms raw,
multi-indexed data into a 'Tidy' long-format DataFrame.
"""
import pandas as pd
import yfinance as yf

def load_data(tickers: list[str], start_date: str, end_date: str = None) -> pd.DataFrame:

    # Performs bulk download to reduce API overhead and improves ingestion speed
    # auto_adjust = False ensures we get both raw Close and Dividend-Adjusted Close (adj_close)
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, group_by="ticker")
    
    # Creates empty list
    frames = []

    # Iterates through multi-index columns to restructure the data
    for ticker in tickers:

        #Extract the specific ticker's slice from the bulk-downloaded DataFrame
        df = data[ticker].copy()

        # Add a ticker column to maintain asset identity after we concatenate the frames
        df["ticker"] = ticker
        frames.append(df)

    # Combine all individual stock DataFrames into a single master dataset
    prices = pd.concat(frames).reset_index()

    # Standardize column names for consistent mapping across the backtesting engine
    prices.columns = ["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]

    # Sort by asset and then time to ensure chronological integrity for rolling calculations
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Returns the cleaned, long-format price data
    return prices
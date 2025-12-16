import pandas as pd
import yfinance as yf

def load_prices(tickers=None, start="2020-01-01", end="2025-12-15"):
    """
    Load historical daily stock prices from yfinance.
    Returns a DataFrame with Date, Ticker, Open, High, Low, Close, Volume.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOG"]  # default test tickers

    all_data = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        df = yf.download(ticker, start=start, end=end)
        df.reset_index(inplace=True)
        df["Ticker"] = ticker
        all_data.append(df)

    result = pd.concat(all_data, ignore_index=True)
    print(f"Downloaded data for {len(tickers)} tickers. Shape: {result.shape}")
    return result

# Test the function if this file is run directly
if __name__ == "__main__":
    df = load_prices()
    print(df.head())

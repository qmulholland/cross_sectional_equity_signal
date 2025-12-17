"""
Diagnostics for cross_sectional_equity_signal project.

Includes:
- Data integrity checks
- Feature sanity checks
- Backtest sanity checks
"""

import pandas as pd
from data.loader import load_prices
from features.technical import (
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from backtest import backtest_decile
import matplotlib.pyplot as plt

def check_data(prices: pd.DataFrame):
    """Check basic data integrity."""
    print("=== DATA CHECK ===")
    print(f"Shape: {prices.shape}")
    print("Columns:", prices.columns.tolist())
    print("Null values per column:\n", prices.isnull().sum())
    print("Unique tickers:", prices['ticker'].nunique())
    print("Date range:", prices['date'].min(), "to", prices['date'].max())
    print()


def check_features(prices: pd.DataFrame, feature_cols: list[str]):
    """Check basic stats of features."""
    print("=== FEATURE CHECK ===")
    for col in feature_cols:
        if col not in prices.columns:
            print(f"WARNING: Column {col} not found")
            continue
        print(f"{col} | min: {prices[col].min():.4f}, max: {prices[col].max():.4f}, mean: {prices[col].mean():.4f}, std: {prices[col].std():.4f}")
    print()


def check_signal(prices: pd.DataFrame, signal_col: str):
    """Check signal distribution."""
    print("=== SIGNAL CHECK ===")
    if signal_col not in prices.columns:
        print(f"WARNING: Signal column {signal_col} not found")
        return
    print(f"{signal_col} | min: {prices[signal_col].min():.4f}, max: {prices[signal_col].max():.4f}, mean: {prices[signal_col].mean():.4f}, std: {prices[signal_col].std():.4f}")
    print()


def check_backtest(prices: pd.DataFrame, signal_col='pred_signal', ret_col='ret_1d'):
    """Run simple backtest and plot cumulative PnL."""
    print("=== BACKTEST CHECK ===")
    daily_pnl = backtest_decile(prices, signal_col=signal_col, ret_col=ret_col)
    print("Daily PnL head:\n", daily_pnl.head())
    print("Cumulative PnL:\n", daily_pnl.cumsum().tail())

    # Plot cumulative PnL
    cumulative = daily_pnl.cumsum()
    plt.figure(figsize=(10,5))
    cumulative.plot(title="Cumulative PnL")
    plt.xlabel("Date")
    plt.ylabel("Cumulative PnL")
    plt.show()


if __name__ == "__main__":
    # Example workflow
    tickers = ["AAPL", "MSFT", "GOOG"]
    prices = load_prices(tickers=tickers, start_date="2020-01-01")

    # Compute features
    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)

    features_to_zscore = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features_to_zscore)

    # Generate signal
    signal_features = [f"{f}_z" for f in features_to_zscore]
    prices = generate_signal(prices, signal_features)

    # Run diagnostics
    check_data(prices)
    check_features(prices, feature_cols=signal_features)
    check_signal(prices, signal_col='pred_signal')
    check_backtest(prices)

"""
Performance Analysis for cross_sectional_equity_signal

Milestones implemented:
1. Mean daily return, Std Dev, Sharpe, Hit Rate
2. Transaction costs
3. Out-of-sample split (train/test by date)
4. Benchmark comparison (equal-weighted portfolio and SPY)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data.loader import load_prices
from features.technical import (
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)
from backtest.backtest import backtest_decile
from costs.costs import apply_transaction_costs

# Add SPY ticker constant
SPY_TICKER = "SPY"


def compute_metrics(pnl: pd.Series):
    """Compute mean, std, Sharpe, hit rate."""
    mean = pnl.mean()
    std = pnl.std()
    sharpe = mean / std * np.sqrt(252)  # annualized Sharpe
    hit_rate = (pnl > 0).sum() / len(pnl)
    return mean, std, sharpe, hit_rate

def print_metrics_summary(label, pnl_series, initial_capital=1000):
    """Helper to print requested metrics including Maximum Drawdown."""
    # Basic Metrics
    mean = pnl_series.mean()
    std = pnl_series.std()
    sharpe = (mean / std) * np.sqrt(252)
    hit_rate = (pnl_series > 0).mean() * 100
    
    # Compounded Growth
    cum_returns = (1 + pnl_series).cumprod()
    final_value = initial_capital * cum_returns.iloc[-1]
    
    # Maximum Drawdown calculation
    # We use the compounded returns series to find the peak-to-trough decline
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # Expressed as a negative percentage
    
    print(f"=== {label} ===")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Compounded Growth (Start $1000): ${final_value:,.2f}")
    print()

def performance_analysis(tickers, start_date="2020-01-01", split_date="2023-01-01"):
    # ... [Data loading and feature engineering code remains the same] ...
    prices = load_prices(tickers, start_date=start_date)
    prices = compute_daily_returns(prices)
    prices = compute_momentum(prices)
    prices = compute_volatility(prices)
    
    features = ["mom_5", "mom_10", "mom_21", "vol_5", "vol_10", "vol_21"]
    prices = cross_sectional_zscore(prices, features)
    signal_features = [f"{f}_z" for f in features]
    prices = generate_signal(prices, signal_features)

    train = prices[prices["date"] < split_date].copy()
    test = prices[prices["date"] >= split_date].copy()

    # --- In-sample performance ---
    pnl_train = backtest_decile(train, signal_col="pred_signal", ret_col="ret_1d")
    pnl_train = apply_transaction_costs(pnl_train, cost_bps=5)
    print_metrics_summary("IN-SAMPLE PERFORMANCE", pnl_train)
    ((1 + pnl_train).cumprod() - 1).plot(title="In-Sample Compounded PnL", figsize=(10,5),
                                            ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    # --- Out-of-sample performance ---
    pnl_test = backtest_decile(test, signal_col="pred_signal", ret_col="ret_1d")
    pnl_test = apply_transaction_costs(pnl_test, cost_bps=5)
    print_metrics_summary("OUT-OF-SAMPLE PERFORMANCE", pnl_test)
    ((1 + pnl_test).cumprod() - 1).plot(title="Out-of-Sample Compounded PnL", figsize=(10,5),
                                        ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    # --- Benchmark: equal-weight portfolio ---
    ew = test.groupby("date")["ret_1d"].mean()
    print_metrics_summary("BENCHMARK: EQUAL-WEIGHT PORTFOLIO", ew)
    ((1 + ew).cumprod() - 1).plot(title="Equal-Weight Compounded PnL", figsize=(10,5),
                                    ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    # --- Benchmark: SPY ---
    spy_prices = load_prices([SPY_TICKER], start_date=start_date)
    spy_prices = compute_daily_returns(spy_prices)
    spy_test = spy_prices[spy_prices["date"] >= split_date].copy()
    spy_returns = spy_test.set_index("date")["ret_1d"]
    
    print_metrics_summary("BENCHMARK: SPY", spy_returns)
    ((1 + spy_returns).cumprod() - 1).plot(title="SPY Compounded Returns", figsize=(10,5),
                                            ylabel="Compounded Total Return", xlabel="Date")
    plt.show()

    # --- Comparison Overlay ---
    cum_df = pd.DataFrame({
        "Strategy OOS": (1 + pnl_test).cumprod() - 1,
        "Equal-Weight": (1 + ew).cumprod() - 1,
        "SPY": (1 + spy_returns).cumprod() - 1,
    })
    cum_df.plot(title="Compounded Returns Comparison", figsize=(12,6), ylabel="Total Return")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.show()

if __name__ == "__main__":
    # Example tickers; you can expand to 10 or more later
    tickers = ["AAPL", "NVDA", "MSFT", "LCID"]
    performance_analysis(tickers)

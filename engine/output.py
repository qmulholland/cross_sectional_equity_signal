import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def print_metrics_summary(label, pnl_series, initial_capital=1000):
    """Formats and prints detailed performance report including capital growth and risk."""
    mean = pnl_series.mean()
    std = pnl_series.std()
    sharpe = (mean / std) * np.sqrt(252)
    hit_rate = (pnl_series > 0).mean() * 100
    
    # Calculates the sequence of portfolio values
    cum_returns = (1 + pnl_series).cumprod()
    final_value = initial_capital * cum_returns.iloc[-1]
    
    # Measures Drawdown
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    print(f"=== {label} ===")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Hit rate: {hit_rate:.1f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Compounded Growth (Start $1000): ${final_value:,.2f}")
    print()

def generate_performance_report(pnl_is, pnl_oos, test_data, spy_returns):
    """Generates visual charts with straightforward linear Y-axis scaling."""
    
    # 1. In-Sample Plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    is_growth = (1 + pnl_is).cumprod() * 1000  # Growth of $1000
    is_growth.plot(ax=ax1, title="In-Sample Strategy Growth ($1,000 Start)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)
    print_metrics_summary("IN-SAMPLE PERFORMANCE", pnl_is)
    plt.show()

    # 2. Out-of-Sample Plot
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    oos_growth = (1 + pnl_oos).cumprod() * 1000
    oos_growth.plot(ax=ax2, title="Out-of-Sample Strategy Growth ($1,000 Start)")
    ax2.set_ylabel("Portfolio Value ($)")
    ax2.grid(True, alpha=0.3)
    print_metrics_summary("OUT-OF-SAMPLE PERFORMANCE", pnl_oos)
    plt.show()

    # 3. Equal-Weight Benchmark
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ew = test_data.groupby("date")["ret_1d"].mean()
    ew_growth = (1 + ew).cumprod() * 1000
    ew_growth.plot(ax=ax3, title="Equal-Weight Portfolio Growth ($1,000 Start)")
    ax3.set_ylabel("Portfolio Value ($)")
    ax3.grid(True, alpha=0.3)
    print_metrics_summary("BENCHMARK: EQUAL-WEIGHT PORTFOLIO", ew)
    plt.show()

    # 4. SPY Benchmark
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    spy_growth = (1 + spy_returns).cumprod() * 1000
    spy_growth.plot(ax=ax4, title="SPY Benchmark Growth ($1,000 Start)")
    ax4.set_ylabel("Portfolio Value ($)")
    ax4.grid(True, alpha=0.3)
    print_metrics_summary("BENCHMARK: SPY", spy_returns)
    plt.show()

    # 5. Final Comparison Chart (Multi-line)
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    cum_df = pd.DataFrame({
        "Our Strategy": oos_growth,
        "Equal-Weight": ew_growth,
        "SPY Index": spy_growth,
    })
    cum_df.plot(ax=ax5, title="Comparison: Strategy vs Benchmarks (OOS Period)")
    ax5.set_ylabel("Portfolio Value ($)")
    ax5.axhline(1000, color='black', lw=1, ls='--') # Starting capital line
    ax5.grid(True, alpha=0.3)
    plt.show()
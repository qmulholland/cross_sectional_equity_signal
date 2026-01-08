"""
The output.py module is the reporting engine of the framework. It
translates raw daily returns into professional performance metrics
and visualizes the strategy's growth compared to a market benchmark
(the SPY). By providing a clear side-by-side comparison with the
S&P 500 (SPY), it allows for an immediate assessment of the strategy's
effectiveness and value-add. It utilizes NumPy for vectorized 
calculations of risk metrics like the Sharpe Ratio and Maximum Drawdown.
The visualization suite is built on Matplotlib, utilizing advanced
date-locators and formatters to handle multi-year time-series data
without overlapping labels, ensuring the final report is 
production-quality and easily interpretable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_performance_report(strat_returns, spy_returns, start_year):

    # Calculates the Annualized Sharpe Ratio to measure risk-adjusted returns
    # We multiply by the square root of 252 (trading days in a year) to annualize daily data
    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0

    # Calculate the terminal value of the portfolio based on a hypothetical $1000 initial investment
    cum_growth = (1 + strat_returns).prod() * 1000
    
    # Calculate Drawdown: The peak-to-trough decline during a specific record period
    cum_rets = (1 + strat_returns).cumprod()
    running_max = cum_rets.cummax()
    drawdown = (cum_rets / running_max) - 1
    max_dd = drawdown.min()

    # Output key performance metrics to the console for quick analysis
    print("\n" + "="*45)
    print(f"=== COMBINED PERFORMANCE (SINCE {start_year}) ===")
    print(f"Strategy's Annualized Sharpe: {sharpe:.2f}")
    print(f"Strategy's Max Drawdown:      {max_dd:.2%}")
    print(f"Strategy's Ending Value ($1000 Start): ${cum_growth:.2f}")
    
    # Calculate benchmark growth for a direct comparison of the strategy's "Alpha"
    spy_growth = (1 + spy_returns).prod() * 1000
    print(f"SPY's Ending Value ($1000 Start):      ${spy_growth:.2f}")
    print("="*45 + "\n")

    # Initialize the plot with professional sizing for visibility
    fig, ax = plt.subplots(figsize=(14, 7)) 
    
    # Convert daily percentage returns into a cumulative growth series for visualization
    strat_cum = (1 + strat_returns).cumprod() * 1000
    spy_cum = (1 + spy_returns).cumprod() * 1000
    
    # Plot the strategy against the benchmark with distinct styles for clarity
    ax.plot(strat_cum, label="Our Strategy", color="#1f77b4", lw=2)
    ax.plot(spy_cum, label="SPY Benchmark", color="#d62728", linestyle="--", alpha=0.8)
    
    # Configure the X-axis for better readability of time-series data
    #Show major date markers every 4 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    
    # Add minor month-by-month ticks to provide more granular visual context
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate labels to prevent overlapping on the X-axis
    plt.xticks(rotation=45)

    # Add chart metadata 
    ax.set_title(f"Strategy Growth vs. SPY Benchmark (Since {start_year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3) 
    
    # Optimize layout to ensure no text elements are cut off
    plt.tight_layout()
    plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_performance_report(strat_returns, spy_returns, start_year):
    """
    Generates a performance report with highly detailed x-axis (Day-Month-Year).
    """
    # 1. Calculate Metrics
    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
    cum_growth = (1 + strat_returns).prod() * 1000
    
    # Drawdown calculation
    cum_rets = (1 + strat_returns).cumprod()
    running_max = cum_rets.cummax()
    drawdown = (cum_rets / running_max) - 1
    max_dd = drawdown.min()

    # 2. Print Output
    print("\n" + "="*45)
    print(f"=== COMBINED PERFORMANCE (SINCE {start_year}) ===")
    print(f"Strategy's Annualized Sharpe: {sharpe:.2f}")
    print(f"Strategy's Max Drawdown:      {max_dd:.2%}")
    print(f"Strategy's Ending Value ($1000 Start): ${cum_growth:.2f}")
    
    spy_growth = (1 + spy_returns).prod() * 1000
    print(f"SPY's Ending Value ($1000 Start):      ${spy_growth:.2f}")
    print("="*45 + "\n")

    # 3. Combined Graph
    fig, ax = plt.subplots(figsize=(14, 7)) # Slightly wider for date room
    
    strat_cum = (1 + strat_returns).cumprod() * 1000
    spy_cum = (1 + spy_returns).cumprod() * 1000
    
    ax.plot(strat_cum, label="Our Strategy", color="#1f77b4", lw=2)
    ax.plot(spy_cum, label="SPY Benchmark", color="#d62728", linestyle="--", alpha=0.8)
    
    # --- DETAILED X-AXIS FORMATTING ---
    # Major Ticks: Labels showing Date-Month-Year every 4 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    
    # Minor Ticks: Small marks for every month (no labels) to show density
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    
    # Rotate labels 45 degrees for readability
    plt.xticks(rotation=45)
    # ----------------------------------

    ax.set_title(f"Strategy Growth vs. SPY Benchmark (Since {start_year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3) # Show grid for major and minor ticks
    
    plt.tight_layout()
    plt.show()
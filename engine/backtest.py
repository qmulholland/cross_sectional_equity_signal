import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, initial_capital=1000):
    """
    Vectorized backtesting engine.
    Applies a Top 25 Volume filter and calculates portfolio returns without loops.
    """
    df = df.copy()

    # 1. Rank tickers by volume for each date (Cross-sectional filter)
    # We use 'first' to handle cases where volumes might be identical
    df['vol_rank'] = df.groupby('date')['volume'].rank(ascending=False, method='first')

    # 2. Identify the active universe (Top 25)
    df['is_top_25'] = df['vol_rank'] <= 25

    # 3. Define the Strategy Signal
    # We only trade if the signal is '1' AND the stock is in the Top 25
    df['active_signal'] = np.where((df['is_top_25']) & (df['signal'] == 1), 1, 0)
    
    # Note: If your strategy has an exit signal (-1), 
    # the logic here simplifies to holding as long as the signal is positive.

    # 4. Calculate Daily Returns per Ticker
    df['daily_ret'] = df.groupby('ticker')['adj_close'].pct_change()

    # 5. Apply Signal to Returns
    # Shift signals by 1 day to ensure we enter AFTER the signal is generated (No look-ahead bias)
    df['strat_ret_ticker'] = df.groupby('ticker')['active_signal'].shift(1) * df['daily_ret']

    # 6. Aggregate Portfolio Returns
    # We average the returns of the stocks we are currently holding
    portfolio_returns = df.groupby('date')['strat_ret_ticker'].mean().fillna(0)

    # 7. Convert Returns to Growth of Initial Capital
    # This matches the 'total_value' logic in your iterative code
    cumulative_growth = (1 + portfolio_returns).cumprod() * initial_capital
    
    # Return as a percentage change series to satisfy output.py requirements
    return cumulative_growth.pct_change().fillna(0)
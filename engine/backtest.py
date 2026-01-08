"""
The backtest.py module is the mathematical core of the 
framework. It transforms theorhetical trading signals
into a simulated portfolio performance. It utilizes a
cross-sectional ranking logic to filter the top 25 
stocks by volume daily, ensuring we are only trading 
liquid assets. To maintain statistical integrity, it 
employs a one-day signal lag (.shift(1)) to eliminate
look-ahead bias, ensuring that execution occurs at the 
price following the signal generation.
"""
import pandas as pd
import numpy as np

def run_backtest(df: pd.DataFrame, initial_capital=1000):

    # Create a deep copy to avoid modifying the original dataframe
    df = df.copy()

    # 1. Cross-Sectional Ranking: Rank all tickers by volume for every unique date
    # method='first' ensures a stable rank even if two stocks have identical volume
    df['vol_rank'] = df.groupby('date')['volume'].rank(ascending=False, method='first')

    # 2. Liquidity Filter: Create a boolean mask for the most liquid assets
    df['is_top_25'] = df['vol_rank'] <= 25

    # 3. Signal Execution: Only allow 'buy' signals (1) if the stock meets the liquidity requirement
    # This prevents the strategy from "trading" illiquid stocks with unreliable signals
    df['active_signal'] = np.where((df['is_top_25']) & (df['signal'] == 1), 1, 0)
    
    # 4. Returns Calculation: Compute the daily percentage change in adjusted close prices per ticker
    df['daily_ret'] = df.groupby('ticker')['adj_close'].pct_change()

    # 5. Vectorized P&L: Apply the signal from the previous day to today's return
    # .shift(1) is critical to prevent Look-Ahead Bias (buying at yesterday's signal using today's price)
    df['strat_ret_ticker'] = df.groupby('ticker')['active_signal'].shift(1) * df['daily_ret']

    # 6. Portfolio Aggregation: Average the returns of all active positions for each date
    # This assumes an equal-weighted allocation across all signaled stocks
    portfolio_returns = df.groupby('date')['strat_ret_ticker'].mean().fillna(0)

    # 7. Compounding: Calculate the growth of the initial capital over the entire backtest period
    cumulative_growth = (1 + portfolio_returns).cumprod() * initial_capital
    
    # Return the daily percentage change of the total portfolio value
    return cumulative_growth.pct_change().fillna(0)
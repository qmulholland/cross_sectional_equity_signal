import pandas as pd

def backtest_decile(df: pd.DataFrame, signal_col='pred_signal', ret_col='ret_1d', top_pct=0.1, cost_per_trade=0.0005):
    """
    Simple long-short decile backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', signal_col, and ret_col
    top_pct : float
        Fraction of tickers to long/short
    cost_per_trade : float
        Transaction cost per trade (as fraction of notional)

    Returns
    -------
    pd.Series
        Daily P&L series
    """
    df = df.copy()
    df['rank'] = df.groupby('date')[signal_col].rank(method='first', ascending=False)
    df['n_tickers'] = df.groupby('date')['ticker'].transform('count')

    # Assign weights
    df['weight'] = 0.0
    df.loc[df['rank'] <= df['n_tickers']*top_pct, 'weight'] = 1.0  # Long top decile
    df.loc[df['rank'] > df['n_tickers']*(1-top_pct), 'weight'] = -1.0  # Short bottom decile

    # Normalize by total absolute weight
    df['weight'] /= df.groupby('date')['weight'].transform(lambda x: x.abs().sum())

    # Compute daily return
    df['pnl'] = df['weight'] * df[ret_col]

    # Aggregate by date
    daily_pnl = df.groupby('date')['pnl'].sum()

    # Subtract simple transaction costs per trade
    trades = (df['weight'].diff().abs()).fillna(0)
    daily_costs = df.groupby('date')['weight'].sum() * cost_per_trade  # crude approximation
    daily_pnl -= daily_costs

    return daily_pnl

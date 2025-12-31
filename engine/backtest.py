import pandas as pd

def run_backtest(df: pd.DataFrame, signal_col: str = "pred_signal", cost_bps: int = 5) -> pd.Series:
    df = df.dropna(subset=[signal_col, "ret_1d"]).copy()
    
    # Prevent look-ahead bias
    df["signal_to_use"] = df.groupby("ticker")[signal_col].shift(1)
    df = df.dropna(subset=["signal_to_use"])
    
    # Concentration: Top 10% (Decile 9)
    df["decile"] = df.groupby("date")["signal_to_use"].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    
    # Select the Leaders
    leaders = df[df["decile"] == 9]
    
    # Equal-Weight of the TOP stocks only
    daily_pnl = leaders.groupby("date")["ret_1d"].mean()
    
    # Transaction costs (crucial when trading a concentrated portfolio)
    cost = cost_bps / 10000
    return daily_pnl - cost
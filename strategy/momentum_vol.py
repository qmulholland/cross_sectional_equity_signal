import pandas as pd
import numpy as np

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Calculate Returns
    df["ret_1d"] = df.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    
    # 2. Features: Look for 3-month (63 day) and 1-month (21 day) trends
    # Long-term momentum usually outperforms short-term 'noise'
    windows = [21, 63]
    for w in windows:
        df[f"mom_{w}"] = df.groupby("ticker")["adj_close"].pct_change(periods=w, fill_method=None)
        df[f"vol_{w}"] = (
            df.groupby("ticker")["ret_1d"]
            .rolling(w)
            .std()
            .reset_index(level=0, drop=True)
        )

    # 3. Quality Factor: Momentum / Volatility
    # This finds 'smooth' gainers which the Equal-Weight portfolio can't distinguish
    df["quality_mom"] = df["mom_63"] / (df["vol_63"] + 1e-6)
    
    # 4. Cross-Sectional Z-Score
    # We rank stocks against each other for that specific day
    df["pred_signal"] = df.groupby("date")["quality_mom"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )
    
    return df
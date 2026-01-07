import pandas as pd
import pandas_ta as ta

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['ticker', 'date'])
    
    # 1. 9-period RSI
    df['rsi'] = df.groupby('ticker')['adj_close'].transform(lambda x: ta.rsi(x, length=9))
    
    # 2. MACD (Standard 12, 26, 9)
    # This returns a DataFrame with: MACD_12_26_9, MACDs_12_26_9 (signal), MACDh_12_26_9 (hist)
    def get_macd(x):
        return ta.macd(x, fast=12, slow=26, signal=9)

    macd_df = df.groupby('ticker')['adj_close'].apply(lambda x: get_macd(x)).reset_index(level=0, drop=True)
    df = pd.concat([df, macd_df], axis=1)
    
    # Rename columns for clarity (pandas_ta uses specific naming)
    df = df.rename(columns={
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal'
    })

    # 3. Combined Signal Logic
    df['signal'] = 0
    
    # ENTRY: RSI > 70 AND MACD is above its Signal line (Positive Momentum)
    entry_condition = (df['rsi'] > 70) & (df['macd'] > df['macd_signal'])
    df.loc[entry_condition, 'signal'] = 1
    
    # EXIT: RSI < 50 OR MACD crosses below Signal line (Trend Weakness)
    exit_condition = (df['rsi'] < 50) | (df['macd'] < df['macd_signal'])
    df.loc[exit_condition, 'signal'] = -1
    
    return df
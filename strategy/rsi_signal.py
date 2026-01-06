import pandas as pd
import pandas_ta as ta

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure data is sorted for TA calculation
    df = df.sort_values(['ticker', 'date'])
    
    # Calculate 9-period RSI
    df['rsi'] = df.groupby('ticker')['adj_close'].transform(lambda x: ta.rsi(x, length=9))
    
    # Signal Logic: 
    # 1 = Potential Buy (RSI > 70)
    # -1 = Potential Sell (RSI < 50)
    df['signal'] = 0
    df.loc[df['rsi'] > 70, 'signal'] = 1
    df.loc[df['rsi'] < 50, 'signal'] = -1
    
    return df
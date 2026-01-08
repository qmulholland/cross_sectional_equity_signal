import pandas as pd
import pandas_ta as ta

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    
    # Ensures chronological order within ticker group for accurate rolling calculations
    df = df.sort_values(['ticker', 'date'])
    
    # Calculates a 9-period Relative Strength Index (RSI)
    # Uses transform to map the vectorized RSI calculation back to the original data frame index
    df['rsi'] = df.groupby('ticker')['adj_close'].transform(lambda x: ta.rsi(x, length=9))
    
    # Calculates MACD (Standard 12, 26, 9 Parameters)
    # The MACD provides trend direction, while signal line acts as a smoothed moving average
    def get_macd(x):
        return ta.macd(x, fast=12, slow=26, signal=9)

    # Apply MACD calculation cross-sectionally across all ticker groups
    macd_df = df.groupby('ticker')['adj_close'].apply(lambda x: get_macd(x)).reset_index(level=0, drop=True)

    # Concatenate the resulting MACD features (MACD line, Signal Line, and Histogram) to the main dataset
    df = pd.concat([df, macd_df], axis=1)
    
    # Standardizes column naming for cleaner downstream signal logic
    df = df.rename(columns={
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal'
    })

    # Initializes a neutral state
    df['signal'] = 0
    
    # ENTRY LOGIC: Define high-conviction bullish momentum
    # Condition: RSI indicates strong buying pressure (>70) and MACD confirms upward trend
    entry_condition = (df['rsi'] > 70) & (df['macd'] > df['macd_signal'])
    df.loc[entry_condition, 'signal'] = 1
    
    # EXIT LOGIC: Identify momentum exhaustion or trend reversal
    # Condition: RSI mean-reverts (<50) or MACD indicates a bearish crossover
    exit_condition = (df['rsi'] < 50) | (df['macd'] < df['macd_signal'])
    df.loc[exit_condition, 'signal'] = -1
    
    # Returns the dataframe
    return df
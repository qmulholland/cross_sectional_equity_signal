"""
Currently acts as the  'engine' of the strategy by implementing a market-neutral,
long-short ranking system. It isolates highest and lowest signal performers each day
to capture the spread between the two groups while maintaining equal-weighted
exposure within each leg (long/short). 

In the future, it can be expanded to support flexible sizing or multi-factor ranking
logic. This would allow for testing differnt strategies, such as quintile spreads
(top 20% vs bottom 20%) or weighting portions based on the specific strength
of the signals.
"""
import pandas as pd     #imports pandas 


def backtest_decile(        #defines function that runs cross-sectional decile backtest
    df: pd.DataFrame,       #DataFrame, multiple assets across time
    signal_col: str,        #Column name of cross-sectional signal (factor)
    ret_col: str            #Column name of the forward return used to compute PnL
) -> pd.Series:             #Returns pandas Series of daily long-short portfolio returns, date-indexed

    """
    Long top decile, short bottom decile, equally weighted.
    Plain text: 'Go long on top 10% of asssets by signal', 'Go short on bottom 10%', 'Equal assets within each leg'
    """

    
    df = df.dropna(subset=[signal_col, ret_col]).copy()     #Removes where signal or forward return is missing.
                                                            #Prevents invalid assignments and look-ahead bias
                                                            #Copied to avoid mutating original DataFrame

    df["decile"] = (                                            #Ranking logic of Strategy
        df.groupby("date")[signal_col]                          #Rank assets by signal_col
        .transform(lambda x: pd.qcut(x, 10, labels=False))      #Assign each asset to one of 10 quantiles (deciles)
    )                                                           # 0: Lowest signal values
                                                                # 9: Highest signal values
                                                                #Transform ensures decile labels align row-by-row with original DataFrame
                                                                #Implements daily cross-sectional ranking

    long = df[df["decile"] == 9]        #If asset in top decile, go long
    short = df[df["decile"] == 0]       #If asset in bottom decile, go short

    daily_pnl = (                                   #Constructs daily long-short return series
        long.groupby("date")[ret_col].mean()        #Compute the equal-weight average return of the long leg each day
        - short.groupby("date")[ret_col].mean()     #Compute the equal-weight average return of the short leg each day
    )                                               #Subtracts short from long to obtain the portfolio's return
                                                    #Produces market-neutral factor return

    return daily_pnl            #Returns the time series of daily long-short returns
                                #Can be used for:
                                #Performance evaluation, Cumulative return plots, Sharpe Ratio Computation, IS/OOS analysis

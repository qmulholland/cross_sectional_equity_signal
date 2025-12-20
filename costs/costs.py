"""
Currently acts as accounting layer that transforms theorhetical market
returns into realistic net returns. By deducting a fixed cost per trade
it prevents the backtest from overestimating the strategy's 
profitability.

In the future, it can be upgraded to include dynamic slippage or
variable commission structures depending on the platform. This
would make the simulation more accurate on how larger/smaller
orders impact market prices and change margins.
"""

def apply_transaction_costs(returns, cost_bps=5):       #Defines function to adjust trading returns by subtracting fixed fee

    cost = cost_bps / 10000     #Converts basis points (bps) into decimal format by dividing by 10,000
    return returns - cost       #Subtracts decimal cost from the daily return series to get the net PnL

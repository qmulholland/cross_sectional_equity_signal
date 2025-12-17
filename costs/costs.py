"""
Transaction Costs & Slippage module for cross_sectional_equity_signal.

Includes:
- Flat per-trade costs
- Percentage-of-notional costs
- Simple slippage models
"""

import pandas as pd


def apply_transaction_costs(df: pd.DataFrame, weight_col='weight', cost_per_trade=0.0005) -> pd.DataFrame:
    """
    Apply transaction costs to a DataFrame with position weights.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'date', 'ticker', and weight_col
    weight_col : str
        Column name for portfolio weights
    cost_per_trade : float
        Cost per trade as fraction of notional (e.g., 5 bps = 0.0005)

    Returns
    -------
    pd.DataFrame
        Same df with new column 'cost' and adjusted PnL if applicable
    """
    df = df.copy()

    # Compute change in position (assuming rebalancing daily)
    df['trade'] = df.groupby('ticker')[weight_col].diff().fillna(df[weight_col])

    # Absolute value = amount traded
    df['trade_abs'] = df['trade'].abs()

    # Cost per trade
    df['cost'] = df['trade_abs'] * cost_per_trade

    return df


def apply_slippage(df: pd.DataFrame, slippage_bps=0.0002, price_col='close') -> pd.DataFrame:
    """
    Apply simple slippage to returns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ticker' and price_col
    slippage_bps : float
        Slippage per trade in fraction (e.g., 2 bps = 0.0002)
    price_col : str
        Column name for price to calculate impact (optional)

    Returns
    -------
    pd.DataFrame
        DataFrame with new column 'slippage' representing slippage cost
    """
    df = df.copy()

    # Slippage as a fixed fraction of notional (simplest model)
    df['slippage'] = slippage_bps * df['close']

    return df

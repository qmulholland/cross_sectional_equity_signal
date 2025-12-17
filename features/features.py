"""
Feature Aggregator Module

Purpose:
- Provide a single interface for computing all features
- Can be extended later for additional features (fundamental, sentiment, etc.)
- Avoids cluttering main scripts with multiple imports
"""

import pandas as pd
from .technical import (
    compute_daily_returns,
    compute_momentum,
    compute_volatility,
    cross_sectional_zscore,
    generate_signal
)


def compute_all_features(prices: pd.DataFrame, momentum_windows=None, volatility_windows=None) -> pd.DataFrame:
    """
    Compute the full technical feature set including:
    - Daily returns
    - Rolling momentum
    - Rolling volatility
    - Cross-sectional z-score normalization

    Parameters
    ----------
    prices : pd.DataFrame
        Must contain OHLCV data in long format: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    momentum_windows : list[int], optional
        Rolling windows for momentum calculation, default [5, 10, 21]
    volatility_windows : list[int], optional
        Rolling windows for volatility calculation, default [5, 10, 21]

    Returns
    -------
    pd.DataFrame
        DataFrame with all features added
    """
    if momentum_windows is None:
        momentum_windows = [5, 10, 21]
    if volatility_windows is None:
        volatility_windows = [5, 10, 21]

    df = prices.copy()

    # Step 1: Daily returns
    df = compute_daily_returns(df)

    # Step 2: Momentum
    df = compute_momentum(df, windows=momentum_windows)

    # Step 3: Volatility
    df = compute_volatility(df, windows=volatility_windows)

    # Step 4: Cross-sectional z-scoring for all new features
    feature_cols = [f"mom_{w}" for w in momentum_windows] + [f"vol_{w}" for w in volatility_windows]
    df = cross_sectional_zscore(df, feature_cols)

    # Step 5: Generate composite signal
    z_cols = [f"{col}_z" for col in feature_cols]
    df = generate_signal(df, feature_cols=z_cols)

    return df

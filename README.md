**Quantitative Backtesting Engine**
-A Modular Python Framework for Equity Research
        -This project is a custom-built Backtesting 
        Engine designed to process financial
        datasets and simulate trading performance.
        The primary objective was to engineer a 
        technically sound "backtester" that accounts
        for data biases and liquidity constraints,
        rather than focusing on finding a "winning"
        strategy.

**The Backtest Engine Architecture**
-My priority was to create a modular system where data
ingestion, signal processing, and performance attribution
are cleanly separated. This ensures the code is 
reproducible and easy to debug.

    - Vectorized Computation Layer
        -Instead of using slow "for-loops" to simulate 
        trading day-by-day (Iterative), this engine was
        built to be Vectorized. This means it treats the
        entire four-year dataset as a series of mathematical
        arrays (vectors).

        -Efficiency: By using Pandas and NumPy, the engine
        can calculate PnL for thousands of data points
        simultaneously in milliseconds

        -Bias Prevention: I utilized .shift(1) logic on the
        signal vector to ensure the engine doesn't "cheat" by
        buying at a price it couldn't have known yet (Look-Ahead
        Bias)

    -Cross-Sectional Liquidity Filter
        -To keep the simulation realistic, the system dynamically
        ranks the provided ticker universe by daily trading volume. Only the Top 25 most liquid assets within that universe are eligible for trade signals that day. This means that the strategy only trades with high volume, which makes technical indicators (RSI/MACD) more reliable.

    -Modular Pipeline Design
        -loader.py: A dedicated layer for data ingestion
        and price adjustment via Yahoo Finance API
        -backtest.py: The mathematical core where volume ranking and portfolio returns are calculated using vectorized operations
        -output.py: Performance attribution that calculates the Sharpe Ratio, Maximum Drawdown, and generates a combined growth chart against the SPY benchmark.

**Data Granularity and Engineering Constraints**
-The Granularity Challenge
    -This project currently operates on Daily (1d) data. A significant future engineering goal is to move from daily bars to Intraday data.
-Scaling
    -Processing second-by-second data would require moving from this memory-based vectorized approach to an "Event-Driven" architecture to handle the massive increase in data volume.

**Setup and Usage**
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the engine: python main.py
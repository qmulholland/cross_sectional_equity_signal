from engine.loader import load_data
from engine.backtest import run_backtest
from engine.output import generate_performance_report
from strategy.rsi_signal import apply_strategy
import pandas as pd

def main():
    # Define start date here
    START_DATE = "2020-01-01"
    # Extract the year for the report header
    start_year = START_DATE.split("-")[0]
    
    # 1. Universe
    tickers = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", 
               "NFLX", "INTC", "PYPL", "AVGO", "SPY"] 
    
    # 2. Load Data
    print(f"Loading data since {start_year}:")
    raw_data = load_data(tickers, start_date=START_DATE)
    
    # 3. Apply Strategy
    processed_data = apply_strategy(raw_data)
    
    # 4. Filter Strategy Assets
    strat_data = processed_data[processed_data['ticker'] != 'SPY'].copy()
    
    # 5. Run Combined Backtest
    print("Running Backtest:")
    strat_returns = run_backtest(strat_data)
    
    # 6. Extract SPY Benchmark Returns
    spy_df = processed_data[processed_data['ticker'] == 'SPY'].copy()
    spy_returns = spy_df.set_index('date')['adj_close'].pct_change().fillna(0)
    
    # Align SPY returns to the strategy timeline
    spy_returns = spy_returns.reindex(strat_returns.index).fillna(0)
    
    # 7. Generate Combined Report (Passing start_year)
    print("Generating Combined Performance Report:")
    generate_performance_report(strat_returns, spy_returns, start_year)

if __name__ == "__main__":
    main()
from engine.loader import load_data
from engine.backtest import run_backtest
from engine.output import generate_performance_report
from strategy.rsi_signal import apply_strategy

def main():
    # 1. Define Universe (Expanded list to ensure Top 25 volume filter has enough data)
    tickers = ["AAPL", "TSLA", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META", 
               "NFLX", "INTC", "PYPL", "AVGO"] 
    
    # 2. Load Data (Matches your loader.py: load_data(tickers, start_date))
    print("Loading data:")
    raw_data = load_data(tickers, start_date="2021-01-01")
    
    # 3. Apply RSI 9 Strategy Logic
    print("Calculating RSI Signals:")
    processed_data = apply_strategy(raw_data)
    
    # 4. Split Data & Pre-calculate Returns for the Output Engine
    split_date = "2024-01-01"
    is_data = processed_data[processed_data['date'] < split_date].copy()
    oos_data = processed_data[processed_data['date'] >= split_date].copy()
    
    # FIX: Add 1-day returns column required by engine/output.py
    # We apply this to oos_data as it is used for the benchmark plotting
    oos_data['ret_1d'] = oos_data.groupby('ticker')['adj_close'].pct_change().fillna(0)
    
    # 5. Run Signal-Driven Backtest (Simulates trades based on $1,000 start)
    print("Running Backtest Simulations...")
    strat_returns_is = run_backtest(is_data)
    strat_returns_oos = run_backtest(oos_data)
    
    # 6. Create Benchmark Returns (Average of all tickers)
    # This satisfies the 'spy_returns' positional argument in generate_performance_report
    spy_returns = oos_data.groupby('date')['ret_1d'].mean().fillna(0)
    
    # 7. Generate Output Report
    # Passing the 4 required arguments: IS PnL, OOS PnL, Raw Data, and Benchmark PnL
    print("Generating Performance Report...")
    generate_performance_report(
        strat_returns_is, 
        strat_returns_oos, 
        oos_data,
        spy_returns
    )

if __name__ == "__main__":
    main()
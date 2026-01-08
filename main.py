from engine.loader import load_data
from engine.backtest import run_backtest
from engine.output import generate_performance_report
from strategy.rsi_and_macd_signals import apply_strategy
import pandas as pd

def main():

    # Defines start point for the backtest
    START_DATE = "2020-01-01"

    # Extracts year for labeling in the final performance report
    start_year = START_DATE.split("-")[0]
    
    # Defines ticker 'universe'
    # NEED to keep 'SPY' in, everything else can be changed
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                'META', 'TSLA', 'BRK-B', 'V', 'UNH', 'JPM', 
                'JNJ', 'WMT', 'MA', 'PG', 'HD', 'LLY', 'AVGO', 
                'ORCL', 'CVX', 'COST', 'PEP', 'ABBV', 'KO', 
                'ADBE', 'WFC', 'BAC', 'CSCO', 'CRM', 'TMO', 
                'PFE', 'NFLX', 'DHR', 'ACN', 'AMD', 'LIN', 
                'ABT', 'DIS', 'TXN', 'PM', 'INTC', 'HON', 'VZ', 
                'AMGN', 'INTU', 'IBM', 'GE', 'CAT', 'UBER', 
                'QCOM', 'SPY']
    
    # Ingests historical data using the modular loader engine
    print(f"Loading data since {start_year}:")
    raw_data = load_data(tickers, start_date=START_DATE)
    
    # Feature engineering and signal generation via strategy module
    processed_data = apply_strategy(raw_data)
    
    # Isolates the equities from the benchmark to run the strategy simulation
    strat_data = processed_data[processed_data['ticker'] != 'SPY'].copy()
    
    # Executes the vectorized backtesting engine to calculate daily strategy returns
    print("Running Backtest:")
    strat_returns = run_backtest(strat_data)
    
    # Prepare benchmark data: Calculate daily percentage returns for SPY
    spy_df = processed_data[processed_data['ticker'] == 'SPY'].copy()
    spy_returns = spy_df.set_index('date')['adj_close'].pct_change().fillna(0)
    
    # Align benchmark indicies with strategy results to ensure "apples-to-apples"
    spy_returns = spy_returns.reindex(strat_returns.index).fillna(0)
    
    # Visualize results and calculate risk metrics (Sharpe Ratio, Drawdown, etc.)
    print("Generating Combined Performance Report:")
    generate_performance_report(strat_returns, spy_returns, start_year)

if __name__ == "__main__":
    # Standard Python entry point to execute main function
    main()
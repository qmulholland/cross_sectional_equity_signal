import pandas as pd

def run_backtest(df: pd.DataFrame, initial_capital=1000):
    df = df.sort_values(['date', 'ticker'])
    dates = df['date'].unique()
    
    cash = initial_capital
    portfolio = {} # {ticker: shares}
    daily_equity = []

    for date in dates:
        day_data = df[df['date'] == date].copy()
        
        # A. Filter Universe: Top 25 by Volume
        top_25 = day_data.nlargest(25, 'volume')['ticker'].tolist()
        
        # B. Check for Sells (Exit if Signal is -1)
        to_sell = []
        for ticker, shares in portfolio.items():
            stock_row = day_data[day_data['ticker'] == ticker]
            if not stock_row.empty:
                if stock_row['signal'].values[0] == -1:
                    cash += shares * stock_row['adj_close'].values[0]
                    to_sell.append(ticker)
        for t in to_sell: del portfolio[t]

        # C. Check for Buys (Buy if Signal is 1 and in Top 25)
        # Dynamic slot sizing based on current total wealth / 25
        current_mkt_val = sum(shares * day_data[day_data['ticker'] == t]['adj_close'].values[0] 
                              for t, shares in portfolio.items() if not day_data[day_data['ticker'] == t].empty)
        total_equity = cash + current_mkt_val
        slot_size = total_equity / 25

        for ticker in top_25:
            if ticker not in portfolio and len(portfolio) < 25:
                stock_row = day_data[day_data['ticker'] == ticker]
                if not stock_row.empty and stock_row['signal'].values[0] == 1:
                    price = stock_row['adj_close'].values[0]
                    if cash >= slot_size:
                        shares_to_buy = slot_size / price
                        cash -= shares_to_buy * price
                        portfolio[ticker] = shares_to_buy

        # D. Record Total Value
        daily_equity.append({'date': date, 'total_value': total_equity})

    # Convert to daily returns series for output.py
    results_df = pd.DataFrame(daily_equity).set_index('date')
    return results_df['total_value'].pct_change().fillna(0)
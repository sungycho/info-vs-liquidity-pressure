"""
TAQ data loader with basic cleaning
"""
import wrds
import pandas as pd
from typing import Optional


def load_trades(
    ticker: str,
    date: str,  # 'YYYY-MM-DD'
    start_time: str = "09:30:00",
    end_time: str = "16:00:00"
) -> pd.DataFrame:
    """
    Load and clean TAQ trades for one ticker-date.
    
    Returns:
        DataFrame with: timestamp, ticker, price, size, direction (placeholder)
    """
    
    db = wrds.Connection()
    
    # Convert date format for table name
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        price,
        size
    FROM taqm_{year}.ctm_{date_str}
    WHERE sym_root = '{ticker}'
        AND time_m BETWEEN '{start_time}' AND '{end_time}'
        AND price > 0
        AND size > 0
    ORDER BY time_m
    LIMIT 100000
    """
    
    try:
        df = db.raw_sql(query)
        db.close()
        
        if len(df) == 0:
            print(f"Warning: No trades found for {ticker} on {date}")
            return pd.DataFrame()
        
        # Convert timestamp to proper format
        df['timestamp'] = pd.to_datetime(date + ' ' + df['timestamp'].astype(str))
        
        print(f"✓ Loaded {len(df):,} trades for {ticker} on {date}")
        return df
        
    except Exception as e:
        db.close()
        print(f"Error loading {ticker} on {date}: {e}")
        return pd.DataFrame()


def load_quotes(
    ticker: str,
    date: str,
    start_time: str = "09:30:00",
    end_time: str = "10:30:00"
) -> pd.DataFrame:
    """
    Load and clean TAQ quotes for one ticker-date.
    
    Returns:
        DataFrame with: timestamp, ticker, bid, ask, midpoint, spread
    """
    
    db = wrds.Connection()
    
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        bid,
        ask
    FROM taqm_{year}.cqm_{date_str}
    WHERE sym_root = '{ticker}'
        AND time_m BETWEEN '{start_time}' AND '{end_time}'
        AND bid > 0
        AND ask > 0
        AND ask > bid
    ORDER BY time_m
    LIMIT 100000
    """
    
    try:
        df = db.raw_sql(query)
        db.close()
        
        if len(df) == 0:
            print(f"Warning: No quotes found for {ticker} on {date}")
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(date + ' ' + df['timestamp'].astype(str))
        df['midpoint'] = (df['bid'] + df['ask']) / 2
        df['spread'] = (df['ask'] - df['bid']) / df['midpoint']
        
        print(f"✓ Loaded {len(df):,} quotes for {ticker} on {date}")
        return df
        
    except Exception as e:
        db.close()
        print(f"Error loading quotes for {ticker} on {date}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test
    trades = load_trades("AAPL", "2023-05-03")
    print("\nSample trades:")
    print(trades.head())
    
    quotes = load_quotes("AAPL", "2023-05-03")
    print("\nSample quotes:")
    print(quotes.head())
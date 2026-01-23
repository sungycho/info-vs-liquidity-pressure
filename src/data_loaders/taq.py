"""
Research-grade TAQ loader - SQLAlchemy 2.0 compatible
"""
import wrds
import pandas as pd
import logging
from typing import Optional
from sqlalchemy import text

logger = logging.getLogger(__name__)


def load_trades(
    permno: int,
    date: str,
    db: wrds.Connection,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00"
) -> pd.DataFrame:
    """Load TAQ trades for one permno-date with research-grade cleaning."""
    
    # Step 1: Get ticker
    ticker_query = text(f"""
    SELECT ticker
    FROM crsp.stocknames
    WHERE permno = {permno}
        AND :date >= namedt 
        AND :date <= nameenddt
    LIMIT 1
    """)
    
    try:
        result = db.connection.execute(ticker_query, {"date": date})
        row = result.fetchone()
        
        if row is None:
            logger.warning(f"No ticker mapping for permno {permno} on {date}")
            return pd.DataFrame()
        
        ticker = row[0]
        
    except Exception as e:
        logger.error(f"Error getting ticker for permno {permno}: {e}")
        return pd.DataFrame()
    
    # Step 2: Load TAQ trades
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = text(f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        price,
        size,
        ex as exchange
    FROM taqm_{year}.ctm_{date_str}
    WHERE sym_root = :ticker
        AND time_m >= :start_time
        AND time_m <= :end_time
        AND price > 0
        AND size > 0
        AND tr_corr = '00'
        AND ex IN ('N', 'Q', 'A', 'P', 'Z')
    ORDER BY time_m
    """)
    
    try:
        result = db.connection.execute(query, {
            "ticker": ticker,
            "start_time": start_time,
            "end_time": end_time
        })
        rows = result.fetchall()
        
        if len(rows) == 0:
            logger.warning(f"No trades for permno {permno} ({ticker}) on {date}")
            return pd.DataFrame()
        
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp - handle time objects properly
        df['timestamp'] = df['timestamp'].astype(str)
        base_date = pd.to_datetime(date).tz_localize('America/New_York')
        df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'])
        
        df['permno'] = permno
        df['date'] = pd.to_datetime(date)
        
        logger.info(f"Loaded {len(df):,} trades for permno {permno} on {date}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading trades for permno {permno} on {date}: {e}")
        return pd.DataFrame()


def load_quotes(
    permno: int,
    date: str,
    db: wrds.Connection,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00"
) -> pd.DataFrame:
    """Load TAQ quotes for one permno-date with research-grade cleaning."""
    
    # Get ticker
    ticker_query = text(f"""
    SELECT ticker
    FROM crsp.stocknames
    WHERE permno = {permno}
        AND :date >= namedt 
        AND :date <= nameenddt
    LIMIT 1
    """)
    
    try:
        result = db.connection.execute(ticker_query, {"date": date})
        row = result.fetchone()
        
        if row is None:
            logger.warning(f"No ticker for permno {permno} on {date}")
            return pd.DataFrame()
        
        ticker = row[0]
        
    except Exception as e:
        logger.error(f"Error getting ticker: {e}")
        return pd.DataFrame()
    
    # Load quotes
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = text(f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        bid,
        ask,
        ex as exchange
    FROM taqm_{year}.cqm_{date_str}
    WHERE sym_root = :ticker
        AND time_m >= :start_time
        AND time_m <= :end_time
        AND bid > 0
        AND ask > 0
        AND ask > bid
        AND ex IN ('N', 'Q', 'A', 'P', 'Z')
    ORDER BY time_m
    """)
    
    try:
        result = db.connection.execute(query, {
            "ticker": ticker,
            "start_time": start_time,
            "end_time": end_time
        })
        rows = result.fetchall()
        
        if len(rows) == 0:
            logger.warning(f"No quotes for permno {permno} on {date}")
            return pd.DataFrame()
        
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)
        
        # Convert timestamp - handle time objects properly
        df['timestamp'] = df['timestamp'].astype(str)
        base_date = pd.to_datetime(date).tz_localize('America/New_York')
        df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'])
        
        # Calculate spread
        df['midpoint'] = (df['bid'] + df['ask']) / 2
        df['spread'] = (df['ask'] - df['bid']) / df['midpoint']
        
        # Remove extreme spreads
        n_before = len(df)
        df = df[df['spread'] < 0.50]
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} quotes with spread > 50%")
        
        df['permno'] = permno
        df['date'] = pd.to_datetime(date)
        
        logger.info(f"Loaded {len(df):,} quotes for permno {permno} on {date}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading quotes: {e}")
        return pd.DataFrame()


def load_trades_window(
    permno: int,
    start_date: str,
    end_date: str,
    db: wrds.Connection
) -> pd.DataFrame:
    """Load trades for an event window."""
    
    date_range = pd.bdate_range(start=start_date, end=end_date)
    all_trades = []
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        trades = load_trades(permno, date_str, db)
        
        if len(trades) > 0:
            all_trades.append(trades)
    
    if len(all_trades) == 0:
        logger.warning(f"No trades for permno {permno} in {start_date} to {end_date}")
        return pd.DataFrame()
    
    result = pd.concat(all_trades, ignore_index=True)
    logger.info(f"Loaded {len(result):,} trades across {len(date_range)} days")
    
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    db = wrds.Connection()
    
    try:
        print("Testing single-day load...")
        trades = load_trades(14593, "2023-05-03", db)
        print(f"\nTrades shape: {trades.shape}")
        if len(trades) > 0:
            print(trades.head())
        
        print("\nTesting quotes load...")
        quotes = load_quotes(14593, "2023-05-03", db)
        print(f"\nQuotes shape: {quotes.shape}")
        if len(quotes) > 0:
            print(quotes.head())
        
    finally:
        db.close()
        print("\n✓ Test complete")
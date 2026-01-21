"""
Research-grade TAQ loader with proper cleaning and permno-based access

Design principles:
- Permno-based (handles ticker changes, share classes)
- Connection reuse (external management)
- No artificial limits (complete data)
- Conservative cleaning (defensible in research)
- Production logging (thread-safe, level-controlled)

Cleaning contract:
- Regular hours only (09:30-16:00 ET)
- price > 0, size > 0
- tr_corr = '00' (no corrections)
- Valid exchanges (NYSE, NASDAQ, AMEX, Arca, BATS)
- Spread < 50% (remove data glitches only)
"""
import wrds
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_trades(
    permno: int,
    date: str,
    db: wrds.Connection,
    start_time: str = "09:30:00",
    end_time: str = "16:00:00"
) -> pd.DataFrame:
    """
    Load TAQ trades for one permno-date with research-grade cleaning.
    
    Args:
        permno: CRSP permanent identifier
        date: Trading date 'YYYY-MM-DD'
        db: Active WRDS connection (caller manages lifecycle)
        start_time: Session start in 'HH:MM:SS' (default market open)
        end_time: Session end in 'HH:MM:SS' (default market close)
        
    Returns:
        DataFrame with columns: timestamp, permno, ticker, price, size, exchange, date
        Empty DataFrame if no data (does NOT raise exception)
        
    Note:
        - Ticker lookup adds ~100ms per call; consider caching for large batches
        - Returns timezone-aware timestamps (America/New_York)
    """
    
    # Step 1: Get ticker for this permno on this date
    ticker_query = f"""
    SELECT ticker
    FROM crsp.stocknames
    WHERE permno = {permno}
        AND '{date}' BETWEEN namedt AND nameenddt
    LIMIT 1
    """
    
    try:
        ticker_df = db.raw_sql(ticker_query)
        
        if len(ticker_df) == 0:
            logger.warning(f"No ticker mapping for permno {permno} on {date}")
            return pd.DataFrame()
        
        ticker = ticker_df.iloc[0]['ticker']
        
    except Exception as e:
        logger.error(f"Error getting ticker for permno {permno}: {e}")
        return pd.DataFrame()
    
    # Step 2: Load TAQ trades
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        price,
        size,
        ex as exchange
    FROM taqm_{year}.ctm_{date_str}
    WHERE sym_root = '{ticker}'
        AND time_m BETWEEN '{start_time}' AND '{end_time}'
        AND price > 0
        AND size > 0
        AND tr_corr = '00'  -- No corrections (minimal cleaning to avoid overfitting)
        AND ex IN ('N', 'Q', 'A', 'P', 'Z')  -- Conservative filter for S&P 500 focus
    ORDER BY time_m
    """
    
    try:
        df = db.raw_sql(query)
        
        if len(df) == 0:
            logger.warning(f"No trades for permno {permno} ({ticker}) on {date}")
            return pd.DataFrame()
        
        # Convert timestamp with timezone awareness
        base_date = pd.to_datetime(date).tz_localize('America/New_York')
        df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'])
        
        # Add identifiers for joining
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
    """
    Load TAQ quotes for one permno-date with research-grade cleaning.
    
    Args:
        permno: CRSP permanent identifier
        date: Trading date 'YYYY-MM-DD'
        db: Active WRDS connection
        start_time: Session start in 'HH:MM:SS'
        end_time: Session end in 'HH:MM:SS'
        
    Returns:
        DataFrame with: timestamp, permno, ticker, bid, ask, midpoint, spread, exchange, date
        Empty DataFrame if no data
        
    Note:
        - Spread threshold (50%) removes data glitches, not liquidity stress
        - Lower thresholds (10-20%) risk removing legitimate illiquidity episodes
    """
    
    # Get ticker
    ticker_query = f"""
    SELECT ticker
    FROM crsp.stocknames
    WHERE permno = {permno}
        AND '{date}' BETWEEN namedt AND nameenddt
    LIMIT 1
    """
    
    try:
        ticker_df = db.raw_sql(ticker_query)
        if len(ticker_df) == 0:
            logger.warning(f"No ticker for permno {permno} on {date}")
            return pd.DataFrame()
        ticker = ticker_df.iloc[0]['ticker']
    except Exception as e:
        logger.error(f"Error getting ticker: {e}")
        return pd.DataFrame()
    
    # Load quotes
    date_str = pd.to_datetime(date).strftime('%Y%m%d')
    year = pd.to_datetime(date).year
    
    query = f"""
    SELECT 
        time_m as timestamp,
        sym_root as ticker,
        bid,
        ask,
        ex as exchange
    FROM taqm_{year}.cqm_{date_str}
    WHERE sym_root = '{ticker}'
        AND time_m BETWEEN '{start_time}' AND '{end_time}'
        AND bid > 0
        AND ask > 0
        AND ask > bid
        AND ex IN ('N', 'Q', 'A', 'P', 'Z')
    ORDER BY time_m
    """
    
    try:
        df = db.raw_sql(query)
        
        if len(df) == 0:
            logger.warning(f"No quotes for permno {permno} on {date}")
            return pd.DataFrame()
        
        # Convert timestamp
        base_date = pd.to_datetime(date).tz_localize('America/New_York')
        df['timestamp'] = base_date + pd.to_timedelta(df['timestamp'])
        
        # Calculate spread
        df['midpoint'] = (df['bid'] + df['ask']) / 2
        df['spread'] = (df['ask'] - df['bid']) / df['midpoint']
        
        # Remove extreme spreads (>50% = data glitches, not liquidity stress)
        n_before = len(df)
        df = df[df['spread'] < 0.50]
        n_removed = n_before - len(df)
        
        if n_removed > 0:
            logger.debug(f"Removed {n_removed} quotes with spread > 50% for permno {permno}")
        
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
    """
    Load trades for an event window (e.g., [-10, -1] pre-earnings).
    
    Args:
        permno: CRSP permanent identifier
        start_date: First date 'YYYY-MM-DD'
        end_date: Last date 'YYYY-MM-DD'
        db: Active WRDS connection
        
    Returns:
        DataFrame with all trades concatenated across dates
        Empty DataFrame if no data found
        
    Usage:
        # Load pre-earnings window
        trades = load_trades_window(14593, '2023-04-20', '2023-05-02', db)
        daily_ofi = trades.groupby('date').apply(calculate_ofi)
        
    Memory note:
        For high-volume stocks (500K trades/day × 10 days), aggregate to daily
        features ASAP after loading to free memory.
    """
    
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
    logger.info(f"Loaded {len(result):,} trades for permno {permno} across {len(date_range)} days")
    
    return result


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with AAPL permno
    db = wrds.Connection()
    
    try:
        # AAPL = permno 14593
        print("Testing single-day load...")
        trades = load_trades(14593, "2023-05-03", db)
        print(f"\nTrades shape: {trades.shape}")
        print(trades.head())
        
        print("\nTesting quotes load...")
        quotes = load_quotes(14593, "2023-05-03", db)
        print(f"\nQuotes shape: {quotes.shape}")
        print(quotes.head())
        
        print("\nTesting window load...")
        window = load_trades_window(14593, "2023-05-01", "2023-05-03", db)
        print(f"\nWindow trades shape: {window.shape}")
        print(window.groupby('date').size())
        
    finally:
        db.close()
        print("\n✓ Test complete")
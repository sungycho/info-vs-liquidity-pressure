"""
Daily feature pipeline: Earnings events → TAQ data → Microstructure features

Purpose:
    Orchestrate loading of TAQ data and computation of daily features
    for earnings event windows. Does NOT contain feature calculation logic.

Responsibility:
    - Load event table
    - Generate date windows for each event
    - Load TAQ data (trades/quotes) per (permno, date)
    - Call feature modules (order_flow, volume, liquidity)
    - Merge results into flat dict
    - Save to parquet

NOT responsible for:
    - Feature calculation logic (delegated to feature modules)
    - ADV calculation (Phase 2)
    - Research interpretation (downstream)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import wrds
import logging
from typing import List, Dict
import pandas_market_calendars as mcal

from src.data_loaders.taq import load_trades, load_quotes
from src.features.order_flow import compute_daily_order_flow
from src.features.volume import compute_daily_volume
from src.features.liquidity import compute_daily_liquidity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

# Test mode: reduce scope for quick validation
TEST_MODE = True
MAX_EVENTS = 5 if TEST_MODE else None

# Event window definition (Phase 1)
# Window length determined by WINDOW_START and WINDOW_END below
WINDOW_START = -2  # days before rdq
WINDOW_END = -1    # days before rdq

# File paths
EVENT_TABLE_PATH = "data/processed/event_table_2023_2024.parquet"
OUTPUT_PATH = "data/processed/daily_features_2023_2024.parquet"


# ============================================================
# Main Pipeline
# ============================================================

def build_daily_features(
    event_table_path: str = EVENT_TABLE_PATH,
    output_path: str = OUTPUT_PATH,
    test_mode: bool = TEST_MODE,
    max_events: int = MAX_EVENTS
):
    """
    Main orchestration function: Load events → compute features → save.
    
    Args:
        event_table_path: Path to earnings event table
        output_path: Path to save daily features
        test_mode: If True, limit scope for testing
        max_events: Max number of events to process (None = all)
    """
    
    logger.info("="*60)
    logger.info("Starting daily feature pipeline")
    logger.info("="*60)
    logger.info(f"Test mode: {test_mode}")
    logger.info(f"Max events: {max_events if max_events else 'all'}")
    logger.info(f"Event window: [{WINDOW_START}, {WINDOW_END}] days before rdq")
    
    # Step 1: Load event table
    logger.info(f"Loading event table from {event_table_path}")
    events_df = pd.read_parquet(event_table_path)
    logger.info(f"  Loaded {len(events_df)} events")
    
    # Test mode: limit events
    if test_mode and max_events:
        events_df = events_df.head(max_events)
        logger.info(f"  Test mode: Processing only {len(events_df)} events")
    
    # Step 2: Open WRDS connection (single connection for all queries)
    logger.info("Connecting to WRDS...")
    db = wrds.Connection()
    logger.info("  Connected successfully")
    
    # Step 3: Process events
    all_features = []
    
    try:
        for idx, event in events_df.iterrows():
            event_features = process_event(event, db)
            all_features.extend(event_features)
        
        # Step 4: Save results
        if len(all_features) > 0:
            save_features(all_features, output_path)
        else:
            logger.warning("No features computed, skipping save")
    
    finally:
        db.close()
        logger.info("WRDS connection closed")
    
    logger.info("="*60)
    logger.info("Pipeline complete")
    logger.info("="*60)


def process_event(event: pd.Series, db: wrds.Connection) -> List[Dict]:
    """
    Process a single earnings event: compute features for each day in window.
    
    Args:
        event: Event row with event_id, permno, rdq
        db: Active WRDS connection
        
    Returns:
        List of feature dicts, one per (event_id, permno, date)
    """
    
    event_id = event['event_id']
    permno = int(event['permno'])
    rdq = pd.to_datetime(event['rdq'])
    
    logger.info(f"Processing event {event_id}: permno={permno}, rdq={rdq.date()}")
    
    # Generate date window
    dates = generate_date_window(rdq, WINDOW_START, WINDOW_END)
    logger.info(f"  Date window: {[d.date() for d in dates]}")
    
    event_features = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            # Load TAQ data for this (permno, date)
            logger.debug(f"  Loading TAQ data for {date_str}")
            trades_df = load_trades(permno, date_str, db)
            quotes_df = load_quotes(permno, date_str, db)
            
            # Compute features
            features = compute_daily_features(
                event_id=event_id,
                permno=permno,
                date=date,
                trades_df=trades_df,
                quotes_df=quotes_df
            )
            
            event_features.append(features)
            logger.info(f"  ✓ Computed features for {date_str}")
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to process {date_str}: {e}")
            # Continue to next date (don't break entire event)
            continue
    
    logger.info(f"  Completed event {event_id}: {len(event_features)}/{len(dates)} days successful")
    
    return event_features


def generate_date_window(
    rdq: pd.Timestamp, 
    start_offset: int, 
    end_offset: int,
    use_crsp_calendar: bool = False,
    db = None
) -> List[pd.Timestamp]:
    """
    Generate trading date window using TRADING DAYS offset.
    
    Args:
        rdq: Earnings announcement date
        start_offset: Trading days before rdq (negative, e.g., -2)
        end_offset: Trading days before rdq (negative, e.g., -1)
        use_crsp_calendar: If True, use CRSP calendar (requires db connection)
        db: WRDS connection (required if use_crsp_calendar=True)
        
    Returns:
        List of trading dates in window
    """
    
    if use_crsp_calendar and db is not None:
        # Option A: CRSP calendar (most accurate for historical data)
        year = rdq.year
        query = f"""
        SELECT DISTINCT date 
        FROM crsp.dsi 
        WHERE date BETWEEN '{year-1}-01-01' AND '{year+1}-12-31'
        ORDER BY date
        """
        trading_days = pd.to_datetime(db.raw_sql(query)['date']).tolist()
    else:
        # Option B: NYSE calendar (faster, sufficient for 2023-2024)
        nyse = mcal.get_calendar('NYSE')
        buffer = 30
        start = rdq - pd.Timedelta(days=buffer)
        end = rdq + pd.Timedelta(days=5)
        
        schedule = nyse.schedule(start_date=start, end_date=end)
        trading_days = schedule.index.tolist()
    
    # Find rdq position in trading days
    try:
        rdq_idx = trading_days.index(rdq)
    except ValueError:
        logger.warning(f"RDQ {rdq.date()} not a trading day")
        return []
    
    # Calculate window using TRADING DAYS offset
    start_idx = rdq_idx + start_offset  # e.g., -2
    end_idx = rdq_idx + end_offset      # e.g., -1
    
    if start_idx < 0 or end_idx < 0:
        logger.warning(f"Insufficient history for {rdq.date()}")
        return []
    
    return trading_days[start_idx:end_idx + 1]


def compute_daily_features(
    event_id: str,
    permno: int,
    date: pd.Timestamp,
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame
) -> Dict:
    """
    Compute all daily features for a single (event_id, permno, date).
    
    Orchestrates calls to feature modules and merges results.
    
    Args:
        event_id: Event identifier
        permno: CRSP permno
        date: Trading date
        trades_df: From taq.load_trades()
        quotes_df: From taq.load_quotes()
        
    Returns:
        Flat dict with all features merged
    """
    
    # Base identifiers
    features = {
        'event_id': event_id,
        'permno': permno,
        'date': date
    }
    
    # Order flow features
    try:
        ofi_features = compute_daily_order_flow(trades_df, quotes_df)
        # Remove redundant permno/date (already in base)
        ofi_features.pop('permno', None)
        ofi_features.pop('date', None)
        features.update(ofi_features)
    except Exception as e:
        logger.warning(f"Order flow calculation failed: {e}")
        # Add NaN placeholders
        features.update({
            'ofi': np.nan,
            'ofi_abs': np.nan,
            'ofi_std': np.nan,
            'buy_volume': 0,
            'sell_volume': 0,
            'volume': 0,
            'num_trades': 0,
            'avg_trade_size': np.nan,
            'classification_method': 'failed'
        })
    
    # Volume features
    try:
        vol_features = compute_daily_volume(trades_df, adv_usd=None)
        # Remove redundant
        vol_features.pop('permno', None)
        vol_features.pop('date', None)
        features.update(vol_features)
    except Exception as e:
        logger.warning(f"Volume calculation failed: {e}")
        features.update({
            'dollar_volume': 0.0,
            'share_volume': 0,
            'abnormal_vol_ratio': np.nan,
            'morning_share': np.nan
        })
    
    # Liquidity features
    try:
        liq_features = compute_daily_liquidity(quotes_df)
        # Remove redundant
        liq_features.pop('permno', None)
        liq_features.pop('date', None)
        features.update(liq_features)
    except Exception as e:
        logger.warning(f"Liquidity calculation failed: {e}")
        features.update({
            'spread_mean': np.nan,
            'spread_std': np.nan,
            'spread_stability': np.nan,
            'quoted_spread_mean': np.nan,
            'spread_p95': np.nan,
            'num_quotes': 0,
            'num_quotes_1s': 0
        })
    
    return features


def save_features(features_list: List[Dict], output_path: str):
    """
    Save computed features to parquet.
    
    Args:
        features_list: List of feature dicts
        output_path: Output file path
    """
    
    logger.info(f"Saving {len(features_list)} feature rows to {output_path}")
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to parquet
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✓ Saved successfully")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {len(df.columns)}")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Events: {df['event_id'].nunique()}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    build_daily_features()
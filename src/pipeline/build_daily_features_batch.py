"""
Daily feature pipeline: Earnings events → TAQ data → Microstructure features

BATCH PROCESSING VERSION - FULL PRODUCTION
- Window: Exactly 10 NYSE TRADING days before each earnings announcement
- Uses pandas_market_calendars for 100% NYSE accuracy
- Automatically resumes from last completed event
- Safe to interrupt and restart
- Single output parquet with deduplication
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import wrds
import logging
from typing import List, Dict
import os
import gc
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

BATCH_SIZE = 20
START_IDX = 20

N_TRADING_DAYS = 10  # Exactly 10 NYSE trading days

EVENT_TABLE_PATH = "data/processed/event_table_2023_2024.parquet"
OUTPUT_PATH = "data/processed/daily_features_full.parquet"


# ============================================================
# NYSE Trading Day Calendar
# ============================================================

def generate_date_window(rdq: pd.Timestamp, n_days: int = N_TRADING_DAYS) -> List[pd.Timestamp]:
    """
    Generate EXACTLY n NYSE trading days before rdq.
    
    Uses pandas_market_calendars for 100% accurate NYSE calendar including:
    - Weekends
    - Federal holidays (New Year's, MLK Day, Presidents' Day, etc.)
    - NYSE-specific holidays (Good Friday)
    - Special closures
    
    Args:
        rdq: Earnings announcement date
        n_days: Number of trading days to go back (default: 10)
        
    Returns:
        List of exactly n_days NYSE trading dates before rdq
        
    Example:
        rdq = 2023-04-10 (Monday after Easter)
        n_days = 10
        → Automatically excludes:
          - Weekends
          - 2023-04-07 (Good Friday - NYSE closed)
          - Any other holidays in range
        → Returns exactly 10 actual NYSE trading days
    """
    
    # Get NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Generate schedule for a wide range before rdq
    # Use 60 calendar days back to ensure we capture 10+ trading days
    start_date = rdq - pd.Timedelta(days=60)
    end_date = rdq - pd.Timedelta(days=1)  # Exclude rdq itself
    
    # Get valid NYSE trading days
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    # Extract trading dates (schedule.index contains the trading days)
    trading_days = schedule.index.tolist()
    
    # Take exactly the last n_days
    if len(trading_days) < n_days:
        logger.warning(f"Not enough NYSE trading days before {rdq.date()}: "
                      f"got {len(trading_days)}, need {n_days}")
        return trading_days
    
    trading_days = trading_days[-n_days:]
    
    return trading_days


# ============================================================
# Batch-safe Pipeline
# ============================================================

def get_processed_events(output_path: str) -> set:
    """Load already-processed event_ids from existing parquet."""
    if not os.path.exists(output_path):
        logger.info("No existing output file found - starting fresh")
        return set()
    
    try:
        existing_df = pd.read_parquet(output_path)
        processed = set(existing_df['event_id'].unique())
        logger.info(f"Found {len(processed)} already-processed events")
        return processed
    except Exception as e:
        logger.warning(f"Could not load existing file: {e}")
        return set()


def save_batch_results(
    new_features: List[Dict],
    output_path: str,
    deduplicate: bool = True
):
    """Append new batch results to existing parquet."""
    if len(new_features) == 0:
        logger.warning("No new features to save")
        return
    
    new_df = pd.DataFrame(new_features)
    logger.info(f"Saving {len(new_df)} new feature rows")
    
    if os.path.exists(output_path):
        logger.info(f"Appending to existing file: {output_path}")
        old_df = pd.read_parquet(output_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        
        if deduplicate:
            n_before = len(combined)
            combined = combined.drop_duplicates(subset=['event_id', 'date'], keep='last')
            n_after = len(combined)
            if n_before > n_after:
                logger.info(f"Removed {n_before - n_after} duplicate rows")
        
        combined = combined.sort_values(['event_id', 'date']).reset_index(drop=True)
        combined.to_parquet(output_path, index=False)
        logger.info(f"✓ Saved successfully - Total rows: {len(combined)}")
        logger.info(f"  Unique events: {combined['event_id'].nunique()}")
    else:
        new_df = new_df.sort_values(['event_id', 'date']).reset_index(drop=True)
        new_df.to_parquet(output_path, index=False)
        logger.info(f"✓ Created new file - Total rows: {len(new_df)}")
        logger.info(f"  Unique events: {new_df['event_id'].nunique()}")


def build_daily_features_batch(
    event_table_path: str = EVENT_TABLE_PATH,
    output_path: str = OUTPUT_PATH,
    batch_size: int = BATCH_SIZE,
    start_idx: int = START_IDX
):
    """Build daily features for a batch of events."""
    
    logger.info("="*60)
    logger.info("FULL PRODUCTION - Daily Feature Pipeline")
    logger.info("="*60)
    logger.info(f"Batch size: {batch_size} events")
    logger.info(f"Start index: {start_idx}")
    logger.info(f"Window: Exactly {N_TRADING_DAYS} NYSE trading days")
    logger.info(f"Calendar: pandas_market_calendars (100% NYSE accurate)")
    
    all_events_df = pd.read_parquet(event_table_path)
    total_events = len(all_events_df)
    logger.info(f"  Total events available: {total_events}")
    
    processed_events = get_processed_events(output_path)
    
    end_idx = min(start_idx + batch_size, total_events)
    batch_events_df = all_events_df.iloc[start_idx:end_idx].copy()
    logger.info(f"  Batch range: [{start_idx}, {end_idx}) out of {total_events}")
    
    n_before = len(batch_events_df)
    batch_events_df = batch_events_df[~batch_events_df['event_id'].isin(processed_events)]
    n_after = len(batch_events_df)
    
    if n_before > n_after:
        logger.info(f"  Skipped {n_before - n_after} already-processed events")
    
    if len(batch_events_df) == 0:
        logger.info("  All events in this batch already processed - nothing to do")
        logger.info("  TIP: Increase START_IDX or BATCH_SIZE in config")
        return
    
    logger.info(f"  Processing {len(batch_events_df)} new events")
    
    logger.info("Connecting to WRDS...")
    db = wrds.Connection()
    logger.info("  ✓ Connected successfully")
    
    all_features = []
    
    try:
        for idx, event in batch_events_df.iterrows():
            try:
                event_features = process_event(event, db)
                all_features.extend(event_features)
                gc.collect()
            except Exception as e:
                logger.error(f"Failed to process event {event['event_id']}: {e}")
                continue
        
        if len(all_features) > 0:
            save_batch_results(all_features, output_path)
        else:
            logger.warning("No features computed in this batch")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Batch interrupted by user")
        logger.info("Saving partial results...")
        if len(all_features) > 0:
            save_batch_results(all_features, output_path)
        raise
    
    finally:
        db.close()
        logger.info("WRDS connection closed")
    
    logger.info("="*60)
    logger.info("Batch Complete")
    logger.info("="*60)
    logger.info(f"Events processed in this batch: {len(batch_events_df)}")
    logger.info(f"Feature rows generated: {len(all_features)}")
    
    if os.path.exists(output_path):
        final_df = pd.read_parquet(output_path)
        total_processed = final_df['event_id'].nunique()
        total_rows = len(final_df)
        progress_pct = (total_processed / total_events) * 100
        
        logger.info(f"\n📊 Overall Progress:")
        logger.info(f"  Events: {total_processed}/{total_events} ({progress_pct:.1f}%)")
        logger.info(f"  Rows: {total_rows:,}")
        
        if len(batch_events_df) > 0:
            remaining_events = total_events - total_processed
            estimated_batches = int(np.ceil(remaining_events / batch_size))
            logger.info(f"  Remaining batches: ~{estimated_batches}")
            
            next_start = start_idx + batch_size
            if next_start < total_events:
                logger.info(f"\n💡 Next run: Set START_IDX = {next_start}")


def process_event(event: pd.Series, db: wrds.Connection) -> List[Dict]:
    """Process a single earnings event."""
    
    event_id = event['event_id']
    permno = int(event['permno'])
    rdq = pd.to_datetime(event['rdq'])
    
    logger.info(f"Processing event {event_id}: permno={permno}, rdq={rdq.date()}")
    
    # Generate EXACTLY 10 NYSE trading days
    dates = generate_date_window(rdq, N_TRADING_DAYS)
    
    logger.info(f"  Window: {len(dates)} NYSE trading days (expected: {N_TRADING_DAYS})")
    if len(dates) < N_TRADING_DAYS:
        logger.warning(f"  ⚠ Only {len(dates)} days available (insufficient history)")
    
    logger.debug(f"  Dates: {[d.date() for d in dates]}")
    
    event_features = []
    successful_days = 0
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            trades_df = load_trades(permno, date_str, db)
            quotes_df = load_quotes(permno, date_str, db)
            
            if len(trades_df) == 0:
                logger.warning(f"  ✗ No trades for {date_str} (data gap)")
                continue
            
            features = compute_daily_features(
                event_id=event_id,
                permno=permno,
                date=date,
                trades_df=trades_df,
                quotes_df=quotes_df
            )
            
            event_features.append(features)
            successful_days += 1
            logger.debug(f"  ✓ {date_str}")
            
        except Exception as e:
            logger.warning(f"  ✗ Failed {date_str}: {str(e)[:100]}")
            continue
    
    logger.info(f"  Completed: {successful_days}/{len(dates)} days successful")
    
    return event_features


def compute_daily_features(
    event_id: str,
    permno: int,
    date: pd.Timestamp,
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame
) -> Dict:
    """Compute all daily features for a single (event_id, permno, date)."""
    
    features = {
        'event_id': event_id,
        'permno': permno,
        'date': date
    }
    
    # Order flow features
    try:
        ofi_features = compute_daily_order_flow(trades_df, quotes_df)
        ofi_features.pop('permno', None)
        ofi_features.pop('date', None)
        features.update(ofi_features)
    except Exception as e:
        logger.warning(f"Order flow calculation failed: {e}")
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


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    build_daily_features_batch()
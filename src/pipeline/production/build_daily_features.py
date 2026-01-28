"""
Daily feature pipeline: Earnings events → TAQ data → Microstructure features

BATCH PROCESSING VERSION - SEPARATE FILES
- Each batch saves to independent parquet file
- No duplicate checking (fixed index ranges guarantee no overlap)
- File naming: daily_features_{start}_{end}.parquet
- Merge files separately after all processing complete
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

BATCH_SIZE = 25
START_IDX = 2975

N_TRADING_DAYS = 10

EVENT_TABLE_PATH = "data/processed/event_table_2023_2024.parquet"
OUTPUT_DIR = r"SET_YOUR_OWN_OUTPUT_PATH"


# ============================================================
# NYSE Trading Day Calendar
# ============================================================

def generate_date_window(rdq: pd.Timestamp, n_days: int = N_TRADING_DAYS) -> List[pd.Timestamp]:
    """Generate EXACTLY n NYSE trading days before rdq."""
    
    nyse = mcal.get_calendar('NYSE')
    
    start_date = rdq - pd.Timedelta(days=60)
    end_date = rdq - pd.Timedelta(days=1)
    
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tolist()
    
    if len(trading_days) < n_days:
        logger.warning(f"Not enough NYSE trading days before {rdq.date()}: "
                      f"got {len(trading_days)}, need {n_days}")
        return trading_days
    
    return trading_days[-n_days:]


# ============================================================
# Batch-safe Pipeline
# ============================================================

def check_batch_already_processed(output_dir: str, start_idx: int, end_idx: int) -> bool:
    """Check if this batch file already exists."""
    output_file = Path(output_dir) / f"daily_features_{start_idx}_{end_idx}.parquet"
    exists = output_file.exists()
    
    if exists:
        logger.info(f"✓ Batch file already exists: {output_file.name}")
        try:
            df = pd.read_parquet(output_file)
            logger.info(f"  Contains {len(df)} rows, {df['event_id'].nunique()} events")
        except:
            pass
    
    return exists


def save_batch_results(
    features: List[Dict],
    output_dir: str,
    start_idx: int,
    end_idx: int
):
    """Save batch results to independent parquet file."""
    if len(features) == 0:
        logger.warning("No features to save")
        return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    output_file = Path(output_dir) / f"daily_features_{start_idx}_{end_idx}.parquet"
    
    # Save
    df = pd.DataFrame(features)
    df = df.sort_values(['event_id', 'date']).reset_index(drop=True)
    df.to_parquet(output_file, index=False)
    
    logger.info(f"✓ Saved batch file: {output_file.name}")
    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Events: {df['event_id'].nunique()}")
    logger.info(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")


def build_daily_features_batch(
    event_table_path: str = EVENT_TABLE_PATH,
    output_dir: str = OUTPUT_DIR,
    batch_size: int = BATCH_SIZE,
    start_idx: int = START_IDX
):
    """Build daily features for a batch of events."""
    
    logger.info("="*60)
    logger.info("BATCH PROCESSING - Independent Files")
    logger.info("="*60)
    logger.info(f"Batch size: {batch_size} events")
    logger.info(f"Start index: {start_idx}")
    logger.info(f"Output dir: {output_dir}")
    
    all_events_df = pd.read_parquet(event_table_path)
    total_events = len(all_events_df)
    logger.info(f"Total events available: {total_events}")
    
    end_idx = min(start_idx + batch_size, total_events)
    logger.info(f"Batch range: [{start_idx}, {end_idx})")
    
    # Check if already processed
    if check_batch_already_processed(output_dir, start_idx, end_idx):
        logger.info("⚠ This batch already processed - skipping")
        logger.info("  Delete the file to reprocess, or change START_IDX")
        return
    
    batch_events_df = all_events_df.iloc[start_idx:end_idx].copy()
    logger.info(f"Processing {len(batch_events_df)} events")
    
    logger.info("Connecting to WRDS...")
    db = wrds.Connection()
    logger.info("✓ Connected")
    
    all_features = []
    
    try:
        for idx, event in batch_events_df.iterrows():
            try:
                event_features = process_event(event, db)
                all_features.extend(event_features)
                gc.collect()
            except Exception as e:
                logger.error(f"Failed event {event['event_id']}: {e}")
                continue
        
        if len(all_features) > 0:
            save_batch_results(all_features, output_dir, start_idx, end_idx)
        else:
            logger.warning("No features computed")
    
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted")
        logger.info("Saving partial results...")
        if len(all_features) > 0:
            save_batch_results(all_features, output_dir, start_idx, end_idx)
        raise
    
    finally:
        db.close()
        logger.info("WRDS connection closed")
    
    logger.info("="*60)
    logger.info("Batch Complete")
    logger.info("="*60)
    logger.info(f"Events: {len(batch_events_df)}")
    logger.info(f"Rows: {len(all_features)}")
    
    remaining = total_events - end_idx
    if remaining > 0:
        estimated_batches = int(np.ceil(remaining / batch_size))
        logger.info(f"\n📊 Progress: {end_idx}/{total_events} ({end_idx/total_events*100:.1f}%)")
        logger.info(f"Remaining batches: ~{estimated_batches}")
        logger.info(f"\n💡 Next: Set START_IDX = {end_idx}")
    else:
        logger.info(f"\n🎉 ALL EVENTS PROCESSED!")
        logger.info(f"Merge files with: merge_batch_files.py")


def process_event(event: pd.Series, db: wrds.Connection) -> List[Dict]:
    """Process a single earnings event."""
    
    event_id = event['event_id']
    permno = int(event['permno'])
    rdq = pd.to_datetime(event['rdq'])
    
    logger.info(f"Processing {event_id}: permno={permno}, rdq={rdq.date()}")
    
    dates = generate_date_window(rdq, N_TRADING_DAYS)
    logger.info(f"  {len(dates)} trading days")
    
    event_features = []
    successful_days = 0
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            trades_df = load_trades(permno, date_str, db)
            quotes_df = load_quotes(permno, date_str, db)
            
            if len(trades_df) == 0:
                logger.warning(f"  ✗ No trades {date_str}")
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
    
    logger.info(f"  {successful_days}/{len(dates)} days successful")
    
    return event_features


def compute_daily_features(
    event_id: str,
    permno: int,
    date: pd.Timestamp,
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame
) -> Dict:
    """Compute all daily features."""
    
    features = {
        'event_id': event_id,
        'permno': permno,
        'date': date
    }
    
    # Order flow
    try:
        ofi_features = compute_daily_order_flow(trades_df, quotes_df)
        ofi_features.pop('permno', None)
        ofi_features.pop('date', None)
        features.update(ofi_features)
    except Exception as e:
        logger.warning(f"Order flow failed: {e}")
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
    
    # Volume
    try:
        vol_features = compute_daily_volume(trades_df, adv_usd=None)
        vol_features.pop('permno', None)
        vol_features.pop('date', None)
        features.update(vol_features)
    except Exception as e:
        logger.warning(f"Volume failed: {e}")
        features.update({
            'dollar_volume': 0.0,
            'share_volume': 0,
            'abnormal_vol_ratio': np.nan,
            'morning_share': np.nan
        })
    
    # Liquidity
    try:
        liq_features = compute_daily_liquidity(quotes_df)
        liq_features.pop('permno', None)
        liq_features.pop('date', None)
        features.update(liq_features)
    except Exception as e:
        logger.warning(f"Liquidity failed: {e}")
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


if __name__ == "__main__":
    build_daily_features_batch()
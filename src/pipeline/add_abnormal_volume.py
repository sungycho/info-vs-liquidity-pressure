"""
Add abnormal_vol_ratio to daily_features_full.parquet using CRSP daily volume baseline.

This script:
1. Loads existing daily features (pre-event window only)
2. Queries CRSP for historical daily volume
3. Computes 20-day rolling volume baseline
4. Calculates abnormal_vol_ratio = daily_volume / baseline_volume
5. Appends new columns to daily features

Author: Sung
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import wrds
from pathlib import Path
from dotenv import load_dotenv
import logging
from sqlalchemy import text

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DAILY_FEATURES_PATH = "data/processed/daily_features_full.parquet"
OUTPUT_PATH = "data/processed/daily_features_full_v2.parquet"

# Config
BASELINE_WINDOW = 20  # Trading days for rolling average
MIN_PERIODS = 10      # Minimum periods for rolling calculation


def load_daily_features(path: str) -> pd.DataFrame:
    """Load existing daily features."""
    logger.info(f"Loading daily features from {path}")
    df = pd.read_parquet(path)
    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Unique permnos: {df['permno'].nunique()}")
    return df


def get_crsp_volume(permnos: list, start_date: str, end_date: str, db: wrds.Connection) -> pd.DataFrame:
    """
    Query CRSP daily volume for given permnos and date range.
    
    Args:
        permnos: List of permnos to query
        start_date: Start date (YYYY-MM-DD), should include buffer for rolling baseline
        end_date: End date (YYYY-MM-DD)
        db: WRDS connection
    
    Returns:
        DataFrame with columns: permno, date, crsp_volume
    """
    logger.info("Querying CRSP daily volume...")
    logger.info(f"  Permnos: {len(permnos)}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    
    # Convert permno list to SQL IN clause
    permno_list = ','.join(map(str, permnos))
    
    query = text(f"""
    SELECT 
        permno,
        date,
        vol as crsp_volume
    FROM crsp.dsf
    WHERE permno IN ({permno_list})
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND vol IS NOT NULL
        AND vol > 0
    ORDER BY permno, date
    """)
    
    try:
        result = db.connection.execute(query)
        rows = result.fetchall()
        columns = result.keys()
        
        crsp_df = pd.DataFrame(rows, columns=columns)
        crsp_df['date'] = pd.to_datetime(crsp_df['date'])
        
        logger.info(f"  ✓ Loaded {len(crsp_df):,} CRSP daily volume observations")
        return crsp_df
    
    except Exception as e:
        logger.error(f"Error querying CRSP: {e}")
        raise


def compute_volume_baseline(crsp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling volume baseline for each permno.
    
    Args:
        crsp_df: DataFrame with columns [permno, date, crsp_volume]
    
    Returns:
        DataFrame with added column: vol_baseline
    """
    logger.info("Computing rolling volume baselines...")
    
    # Sort by permno and date
    crsp_df = crsp_df.sort_values(['permno', 'date']).reset_index(drop=True)
    
    # Compute rolling mean with shift(1) to prevent look-ahead
    crsp_df['vol_baseline'] = (
        crsp_df.groupby('permno')['crsp_volume']
        .transform(lambda x: x.rolling(window=BASELINE_WINDOW, min_periods=MIN_PERIODS).mean().shift(1))
    )
    
    # Count non-null baselines
    valid_count = crsp_df['vol_baseline'].notna().sum()
    logger.info(f"  ✓ Computed baselines for {valid_count:,} observations")
    logger.info(f"  Missing baselines: {len(crsp_df) - valid_count:,} (insufficient history)")
    
    return crsp_df


def merge_and_calculate_avr(daily_df: pd.DataFrame, crsp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge CRSP volume baseline into daily features and compute abnormal_vol_ratio.
    
    Args:
        daily_df: Daily features DataFrame
        crsp_df: CRSP volume with baseline
    
    Returns:
        Updated daily features with new columns
    """
    logger.info("Merging CRSP volume with daily features...")
    
    # Ensure date types match
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    crsp_df['date'] = pd.to_datetime(crsp_df['date'])
    
    # Left join: keep all daily features, add CRSP volume
    merged = daily_df.merge(
        crsp_df[['permno', 'date', 'crsp_volume', 'vol_baseline']],
        on=['permno', 'date'],
        how='left'
    )
    
    logger.info(f"  ✓ Merged {len(merged):,} rows")
    
    # Check merge quality
    match_rate = merged['crsp_volume'].notna().mean() * 100
    logger.info(f"  CRSP volume match rate: {match_rate:.1f}%")
    
    # Compute abnormal_vol_ratio
    logger.info("Computing abnormal_vol_ratio...")
    merged['abnormal_vol_ratio'] = merged['volume'] / merged['vol_baseline']
    
    # Validation
    valid_avr = merged['abnormal_vol_ratio'].notna().sum()
    logger.info(f"  ✓ Valid abnormal_vol_ratio: {valid_avr:,} / {len(merged):,}")
    
    # Summary stats
    if valid_avr > 0:
        avr_stats = merged['abnormal_vol_ratio'].describe()
        logger.info(f"\n  AVR Distribution:")
        logger.info(f"    Mean: {avr_stats['mean']:.2f}")
        logger.info(f"    Median: {avr_stats['50%']:.2f}")
        logger.info(f"    P95: {avr_stats['75%']:.2f}")
        logger.info(f"    Max: {avr_stats['max']:.2f}")
    
    return merged


def save_output(df: pd.DataFrame, output_path: str):
    """Save enriched daily features."""
    logger.info(f"Saving output to {output_path}")
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_parquet(output_path, index=False)
    
    # Verify
    file_size = Path(output_path).stat().st_size / (1024**2)  # MB
    logger.info(f"  ✓ Saved {len(df):,} rows ({file_size:.1f} MB)")


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Add Abnormal Volume Ratio Pipeline")
    logger.info("="*60)
    
    # Step 1: Load daily features
    daily_df = load_daily_features(DAILY_FEATURES_PATH)
    
    # Step 2: Determine CRSP query range
    # Need at least BASELINE_WINDOW days before earliest date
    min_date = daily_df['date'].min()
    max_date = daily_df['date'].max()
    
    # Add 30 calendar days buffer for rolling baseline (covers ~20 trading days)
    start_date = (min_date - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = max_date.strftime('%Y-%m-%d')
    
    logger.info(f"\nCRSP query range (with buffer): {start_date} to {end_date}")
    
    # Step 3: Get unique permnos
    permnos = daily_df['permno'].unique().tolist()
    
    # Step 4: Connect to WRDS and query CRSP
    logger.info("\nConnecting to WRDS...")
    db = wrds.Connection()
    logger.info("  ✓ Connected")
    
    try:
        # Query CRSP volume
        crsp_df = get_crsp_volume(permnos, start_date, end_date, db)
        
        # Step 5: Compute rolling baseline
        crsp_df = compute_volume_baseline(crsp_df)
        
        # Step 6: Merge and calculate AVR
        enriched_df = merge_and_calculate_avr(daily_df, crsp_df)
        
        # Step 7: Save output
        save_output(enriched_df, OUTPUT_PATH)
        
    finally:
        db.close()
        logger.info("\nWRDS connection closed")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete")
    logger.info("="*60)
    logger.info(f"Input:  {DAILY_FEATURES_PATH}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"New columns added: crsp_volume, vol_baseline, abnormal_vol_ratio")
    
    # Data quality summary
    avr_coverage = enriched_df['abnormal_vol_ratio'].notna().mean() * 100
    logger.info(f"\n📊 Data Quality:")
    logger.info(f"  abnormal_vol_ratio coverage: {avr_coverage:.1f}%")
    
    if avr_coverage < 90:
        logger.warning("  ⚠ Low coverage - check for missing CRSP data or insufficient history")


if __name__ == "__main__":
    main()
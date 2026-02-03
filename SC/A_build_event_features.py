"""
Build event_features.parquet from daily microstructure data.

FIXED ISSUES:
1. OFI sign preserved (removed abs transform)
2. Added reversal term for liquidity
3. Cross-event z-score normalization
4. Proper component balance

Author: Sung
Date: 2025-01-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DAILY_FEATURES_PATH = "data/processed/daily_features.parquet"
EVENT_TABLE_PATH = "data/processed/event_table.parquet"
OUTPUT_PATH = "SC/A_event_features.parquet"

# Config
MIN_DAYS = 7
VOLUME_BURST_THRESHOLD = 1.5


def load_data():
    """Load daily features and event metadata."""
    logger.info("Loading data...")
    
    daily_df = pd.read_parquet(DAILY_FEATURES_PATH)
    logger.info(f"  Loaded {len(daily_df):,} daily observations")
    
    events_df = pd.read_parquet(EVENT_TABLE_PATH)
    logger.info(f"  Loaded {len(events_df):,} events")
    
    return daily_df, events_df


def add_event_metadata(daily_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Add event metadata (announce date, permno if missing)."""
    logger.info("Adding event metadata...")
    
    events_meta = events_df[['event_id', 'permno', 'rdq']].copy()
    events_meta = events_meta.rename(columns={'rdq': 'announce_date'})
    
    daily_df = daily_df.merge(events_meta, on=['event_id', 'permno'], how='left')
    daily_df['announce_date'] = pd.to_datetime(daily_df['announce_date'])
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    
    logger.info(f"  ✓ Added announce_date for {daily_df['event_id'].nunique()} events")
    
    return daily_df


def filter_sufficient_events(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Drop events with insufficient trading days."""
    logger.info(f"Filtering events with sufficient data (min {MIN_DAYS} trading days)...")
    
    event_counts = daily_df.groupby('event_id').size()
    
    logger.info(f"  Trading days per event distribution:")
    logger.info(f"{event_counts.value_counts().sort_index().to_string()}")
    
    insufficient = (event_counts < MIN_DAYS).sum()
    
    if insufficient > 0:
        logger.info(f"  Dropping {insufficient} events with < {MIN_DAYS} trading days")
        valid_events = event_counts[event_counts >= MIN_DAYS].index
        daily_df = daily_df[daily_df['event_id'].isin(valid_events)]
    
    logger.info(f"  ✓ Final: {daily_df['event_id'].nunique()} events, {len(daily_df):,} rows")
    
    return daily_df


def aggregate_daily_to_event(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily features to event level."""
    logger.info("Aggregating daily features to event level...")
    
    event_agg = daily_df.groupby('event_id').agg({
        # Info components - KEEP SIGN
        'ofi': ['mean', 'std'],
        'morning_share': 'mean',
        'spread_stability': 'mean',
        
        # Liquidity components
        'ofi_abs': 'mean',  # For liquidity intensity
        'abnormal_vol_ratio': lambda x: (x > VOLUME_BURST_THRESHOLD).mean(),
        'spread_std': 'mean',
        'spread_p95': 'max',
        
        # Controls
        'volume': 'mean',
        'num_trades': 'mean',
        
        # Metadata
        'permno': 'first',
        'announce_date': 'first'
    }).reset_index()
    
    event_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in event_agg.columns]
    
    event_agg = event_agg.rename(columns={
        'ofi_mean': 'ofi_mean',  # SIGNED - DO NOT ABS
        'ofi_std': 'ofi_std_mean',
        'ofi_abs_mean': 'ofi_abs_mean',
        'abnormal_vol_ratio_<lambda>': 'volume_burst_fraction',
        'morning_share_mean': 'morning_share_mean',
        'spread_stability_mean': 'spread_stability_mean',
        'spread_std_mean': 'spread_std_mean',
        'spread_p95_max': 'spread_p95_max',
        'permno_first': 'permno',
        'announce_date_first': 'event_date'
    })
    
    logger.info(f"  Aggregated to {len(event_agg)} events")
    
    return event_agg


def compute_ofi_autocorr(daily_df: pd.DataFrame) -> pd.Series:
    """Compute OFI autocorrelation (lag=1) for each event."""
    logger.info("Computing OFI autocorrelation...")
    
    def safe_autocorr(x):
        if len(x) < 4:
            return np.nan
        try:
            return x.autocorr(lag=1)
        except:
            return np.nan
    
    ofi_autocorr = daily_df.groupby('event_id')['ofi'].apply(safe_autocorr)
    
    valid_count = ofi_autocorr.notna().sum()
    logger.info(f"  Valid autocorr: {valid_count} / {len(ofi_autocorr)}")
    
    return ofi_autocorr


def compute_ofi_reversal(daily_df: pd.DataFrame) -> pd.Series:
    """
    Compute OFI reversal tendency.
    
    Reversal = -corr(OFI_t, OFI_t+1) using first differences
    High reversal → liquidity-driven (mean-reverting)
    """
    logger.info("Computing OFI reversal...")
    
    def safe_reversal(x):
        if len(x) < 4:
            return np.nan
        try:
            # Compute first differences
            diff = x.diff().dropna()
            if len(diff) < 3:
                return np.nan
            # Negative autocorr of differences = reversal
            return -diff.autocorr(lag=1)
        except:
            return np.nan
    
    ofi_reversal = daily_df.groupby('event_id')['ofi'].apply(safe_reversal)
    
    valid_count = ofi_reversal.notna().sum()
    logger.info(f"  Valid reversal: {valid_count} / {len(ofi_reversal)}")
    
    return ofi_reversal


def zscore_normalize(event_agg: pd.DataFrame, components: list) -> pd.DataFrame:
    """
    Cross-event z-score normalization.
    
    CRITICAL: Centers at 0 so info/liq scores can be negative.
    """
    logger.info(f"Z-score normalizing {len(components)} components...")
    
    for col in components:
        if col not in event_agg.columns:
            logger.warning(f"  Missing component: {col}")
            continue
        
        mean = event_agg[col].mean()
        std = event_agg[col].std()
        
        if std == 0 or pd.isna(std):
            logger.warning(f"  {col}: zero variance, setting to 0")
            event_agg[f'{col}_z'] = 0.0
        else:
            event_agg[f'{col}_z'] = (event_agg[col] - mean) / std
        
        # Log distribution
        logger.info(f"    {col}: mean={mean:.3f}, std={std:.3f}")
    
    return event_agg


def compute_pressure_scores(event_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pressure scores with FIXED components.
    
    INFO (persistence + direction):
      - ofi_mean (SIGNED - directional pressure)
      - ofi_autocorr (persistence)
      - spread_stability (tight spreads = info edge)
    
    LIQ (reversal + intensity):
      - ofi_reversal (mean-reverting = liquidity)
      - ofi_abs_mean (intensity regardless of sign)
      - volume_burst_fraction (episodic pressure)
      - spread_std_mean (spread volatility)
    """
    logger.info("Computing pressure scores...")
    
    # Define components
    info_components = [
        'ofi_mean',              # SIGNED - DO NOT ABS
        'ofi_autocorr',
        'spread_stability_mean'
    ]
    
    liq_components = [
        'ofi_reversal',          # NEW - KEY FIX
        'ofi_abs_mean',
        'volume_burst_fraction',
        'spread_std_mean'
    ]
    
    # Z-score normalize
    event_agg = zscore_normalize(event_agg, info_components + liq_components)
    
    # Info score (sum of z-scored components)
    info_cols = [f'{c}_z' for c in info_components]
    event_agg['info_score'] = event_agg[info_cols].sum(axis=1)
    
    # Liquidity score (sum of z-scored components)
    liq_cols = [f'{c}_z' for c in liq_components]
    event_agg['liq_score'] = event_agg[liq_cols].sum(axis=1)
    
    # Final pressure score: tanh((info - liq) / 2)
    event_agg['pressure_score'] = np.tanh((event_agg['info_score'] - event_agg['liq_score']) / 2)
    
    logger.info(f"  ✓ Computed scores for {len(event_agg)} events")
    
    # Log component contributions
    logger.info("\n  Component statistics:")
    for comp in info_components + liq_components:
        z_col = f'{comp}_z'
        if z_col in event_agg.columns:
            logger.info(f"    {comp}: mean={event_agg[z_col].mean():.3f}, "
                       f"std={event_agg[z_col].std():.3f}")
    
    return event_agg


def validate_output(event_agg: pd.DataFrame):
    """Validation checks."""
    logger.info("\nValidation:")
    
    # Check missing
    missing = event_agg['pressure_score'].isna().sum()
    logger.info(f"  Missing pressure_score: {missing} / {len(event_agg)}")
    
    # Distribution
    ps_stats = event_agg['pressure_score'].describe()
    logger.info(f"\n  Pressure Score Distribution:")
    logger.info(f"    Mean: {ps_stats['mean']:.3f}")
    logger.info(f"    Std:  {ps_stats['std']:.3f}")
    logger.info(f"    Min:  {ps_stats['min']:.3f}")
    logger.info(f"    Q25:  {ps_stats['25%']:.3f}")
    logger.info(f"    Q50:  {ps_stats['50%']:.3f}")
    logger.info(f"    Q75:  {ps_stats['75%']:.3f}")
    logger.info(f"    Max:  {ps_stats['max']:.3f}")
    
    # Component score distribution
    logger.info(f"\n  Component Score Distribution:")
    for score in ['info_score', 'liq_score']:
        stats = event_agg[score].describe()
        logger.info(f"    {score}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                   f"min={stats['min']:.3f}, max={stats['max']:.3f}")
    
    # Correlation
    corr = event_agg[['info_score', 'liq_score']].corr().iloc[0, 1]
    logger.info(f"\n  Correlation (info_score, liq_score): {corr:.3f}")
    
    if abs(corr) > 0.7:
        logger.warning("    ⚠ High correlation - components may not be independent")
    
    # Check range
    if event_agg['pressure_score'].min() < -1.01 or event_agg['pressure_score'].max() > 1.01:
        logger.warning("    ⚠ pressure_score outside [-1, 1] range")
    
    # Check sign distribution
    n_positive = (event_agg['pressure_score'] > 0).sum()
    n_negative = (event_agg['pressure_score'] < 0).sum()
    logger.info(f"\n  Sign Distribution:")
    logger.info(f"    Positive: {n_positive} ({100*n_positive/len(event_agg):.1f}%)")
    logger.info(f"    Negative: {n_negative} ({100*n_negative/len(event_agg):.1f}%)")
    logger.info(f"    Zero:     {len(event_agg) - n_positive - n_negative}")


def save_output(event_agg: pd.DataFrame, output_path: str):
    """Save event features."""
    logger.info(f"\nSaving output to {output_path}")
    
    output_cols = [
        'event_id', 'permno', 'event_date',
        
        # Raw aggregated features
        'ofi_mean', 'ofi_std_mean', 'ofi_abs_mean', 
        'ofi_autocorr', 'ofi_reversal',
        'morning_share_mean', 'spread_stability_mean', 
        'spread_std_mean', 'spread_p95_max',
        'volume_burst_fraction',
        
        # Scores
        'info_score', 'liq_score', 'pressure_score',
        
        # Controls
        'volume_mean', 'num_trades_mean'
    ]
    
    output_cols = [c for c in output_cols if c in event_agg.columns]
    output_df = event_agg[output_cols].copy()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    
    file_size = Path(output_path).stat().st_size / 1024
    logger.info(f"  ✓ Saved {len(output_df)} events ({file_size:.1f} KB)")


def main():
    """Main execution pipeline."""
    logger.info("="*60)
    logger.info("Build Event Features Pipeline (FIXED)")
    logger.info("="*60)
    
    # Load
    daily_df, events_df = load_data()
    
    # Add metadata
    daily_df = add_event_metadata(daily_df, events_df)
    
    # Filter
    daily_df = filter_sufficient_events(daily_df)
    
    # Aggregate
    event_agg = aggregate_daily_to_event(daily_df)
    
    # Compute OFI metrics
    ofi_autocorr = compute_ofi_autocorr(daily_df)
    ofi_reversal = compute_ofi_reversal(daily_df)
    
    event_agg = event_agg.merge(
        ofi_autocorr.rename('ofi_autocorr').reset_index(),
        on='event_id', how='left'
    )
    event_agg = event_agg.merge(
        ofi_reversal.rename('ofi_reversal').reset_index(),
        on='event_id', how='left'
    )
    
    # Impute missing
    for col in ['ofi_autocorr', 'ofi_reversal']:
        if event_agg[col].isna().any():
            median_val = event_agg[col].median()
            n_missing = event_agg[col].isna().sum()
            logger.info(f"  Imputing {n_missing} missing {col} with median: {median_val:.3f}")
            event_agg[col] = event_agg[col].fillna(median_val)
    
    # Compute scores
    event_agg = compute_pressure_scores(event_agg)
    
    # Validate
    validate_output(event_agg)
    
    # Save
    save_output(event_agg, OUTPUT_PATH)
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline Complete")
    logger.info("="*60)
    logger.info(f"Input:  {DAILY_FEATURES_PATH}")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"Events: {len(event_agg)}")
    logger.info(f"\n📊 Coverage: {len(event_agg)} / {events_df['event_id'].nunique()} original events")


if __name__ == "__main__":
    main()
"""

B_build_event_features.py (REVISED)

Rank-based pressure score construction (Method 1: Baseline Robust Alternative)

"""

import pandas as pd

import numpy as np

from pathlib import Path

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def load_data():

    """Load event table and daily features"""

    event_table = pd.read_parquet('data/processed/event_table.parquet')

    daily_features = pd.read_parquet('data/processed/daily_features.parquet')

    logger.info(f"Loaded {len(event_table)} events, {len(daily_features)} daily observations")

    return event_table, daily_features

def compute_ofi_autocorr(daily_features):

    """

    Compute actual OFI autocorrelation (lag-1) within each event window

    Measures directional persistence, not volatility ratio

    """

    def autocorr_lag1(group):

        ofi_series = group.sort_values('date')['ofi']  # Use 'date' and 'ofi'

        if len(ofi_series) < 3:  # Need minimum obs

            return np.nan

        if ofi_series.std() == 0:  # Constant series

            return np.nan

        return ofi_series.autocorr(lag=1)

    autocorr_df = daily_features.groupby('event_id', group_keys=False).apply(autocorr_lag1).rename('ofi_autocorr').reset_index()

    logger.info(f"OFI autocorr: {len(autocorr_df)} events, "

                f"{autocorr_df['ofi_autocorr'].isna().sum()} NaN ({autocorr_df['ofi_autocorr'].isna().mean()*100:.1f}%)")

    return autocorr_df

def compute_volume_burst_fraction(daily_features):

    """

    Fraction of days with abnormal volume bursts (> 1.5x baseline)

    Measures liquidity event frequency, not average intensity

    """

    burst_threshold = 1.5

    burst_df = daily_features.groupby('event_id', group_keys=False).apply(

        lambda g: (g['abnormal_vol_ratio'] > burst_threshold).mean()

    ).rename('volume_burst_fraction').reset_index()

    logger.info(f"Volume burst fraction computed for {len(burst_df)} events")

    return burst_df

def aggregate_to_event_level(daily_features):

    """Aggregate daily features to event level over [-10, -1] window"""

    # Special features requiring time-series logic

    ofi_autocorr_df = compute_ofi_autocorr(daily_features)

    volume_burst_df = compute_volume_burst_fraction(daily_features)

    # Standard aggregations

    event_agg = daily_features.groupby('event_id').agg({

        # Information components

        'ofi': 'mean',  # Will become ofi_mean

        'ofi_std': 'mean',  # Will become ofi_std_mean

        'spread_stability': 'mean',  # Will become spread_stability_mean

        'morning_share': 'mean',  # Will become morning_share_mean

        # Liquidity components

        'ofi_abs': 'mean',  # Will become ofi_abs_mean

        'spread_std': 'mean',  # Will become spread_std_mean

        'spread_p95': 'max',  # Will become spread_p95_max

        # Additional features

        'volume': 'mean',  # Will become volume_mean

        'num_trades': 'mean',  # Will become num_trades_mean

        'permno': 'first'  # Keep permno for merge

    }).reset_index()

    # Rename columns to match expected names

    event_agg = event_agg.rename(columns={

        'ofi': 'ofi_mean',

        'ofi_std': 'ofi_std_mean',

        'spread_stability': 'spread_stability_mean',

        'morning_share': 'morning_share_mean',

        'ofi_abs': 'ofi_abs_mean',

        'spread_std': 'spread_std_mean',

        'spread_p95': 'spread_p95_max',

        'volume': 'volume_mean',

        'num_trades': 'num_trades_mean'

    })

    # Merge special features

    event_agg = event_agg.merge(ofi_autocorr_df, on='event_id', how='left')

    event_agg = event_agg.merge(volume_burst_df, on='event_id', how='left')

    logger.info(f"Aggregated to {len(event_agg)} events")

    return event_agg

def compute_rank_based_pressure(event_features):

    """

    Rank-based pressure score with critical fixes:

    1. OFI autocorr = actual persistence (not std/mean)

    2. Volume burst = fraction of burst days (not mean ratio)

    3. Spread stability inverted (lower CV = higher stability)

    4. Cross-sectional ranking by quarter to avoid regime mixing

    """

    df = event_features.copy()

    # Create quarter identifier for cross-sectional ranking

    df['year_quarter'] = df['rdq'].dt.to_period('Q')

    # Information components

    # Note: spread_stability_mean is CV-based, so INVERT (lower CV = more stable = higher info)

    df['spread_stability_inv'] = 1 / (df['spread_stability_mean'] + 1e-8)

    df['ofi_mean_abs'] = df['ofi_mean'].abs()

    info_cols = ['ofi_mean_abs', 'ofi_autocorr', 'spread_stability_inv']

    # Liquidity components (higher = more liquidity pressure)

    liq_cols = ['ofi_abs_mean', 'volume_burst_fraction', 'spread_std_mean', 'spread_p95_max']

    # Cross-sectional ranking within each quarter

    for col in info_cols + liq_cols:

        df[f'{col}_rank'] = df.groupby('year_quarter')[col].rank(

            pct=True, 

            method='average', 

            na_option='keep'

        )

    # Aggregate rank scores (only if at least 2 non-NaN components)

    df['info_score'] = df[[f'{col}_rank' for col in info_cols]].mean(axis=1, skipna=True)

    df['liq_score'] = df[[f'{col}_rank' for col in liq_cols]].mean(axis=1, skipna=True)

    # Drop events with insufficient data

    initial_count = len(df)

    df = df.dropna(subset=['info_score', 'liq_score'])

    logger.info(f"Dropped {initial_count - len(df)} events due to insufficient component data")

    # Pressure score: raw difference

    df['pressure_delta'] = df['info_score'] - df['liq_score']

    # Robust scaling using MAD before tanh

    delta_median = df['pressure_delta'].median()

    delta_mad = (df['pressure_delta'] - delta_median).abs().median()

    df['pressure_delta_scaled'] = (df['pressure_delta'] - delta_median) / (1.4826 * delta_mad + 1e-8)

    # Final pressure scores

    df['pressure_score'] = df['pressure_delta']  # Raw version

    df['pressure_score_tanh'] = np.tanh(df['pressure_delta_scaled'])  # Scaled tanh version

    # Diagnostics

    logger.info(f"\nPressure score statistics:")

    logger.info(f"  Raw delta: mean={df['pressure_delta'].mean():.3f}, "

                f"std={df['pressure_delta'].std():.3f}, "

                f"range=[{df['pressure_delta'].min():.3f}, {df['pressure_delta'].max():.3f}]")

    logger.info(f"  Tanh: mean={df['pressure_score_tanh'].mean():.3f}, "

                f"std={df['pressure_score_tanh'].std():.3f}")

    # Liquidity proxy diagnostics

    logger.info(f"\nLiquidity proxy check (correlations with pressure_score):")

    logger.info(f"  spread_std_mean: {df['pressure_score'].corr(df['spread_std_mean']):.3f}")

    logger.info(f"  spread_p95_max: {df['pressure_score'].corr(df['spread_p95_max']):.3f}")

    logger.info(f"  ofi_abs_mean: {df['pressure_score'].corr(df['ofi_abs_mean']):.3f}")

    logger.info(f"  volume_burst_fraction: {df['pressure_score'].corr(df['volume_burst_fraction']):.3f}")

    return df

def main():

    logger.info("Starting rank-based pressure score construction (REVISED)...")

    # Load data

    event_table, daily_features = load_data()

    # Aggregate to event level

    event_features = aggregate_to_event_level(daily_features)

    # Merge with event metadata

    event_features = event_features.merge(

        event_table[['event_id', 'rdq', 'datadate']], 

        on='event_id',

        how='left'

    )

    # Rename datadate to event_date to match original format

    event_features = event_features.rename(columns={'datadate': 'event_date'})

    # Compute rank-based pressure scores

    event_features = compute_rank_based_pressure(event_features)

    # Save

    output_path = Path('Methods_Sung_Cho/B_event_features.parquet')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_features.to_parquet(output_path, index=False)

    logger.info(f"\nSaved to {output_path}")

    logger.info(f"Final sample: {len(event_features)} events")

if __name__ == '__main__':

    main()
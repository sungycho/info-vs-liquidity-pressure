"""

D_build_event_features.py (REVISED)

Regime-aware normalization pressure score (Method 3: Secondary Robustness)

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

    """Actual lag-1 autocorrelation (same as Method B/C)"""

    def autocorr_lag1(group):

        ofi_series = group.sort_values('date')['ofi']

        if len(ofi_series) < 3:

            return np.nan

        if ofi_series.std() == 0:

            return np.nan

        return ofi_series.autocorr(lag=1)

    autocorr_df = daily_features.groupby('event_id', group_keys=False).apply(

        autocorr_lag1

    ).rename('ofi_autocorr').reset_index()

    logger.info(f"OFI autocorr: {autocorr_df['ofi_autocorr'].isna().sum()} NaN "

                f"({autocorr_df['ofi_autocorr'].isna().mean()*100:.1f}%)")

    return autocorr_df

def compute_volume_burst_fraction(daily_features):

    """Fraction of burst days (same as Method B/C)"""

    burst_threshold = 1.5

    burst_df = daily_features.groupby('event_id', group_keys=False).apply(

        lambda g: (g['abnormal_vol_ratio'] > burst_threshold).mean()

    ).rename('volume_burst_fraction').reset_index()

    logger.info(f"Volume burst fraction computed for {len(burst_df)} events")

    return burst_df

def compute_regime_variables(daily_features):

    """

    Compute pre-determined regime variables to avoid endogeneity

    Use full-sample characteristics (not window-specific outcomes)

    """

    # Average spread and volume across ALL available data per stock

    # This avoids regime being contaminated by pressure signal

    stock_chars = daily_features.groupby('permno').agg({

        'spread_mean': 'mean',  # Average spread characteristic

        'crsp_volume': 'mean'   # Average volume characteristic

    }).reset_index()

    stock_chars = stock_chars.rename(columns={

        'spread_mean': 'stock_avg_spread',

        'crsp_volume': 'stock_avg_volume'

    })

    logger.info(f"Computed regime variables for {len(stock_chars)} stocks")

    return stock_chars

def aggregate_to_event_level(daily_features):

    """Aggregate daily features (consistent with Method B/C)"""

    ofi_autocorr_df = compute_ofi_autocorr(daily_features)

    volume_burst_df = compute_volume_burst_fraction(daily_features)

    event_agg = daily_features.groupby('event_id').agg({

        'ofi': 'mean',

        'spread_stability': 'mean',

        'ofi_abs': 'mean',

        'spread_std': 'mean',

        'spread_p95': 'max',

        'permno': 'first'  # Keep permno for regime merge

    }).reset_index()

    event_agg = event_agg.rename(columns={

        'ofi': 'ofi_mean',

        'spread_stability': 'spread_stability_mean',

        'ofi_abs': 'ofi_abs_mean',

        'spread_std': 'spread_std_mean',

        'spread_p95': 'spread_p95_max'

    })

    event_agg = event_agg.merge(ofi_autocorr_df, on='event_id', how='left')

    event_agg = event_agg.merge(volume_burst_df, on='event_id', how='left')

    logger.info(f"Aggregated to {len(event_agg)} events")

    return event_agg

def define_regimes(event_features, stock_chars):

    """

    Define regimes based on pre-determined stock characteristics

    Within each year-quarter to avoid time mixing

    """

    df = event_features.copy()

    # Merge stock characteristics

    df = df.merge(stock_chars, on='permno', how='left')

    # Create year-quarter

    df['year_quarter'] = df['rdq'].dt.to_period('Q')

    # Within each quarter, create spread and volume terciles

    df['spread_regime'] = df.groupby('year_quarter')['stock_avg_spread'].transform(

        lambda x: pd.qcut(x, q=3, labels=['tight', 'mid', 'wide'], duplicates='drop')

    )

    df['volume_regime'] = df.groupby('year_quarter')['stock_avg_volume'].transform(

        lambda x: pd.qcut(x, q=3, labels=['low', 'mid', 'high'], duplicates='drop')

    )

    # Combined regime

    df['regime'] = df['spread_regime'].astype(str) + '_' + df['volume_regime'].astype(str)

    # Check regime sizes

    regime_counts = df.groupby(['year_quarter', 'regime']).size().reset_index(name='count')

    small_regimes = regime_counts[regime_counts['count'] < 30]

    if len(small_regimes) > 0:

        logger.warning(f"Found {len(small_regimes)} quarter-regime cells with <30 events")

    logger.info(f"Created {df['regime'].nunique()} unique regimes across quarters")

    return df

def compute_regime_aware_pressure(event_features, stock_chars):

    """

    Regime-aware normalization with critical fixes:

    1. Feature definitions identical to Method B/C

    2. Regimes based on pre-determined characteristics (not outcomes)

    3. Within quarter-regime normalization

    4. Proper direction for all components

    """

    df = define_regimes(event_features, stock_chars)

    # Information components (same as B/C)

    df['spread_stability_inv'] = 1 / (df['spread_stability_mean'] + 1e-8)

    df['ofi_mean_abs'] = df['ofi_mean'].abs()

    info_cols = ['ofi_mean_abs', 'ofi_autocorr', 'spread_stability_inv']

    # Liquidity components

    liq_cols = ['ofi_abs_mean', 'volume_burst_fraction', 'spread_std_mean', 'spread_p95_max']

    # Within quarter-regime robust normalization

    def robust_zscore_within_regime(group, cols):

        for col in cols:

            median = group[col].median()

            mad = (group[col] - median).abs().median()

            if mad > 0:

                group[f'{col}_norm'] = (group[col] - median) / (1.4826 * mad)

            else:

                group[f'{col}_norm'] = 0

        return group

    # Apply normalization within each quarter-regime cell

    df = df.groupby(['year_quarter', 'regime'], group_keys=False).apply(

        lambda g: robust_zscore_within_regime(g, info_cols + liq_cols)

    )

    # Aggregate normalized scores

    df['info_score'] = df[[f'{col}_norm' for col in info_cols]].mean(axis=1, skipna=True)

    df['liq_score'] = df[[f'{col}_norm' for col in liq_cols]].mean(axis=1, skipna=True)

    # Drop events with insufficient data

    initial_count = len(df)

    df = df.dropna(subset=['info_score', 'liq_score'])

    logger.info(f"Dropped {initial_count - len(df)} events due to insufficient data")

    # Pressure score

    df['pressure_delta'] = df['info_score'] - df['liq_score']

    df['pressure_score'] = np.tanh(df['pressure_delta'])

    # Diagnostics

    logger.info(f"\nPressure score statistics:")

    logger.info(f"  Mean={df['pressure_score'].mean():.3f}, "

                f"Std={df['pressure_score'].std():.3f}")

    # Within-regime normalization validation

    logger.info(f"\nWithin-regime normalization check (median of normalized components):")

    for col in info_cols[:2]:  # Check first 2 info components

        median_by_regime = df.groupby('regime')[f'{col}_norm'].median()

        logger.info(f"  {col}_norm median range: [{median_by_regime.min():.3f}, {median_by_regime.max():.3f}]")

    # Cross-regime comparison

    logger.info(f"\nPressure score by regime:")

    regime_stats = df.groupby('regime').agg({

        'pressure_score': ['mean', 'std', 'count']

    }).round(3)

    logger.info(f"\n{regime_stats}")

    return df

def main():

    logger.info("Starting regime-aware pressure score construction (REVISED)...")

    # Load data

    event_table, daily_features = load_data()

    # Compute regime variables (pre-determined)

    stock_chars = compute_regime_variables(daily_features)

    # Aggregate to event level

    event_features = aggregate_to_event_level(daily_features)

    # Merge with event metadata
    event_features = event_features.merge(
        event_table[['event_id', 'rdq', 'datadate']], 
        on='event_id',
        how='left'
    )

    # Compute regime-aware pressure scores
    event_features = compute_regime_aware_pressure(event_features, stock_chars)

    # Save

    output_path = Path('D_event_features.parquet')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_features.to_parquet(output_path, index=False)

    logger.info(f"\nSaved to {output_path}")

    logger.info(f"Final sample: {len(event_features)} events")

if __name__ == '__main__':

    main()
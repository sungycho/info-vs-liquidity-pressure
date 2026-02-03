"""

C_build_event_features.py (REVISED)

Projection-based / Residualized pressure score (Method 2: Defensive Specification)

"""

import pandas as pd

import numpy as np

from pathlib import Path

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

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

    """Actual lag-1 autocorrelation (same as Method B)"""

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

    """Fraction of burst days (same as Method B)"""

    burst_threshold = 1.5

    burst_df = daily_features.groupby('event_id', group_keys=False).apply(

        lambda g: (g['abnormal_vol_ratio'] > burst_threshold).mean()

    ).rename('volume_burst_fraction').reset_index()

    logger.info(f"Volume burst fraction computed for {len(burst_df)} events")

    return burst_df

def aggregate_to_event_level(daily_features):

    """Aggregate daily features (consistent with Method B)"""

    ofi_autocorr_df = compute_ofi_autocorr(daily_features)

    volume_burst_df = compute_volume_burst_fraction(daily_features)

    event_agg = daily_features.groupby('event_id').agg({

        'ofi': 'mean',

        'ofi_std': 'mean',

        'spread_stability': 'mean',

        'morning_share': 'mean',

        'ofi_abs': 'mean',

        'spread_std': 'mean',

        'spread_p95': 'max',

        'volume': 'mean',

        'num_trades': 'mean',

        'permno': 'first'

    }).reset_index()

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

    event_agg = event_agg.merge(ofi_autocorr_df, on='event_id', how='left')

    event_agg = event_agg.merge(volume_burst_df, on='event_id', how='left')

    logger.info(f"Aggregated to {len(event_agg)} events")

    return event_agg

def compute_residualized_pressure(event_features):

    """

    Residualized pressure score with critical fixes:

    1. Feature definitions identical to Method B

    2. Quarter-level residualization (not global pooled)

    3. Robust standardization within quarters

    4. Focus on pressure_score_pure = tanh(info_resid)

    """

    df = event_features.copy()

    # Create quarter identifier

    df['year_quarter'] = df['rdq'].dt.to_period('Q')

    # Information components (same as Method B)

    df['spread_stability_inv'] = 1 / (df['spread_stability_mean'] + 1e-8)

    df['ofi_mean_abs'] = df['ofi_mean'].abs()

    info_cols = ['ofi_mean_abs', 'ofi_autocorr', 'spread_stability_inv']

    # Liquidity components

    liq_cols = ['ofi_abs_mean', 'volume_burst_fraction', 'spread_std_mean', 'spread_p95_max']

    # Within-quarter robust standardization

    def robust_standardize(group, cols):

        for col in cols:

            median = group[col].median()

            mad = (group[col] - median).abs().median()

            if mad > 0:

                group[f'{col}_std'] = (group[col] - median) / (1.4826 * mad)

            else:

                group[f'{col}_std'] = 0

        return group

    df = df.groupby('year_quarter', group_keys=False).apply(

        lambda g: robust_standardize(g, info_cols + liq_cols)

    )

    # Quarter-level residualization

    info_residuals = []

    for info_col in info_cols:

        df[f'{info_col}_resid'] = np.nan

        for quarter in df['year_quarter'].unique():

            mask = df['year_quarter'] == quarter

            quarter_data = df[mask]

            # Skip if insufficient data

            if len(quarter_data) < 10:

                continue

            X = quarter_data[[f'{col}_std' for col in liq_cols]].values

            y = quarter_data[f'{info_col}_std'].values

            # Check for NaN

            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))

            if valid_mask.sum() < 5:

                continue

            X_valid = X[valid_mask]

            y_valid = y[valid_mask]

            # Fit regression

            reg = LinearRegression()

            reg.fit(X_valid, y_valid)

            # Compute residuals for all (using prediction for valid, NaN for invalid)

            residuals = np.full(len(quarter_data), np.nan)

            residuals[valid_mask] = y_valid - reg.predict(X_valid)

            df.loc[mask, f'{info_col}_resid'] = residuals

            # Log R² for this quarter

            r2 = reg.score(X_valid, y_valid)

            logger.info(f"{quarter} - {info_col}: R²={r2:.3f}, n={valid_mask.sum()}")

        info_residuals.append(f'{info_col}_resid')

    # Aggregate residualized info score

    df['info_score_resid'] = df[info_residuals].mean(axis=1, skipna=True)

    # Aggregate liquidity score

    df['liq_score'] = df[[f'{col}_std' for col in liq_cols]].mean(axis=1, skipna=True)

    # Drop events with insufficient residual data

    initial_count = len(df)

    df = df.dropna(subset=['info_score_resid'])

    logger.info(f"Dropped {initial_count - len(df)} events due to insufficient residual data")

    # Primary pressure score: pure residual (most defensive)

    df['pressure_score_pure'] = np.tanh(df['info_score_resid'])

    # Secondary: residual minus liquidity (for comparison)

    df['pressure_score_hybrid'] = np.tanh(df['info_score_resid'] - df['liq_score'])

    # Main output

    df['pressure_score'] = df['pressure_score_pure']  # Use pure as default

    # Diagnostics

    logger.info(f"\nPressure score statistics:")

    logger.info(f"  Pure (info_resid): mean={df['pressure_score_pure'].mean():.3f}, "

                f"std={df['pressure_score_pure'].std():.3f}")

    logger.info(f"  Hybrid (resid - liq): mean={df['pressure_score_hybrid'].mean():.3f}, "

                f"std={df['pressure_score_hybrid'].std():.3f}")

    # Orthogonality check

    logger.info(f"\nOrthogonality validation:")

    logger.info(f"  info_score_resid vs liq_score: {df['info_score_resid'].corr(df['liq_score']):.3f}")

    return df

def main():

    logger.info("Starting residualized pressure score construction (REVISED)...")

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

    # Compute residualized pressure scores

    event_features = compute_residualized_pressure(event_features)

    # Save

    output_path = Path('SC/C_event_features.parquet')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    event_features.to_parquet(output_path, index=False)

    logger.info(f"\nSaved to {output_path}")

    logger.info(f"Final sample: {len(event_features)} events")

if __name__ == '__main__':

    main()
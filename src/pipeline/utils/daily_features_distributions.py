"""
Analyze daily_features.parquet distributions.

Quick diagnostic of all columns to spot issues before aggregation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style("whitegrid")
DAILY_FEATURES_PATH = "data/processed/daily_features.parquet"

def load_data():
    """Load daily features."""
    print("Loading daily_features.parquet...")
    df = pd.read_parquet(DAILY_FEATURES_PATH)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Events: {df['event_id'].nunique()}")
    return df

def summary_statistics(df: pd.DataFrame):
    """Print summary stats for all numeric columns."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in ['event_id', 'permno', 'event_window_day']:
            continue  # Skip ID columns
        
        stats = df[col].describe()
        missing = df[col].isna().sum()
        
        print(f"\n{col}:")
        print(f"  Count:   {stats['count']:.0f} (missing: {missing})")
        print(f"  Mean:    {stats['mean']:.6f}")
        print(f"  Std:     {stats['std']:.6f}")
        print(f"  Min:     {stats['min']:.6f}")
        print(f"  Q25:     {stats['25%']:.6f}")
        print(f"  Median:  {stats['50%']:.6f}")
        print(f"  Q75:     {stats['75%']:.6f}")
        print(f"  Max:     {stats['max']:.6f}")
        
        # Flag potential issues
        if stats['std'] == 0:
            print(f"  ⚠️  ZERO VARIANCE")
        if abs(stats['mean']) > 10 * stats['std']:
            print(f"  ⚠️  EXTREME MEAN (potential outliers)")
        if stats['max'] > 10 * stats['75%']:
            print(f"  ⚠️  HEAVY RIGHT TAIL")

def plot_distributions(df: pd.DataFrame, output_dir: str = "outputs/diagnostics"):
    """Plot histograms for all numeric columns."""
    print("\n" + "="*80)
    print("PLOTTING DISTRIBUTIONS")
    print("="*80)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get numeric columns (exclude IDs)
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in ['event_id', 'permno', 'event_window_day']]
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Remove outliers for better visualization
        data = df[col].dropna()
        q01 = data.quantile(0.00)
        q99 = data.quantile(1.00)
        data_clipped = data[(data >= q01) & (data <= q99)]
        
        # Plot
        ax.hist(data_clipped, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label='Median')
        
        ax.set_title(f'{col}\n(1-99th percentile)', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = f"{output_dir}/daily_features_distributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def plot_key_features(df: pd.DataFrame, output_dir: str = "outputs/diagnostics"):
    """Detailed plots for key microstructure features."""
    print("\n" + "="*80)
    print("KEY FEATURES DEEP DIVE")
    print("="*80)
    
    key_features = ['ofi', 'ofi_abs', 'spread_stability', 'spread_std', 
                    'volume', 'abnormal_vol_ratio', 'morning_share']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for idx, col in enumerate(key_features):
        if col not in df.columns:
            continue
        
        ax = axes[idx]
        data = df[col].dropna()
        
        # Main histogram
        ax.hist(data, bins=100, alpha=0.6, edgecolor='black', density=True)
        
        # Overlay statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1σ: {std_val:.4f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(key_features), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    output_path = f"{output_dir}/key_features_detailed.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()

def check_extreme_values(df: pd.DataFrame):
    """Identify and report extreme values."""
    print("\n" + "="*80)
    print("EXTREME VALUE DETECTION")
    print("="*80)
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col not in ['event_id', 'permno', 'event_window_day']]
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Calculate outlier thresholds
        q01 = data.quantile(0.01)
        q99 = data.quantile(0.99)
        iqr = data.quantile(0.75) - data.quantile(0.25)
        lower_fence = data.quantile(0.25) - 3 * iqr
        upper_fence = data.quantile(0.75) + 3 * iqr
        
        # Count outliers
        extreme_low = (data < lower_fence).sum()
        extreme_high = (data > upper_fence).sum()
        
        if extreme_low > 0 or extreme_high > 0:
            pct_low = 100 * extreme_low / len(data)
            pct_high = 100 * extreme_high / len(data)
            
            print(f"\n{col}:")
            print(f"  Extreme low (<{lower_fence:.4f}): {extreme_low} ({pct_low:.2f}%)")
            print(f"  Extreme high (>{upper_fence:.4f}): {extreme_high} ({pct_high:.2f}%)")
            print(f"  P01: {q01:.4f}, P99: {q99:.4f}")
            
            if pct_low + pct_high > 5:
                print(f"  ⚠️  >5% extreme values - consider winsorization")

def correlation_matrix(df: pd.DataFrame, output_dir: str = "outputs/diagnostics"):
    """Plot correlation matrix of key features."""
    print("\n" + "="*80)
    print("CORRELATION MATRIX")
    print("="*80)
    
    key_features = ['ofi', 'ofi_abs', 'spread_stability', 'spread_std', 
                    'volume', 'abnormal_vol_ratio', 'morning_share', 'num_trades']
    
    # Filter available columns
    key_features = [col for col in key_features if col in df.columns]
    
    corr = df[key_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = f"{output_dir}/correlation_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    # Print high correlations
    print("\n  High correlations (|r| > 0.7):")
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            if abs(corr.iloc[i, j]) > 0.7:
                print(f"    {corr.index[i]} <-> {corr.columns[j]}: {corr.iloc[i, j]:.3f}")

def main():
    """Run all diagnostics."""
    df = load_data()
    
    # Summary stats
    summary_statistics(df)
    
    # Plots
    plot_distributions(df)
    plot_key_features(df)
    correlation_matrix(df)
    
    # Extreme values
    check_extreme_values(df)
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80)
    print("Check outputs/diagnostics/ for plots")

if __name__ == "__main__":
    main()
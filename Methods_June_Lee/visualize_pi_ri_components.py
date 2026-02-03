"""
Visualize PI and RI component distributions from daily_features.parquet.

This script:
1. Loads daily_features.parquet
2. Computes PI and RI components per event using persistence_reversal.py
3. Creates distribution plots for all 6 components
4. Shows summary statistics and correlations

Author: Cascade
Date: 2025-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append('src')
from features.persistence_reversal import compute_pi_ri

# Setup
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
DAILY_FEATURES_PATH = "data/processed/daily_features.parquet"
OUTPUT_DIR = "outputs"

def load_data():
    """Load daily features data."""
    print("Loading daily_features.parquet...")
    df = pd.read_parquet(DAILY_FEATURES_PATH)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Events: {df['event_id'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df

def compute_pi_ri_components(df):
    """Compute PI and RI components for all events."""
    print("\nComputing PI and RI components...")
    
    # Use the persistence_reversal module
    pi_ri_df = compute_pi_ri(
        df, 
        event_col="event_id", 
        date_col="date", 
        ofi_col="ofi",
        min_days=7
    )
    
    print(f"  Computed for {len(pi_ri_df)} events")
    print(f"  Valid PI: {pi_ri_df['PI'].notna().sum()}")
    print(f"  Valid RI: {pi_ri_df['RI'].notna().sum()}")
    
    return pi_ri_df

def plot_component_distributions(pi_ri_df):
    """Create distribution plots for all 6 components."""
    print("\nCreating component distribution plots...")
    
    # Component names and labels
    pi_components = [
        ('pi_rho1_01', 'Level Autocorrelation\n(rho₁₀₁)'),
        ('pi_consistency', 'Directional Consistency\n(C)'),
        ('pi_drift_ratio', 'Drift Ratio\n(D)')
    ]
    
    ri_components = [
        ('ri_gamma_01', 'Difference Reversal\n(Γ₀₁)'),
        ('ri_flip_rate', 'Flip Rate\n(F)'),
        ('ri_alt_intensity', 'Alternation Intensity\n(A)')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PI and RI Component Distributions', fontsize=16, fontweight='bold')
    
    # Plot PI components (top row)
    for i, (col, label) in enumerate(pi_components):
        ax = axes[0, i]
        data = pi_ri_df[col].dropna()
        
        # Histogram
        ax.hist(data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'PI: {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=9)
    
    # Plot RI components (bottom row)
    for i, (col, label) in enumerate(ri_components):
        ax = axes[1, i]
        data = pi_ri_df[col].dropna()
        
        # Histogram
        ax.hist(data, bins=30, alpha=0.7, color='darkorange', edgecolor='black')
        ax.set_title(f'RI: {label}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        std_val = data.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    output_path = Path(OUTPUT_DIR) / "pi_ri_component_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")
    
    return fig

def plot_summary_statistics(pi_ri_df):
    """Create summary statistics heatmap."""
    print("\nCreating summary statistics...")
    
    # Component columns
    component_cols = [
        'pi_rho1_01', 'pi_consistency', 'pi_drift_ratio',
        'ri_gamma_01', 'ri_flip_rate', 'ri_alt_intensity'
    ]
    
    # Compute summary stats
    stats_df = pi_ri_df[component_cols].describe().T
    stats_df = stats_df[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    print("\nComponent Summary Statistics:")
    print(stats_df.round(4))
    
    # Create correlation heatmap
    corr_df = pi_ri_df[component_cols].corr()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": 0.8})
    
    ax.set_title('Component Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Save plot
    output_path = Path(OUTPUT_DIR) / "pi_ri_component_correlations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved correlation plot to: {output_path}")
    
    return fig, stats_df

def analyze_pi_ri_relationships(pi_ri_df):
    """Analyze relationships between PI, RI and their components."""
    print("\nAnalyzing PI/RI relationships...")
    
    # Filter for valid PI and RI
    valid_df = pi_ri_df.dropna(subset=['PI', 'RI'])
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PI and RI Relationships', fontsize=16, fontweight='bold')
    
    # PI vs RI scatter
    ax = axes[0, 0]
    ax.scatter(valid_df['PI'], valid_df['RI'], alpha=0.6, s=30)
    ax.set_xlabel('Persistence Index (PI)')
    ax.set_ylabel('Reversibility Index (RI)')
    ax.set_title('PI vs RI')
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    corr = valid_df['PI'].corr(valid_df['RI'])
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # PI distribution
    ax = axes[0, 1]
    ax.hist(valid_df['PI'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Persistence Index (PI)')
    ax.set_ylabel('Frequency')
    ax.set_title('PI Distribution')
    ax.grid(True, alpha=0.3)
    
    # RI distribution
    ax = axes[1, 0]
    ax.hist(valid_df['RI'], bins=30, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Reversibility Index (RI)')
    ax.set_ylabel('Frequency')
    ax.set_title('RI Distribution')
    ax.grid(True, alpha=0.3)
    
    # Component contributions to PI
    pi_comp_cols = ['pi_rho1_01', 'pi_consistency', 'pi_drift_ratio']
    pi_contributions = valid_df[pi_comp_cols].mean()
    
    ax = axes[1, 1]
    bars = ax.bar(['Level\nAutocorr', 'Directional\nConsistency', 'Drift\nRatio'], 
                   pi_contributions, color=['steelblue', 'darkblue', 'lightblue'])
    ax.set_ylabel('Average Component Value')
    ax.set_title('PI Component Contributions')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, pi_contributions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(OUTPUT_DIR) / "pi_ri_relationships.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved relationships plot to: {output_path}")
    
    return fig

def main():
    """Main execution."""
    print("="*60)
    print("PI and RI Component Visualization")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Compute PI/RI components
    pi_ri_df = compute_pi_ri_components(df)
    
    # Create visualizations
    plot_component_distributions(pi_ri_df)
    plot_summary_statistics(pi_ri_df)
    analyze_pi_ri_relationships(pi_ri_df)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Events analyzed: {len(pi_ri_df)}")
    print(f"Plots saved:")
    print(f"  - pi_ri_component_distributions.png")
    print(f"  - pi_ri_component_correlations.png") 
    print(f"  - pi_ri_relationships.png")

if __name__ == "__main__":
    main()

"""
Earnings Announcement Timing Checker
-------------------------------------
Identify whether earnings were announced Before Market Open (BMO) or 
After Market Close (AMC) using Compustat anntims_act field.

This helps validate whether t=+1 entry timing is more appropriate.
"""

import pandas as pd
import wrds
from pathlib import Path


def get_earnings_timing(
    event_file: str = "data/processed/event_table.parquet",
    save_path: str = "data/processed/earnings_timing.parquet"
):
    """
    Get earnings announcement timing (AMC/BMO) from Compustat.
    
    Compustat anntims_act codes:
    - 1 = Before Market Open (BMO)
    - 2 = During Market Hours
    - 3 = After Market Close (AMC)
    - 4 = Unknown/Not Available
    """
    
    print("="*70)
    print("Earnings Announcement Timing Checker")
    print("="*70)
    
    # Load event table
    print(f"\nLoading event data: {event_file}")
    
    try:
        events = pd.read_parquet(event_file)
        
        # Handle duplicates
        if 'event_id' in events.columns and events['event_id'].duplicated().any():
            events = events.drop_duplicates(subset='event_id', keep='first')
        
        print(f"✓ Loaded {len(events)} unique events")
        
    except FileNotFoundError:
        print(f"✗ File not found: {event_file}")
        return None
    
    # Find date column
    date_col = None
    for col in ['rdq', 'earnings_date', 'event_date', 'date']:
        if col in events.columns:
            date_col = col
            break
    
    if date_col is None:
        print("✗ Could not find date column")
        return None
    
    # Get unique gvkey-datadate pairs for lookup
    if 'gvkey' not in events.columns or 'datadate' not in events.columns:
        print("✗ Missing gvkey or datadate columns")
        print("  Available columns:", events.columns.tolist())
        return None
    
    # Connect to WRDS
    print("\nConnecting to WRDS...")
    
    try:
        db = wrds.Connection()
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None
    
    # Get timing data from Compustat
    print("\nQuerying Compustat for announcement timing...")
    
    # Create list of (gvkey, datadate) pairs
    lookup_pairs = events[['gvkey', 'datadate']].drop_duplicates()
    
    print(f"  Unique (gvkey, datadate) pairs: {len(lookup_pairs)}")
    
    # Build query
    # Note: anntims_act is the actual announcement time
    # rdq is the report date (usually same as announcement date)
    
    gvkey_list = "','".join(lookup_pairs['gvkey'].astype(str).unique())
    
    query = f"""
    SELECT 
        gvkey,
        datadate,
        rdq,
        anntims_act,
        CASE 
            WHEN anntims_act = 1 THEN 'BMO'
            WHEN anntims_act = 2 THEN 'During_Market'
            WHEN anntims_act = 3 THEN 'AMC'
            ELSE 'Unknown'
        END as timing_label
    FROM comp.fundq
    WHERE gvkey IN ('{gvkey_list}')
        AND rdq IS NOT NULL
        AND datadate >= '2022-01-01'
    """
    
    try:
        timing_df = db.raw_sql(query)
        db.close()
        print(f"✓ Retrieved timing data for {len(timing_df)} quarters")
    except Exception as e:
        db.close()
        print(f"✗ Query failed: {e}")
        return None
    
    # Convert dates
    timing_df['datadate'] = pd.to_datetime(timing_df['datadate'])
    timing_df['rdq'] = pd.to_datetime(timing_df['rdq'])
    
    # Merge with events
    print("\nMerging with event data...")
    
    events['datadate'] = pd.to_datetime(events['datadate'])
    
    merged = events.merge(
        timing_df[['gvkey', 'datadate', 'anntims_act', 'timing_label']],
        on=['gvkey', 'datadate'],
        how='left'
    )
    
    # Summary statistics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nTotal events: {len(merged)}")
    
    timing_counts = merged['timing_label'].value_counts(dropna=False)
    print(f"\nTiming breakdown:")
    for label, count in timing_counts.items():
        pct = 100 * count / len(merged)
        print(f"  {label:<15}: {count:>5} ({pct:>5.1f}%)")
    
    # Coverage
    has_timing = merged['timing_label'].notna() & (merged['timing_label'] != 'Unknown')
    coverage = 100 * has_timing.sum() / len(merged)
    print(f"\nData coverage: {has_timing.sum()}/{len(merged)} ({coverage:.1f}%)")
    
    # Key insight for t=0 vs t=+1
    amc_count = (merged['timing_label'] == 'AMC').sum()
    bmo_count = (merged['timing_label'] == 'BMO').sum()
    
    if amc_count + bmo_count > 0:
        amc_pct = 100 * amc_count / (amc_count + bmo_count)
        print(f"\n{'='*70}")
        print("KEY INSIGHT")
        print(f"{'='*70}")
        print(f"\nOf announcements with known timing:")
        print(f"  AMC (After Market Close): {amc_count} ({amc_pct:.1f}%)")
        print(f"  BMO (Before Market Open):  {bmo_count} ({100-amc_pct:.1f}%)")
        
        print(f"\n→ For AMC announcements, t=+1 entry is more appropriate")
        print(f"  (information only reflected next trading day)")
        print(f"→ For BMO announcements, t=0 entry captures same-day reaction")
    
    # Save results
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(save_path, index=False)
        print(f"\n✓ Saved results to: {save_path}")
    
    return merged


def analyze_timing_impact(
    timing_file: str = "data/processed/earnings_timing.parquet",
    backtest_results_file: str = "data/results/backtest_results_t1_h20.parquet"
):
    """
    Analyze whether AMC vs BMO timing affects backtest results.
    """
    
    print("\n" + "="*70)
    print("TIMING IMPACT ANALYSIS")
    print("="*70)
    
    # Load timing data
    timing_df = pd.read_parquet(timing_file)
    
    # Load backtest results
    backtest_df = pd.read_parquet(backtest_results_file)
    
    # Merge
    merged = backtest_df.merge(
        timing_df[['event_id', 'timing_label']],
        on='event_id',
        how='left'
    )
    
    # Analyze returns by timing
    print("\nAverage returns by announcement timing (20d horizon):")
    
    for timing in ['AMC', 'BMO', 'During_Market', 'Unknown']:
        subset = merged[merged['timing_label'] == timing]['cum_return'].dropna()
        if len(subset) > 0:
            mean_ret = subset.mean()
            print(f"  {timing:<15}: {mean_ret:>8.4f} ({mean_ret*100:>6.2f}%) [n={len(subset)}]")
    
    # Check if extreme bins differ by timing
    print("\nExtreme bins by timing:")
    
    # Top 15%
    top_15_threshold = merged['Pressure_Score'].quantile(0.85)
    bottom_15_threshold = merged['Pressure_Score'].quantile(0.15)
    
    for timing in ['AMC', 'BMO']:
        timing_subset = merged[merged['timing_label'] == timing]
        
        if len(timing_subset) > 0:
            top_ret = timing_subset[timing_subset['Pressure_Score'] >= top_15_threshold]['cum_return'].mean()
            bottom_ret = timing_subset[timing_subset['Pressure_Score'] <= bottom_15_threshold]['cum_return'].mean()
            spread = top_ret - bottom_ret
            
            print(f"\n  {timing}:")
            print(f"    Top 15% return:    {top_ret:>8.4f} ({top_ret*100:>6.2f}%)")
            print(f"    Bottom 15% return: {bottom_ret:>8.4f} ({bottom_ret*100:>6.2f}%)")
            print(f"    Spread:            {spread:>8.4f} ({spread*100:>6.2f}%)")


if __name__ == "__main__":
    
    # Step 1: Get timing data
    timing_df = get_earnings_timing(
        event_file="data/processed/event_table.parquet",
        save_path="data/processed/earnings_timing.parquet"
    )
    
    if timing_df is not None:
        print("\n" + "="*70)
        print("Next Steps")
        print("="*70)
        print("\n1. Review timing breakdown above")
        print("2. If you want detailed impact analysis, run:")
        print("   from this_script import analyze_timing_impact")
        print("   analyze_timing_impact()")
        print("\n3. Consider splitting backtest by AMC vs BMO")
        print("   (AMC events should use t=+1, BMO should use t=0)")
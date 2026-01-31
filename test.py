"""
Earnings Announcement Timing Checker - FIXED VERSION
-----------------------------------------------------
Using comp.wrds_annc_prelim table which contains anntims field.

Based on: Compustat Preliminary History table has announcement timestamps.
"""

import pandas as pd
import wrds
from pathlib import Path
from datetime import time


def get_earnings_timing_fixed(
    event_file: str = "data/processed/event_table.parquet",
    save_path: str = "data/processed/earnings_timing.parquet"
):
    """
    Get earnings announcement timing using comp.wrds_annc_prelim.
    
    Field: anntims (announcement timestamp)
    Logic:
    - Before 09:30 ET → BMO (Before Market Open)
    - After 16:00 ET → AMC (After Market Close)
    - Between 09:30-16:00 → During Market
    """
    
    print("="*70)
    print("Earnings Timing Checker - Using comp.wrds_annc_prelim")
    print("="*70)
    
    # Load events
    print(f"\nLoading: {event_file}")
    events = pd.read_parquet(event_file)
    
    if 'event_id' in events.columns and events['event_id'].duplicated().any():
        events = events.drop_duplicates(subset='event_id', keep='first')
    
    print(f"✓ Loaded {len(events)} unique events")
    
    # Check for required columns
    if 'gvkey' not in events.columns or 'datadate' not in events.columns:
        print("✗ Missing gvkey or datadate columns")
        return None
    
    # Connect to WRDS
    print("\nConnecting to WRDS...")
    try:
        db = wrds.Connection()
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return None
    
    # Get unique gvkey-datadate pairs
    lookup_pairs = events[['gvkey', 'datadate']].drop_duplicates()
    print(f"  Unique (gvkey, datadate) pairs: {len(lookup_pairs)}")
    
    # Build query for comp.wrds_annc_prelim
    gvkey_list = "','".join(lookup_pairs['gvkey'].astype(str).unique())
    
    query = f"""
    SELECT 
        gvkey,
        datadate,
        rdq,
        anntims,
        EXTRACT(HOUR FROM anntims) * 60 + EXTRACT(MINUTE FROM anntims) as anntime_minutes
    FROM comp.wrds_annc_prelim
    WHERE gvkey IN ('{gvkey_list}')
        AND datadate >= '2022-01-01'
        AND anntims IS NOT NULL
    """
    
    print("\nQuerying comp.wrds_annc_prelim for announcement timestamps...")
    
    try:
        timing_df = db.raw_sql(query)
        db.close()
        print(f"✓ Retrieved {len(timing_df)} records")
    except Exception as e:
        db.close()
        print(f"✗ Query failed: {e}")
        print("\nTrying alternative: ciq.ciqtranscript table...")
        return try_ciq_alternative(events, event_file, save_path)
    
    # Convert dates
    timing_df['datadate'] = pd.to_datetime(timing_df['datadate'])
    timing_df['rdq'] = pd.to_datetime(timing_df['rdq'])
    timing_df['anntims'] = pd.to_datetime(timing_df['anntims'])
    
    # Classify timing based on hour
    # Market hours: 09:30 - 16:00 ET
    def classify_timing(minutes):
        if pd.isna(minutes):
            return 'Unknown'
        
        # Convert to minutes since midnight
        # 09:30 = 570 minutes
        # 16:00 = 960 minutes
        
        if minutes < 570:  # Before 9:30 AM
            return 'BMO'
        elif minutes >= 960:  # After 4:00 PM
            return 'AMC'
        else:  # During market hours
            return 'During_Market'
    
    timing_df['timing_label'] = timing_df['anntime_minutes'].apply(classify_timing)
    
    # Merge with events
    print("\nMerging with event data...")
    events['datadate'] = pd.to_datetime(events['datadate'])
    
    merged = events.merge(
        timing_df[['gvkey', 'datadate', 'anntims', 'timing_label']],
        on=['gvkey', 'datadate'],
        how='left'
    )
    
    # Summary
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
    
    # Key insight
    amc_count = (merged['timing_label'] == 'AMC').sum()
    bmo_count = (merged['timing_label'] == 'BMO').sum()
    
    if amc_count + bmo_count > 0:
        amc_pct = 100 * amc_count / (amc_count + bmo_count)
        
        print(f"\n{'='*70}")
        print("KEY INSIGHT FOR BACKTESTING")
        print(f"{'='*70}")
        print(f"\nOf announcements with known timing:")
        print(f"  AMC (After Market Close): {amc_count} ({amc_pct:.1f}%)")
        print(f"  BMO (Before Market Open):  {bmo_count} ({100-amc_pct:.1f}%)")
        
        print(f"\n→ Implication for entry timing:")
        if amc_pct > 60:
            print(f"  Most events are AMC → t=+1 entry is more realistic")
        elif amc_pct < 40:
            print(f"  Most events are BMO → t=0 entry captures same-day reaction")
        else:
            print(f"  Mixed timing → consider splitting backtest by AMC/BMO")
    
    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(save_path, index=False)
        print(f"\n✓ Saved to: {save_path}")
    
    return merged


def try_ciq_alternative(events, event_file, save_path):
    """
    Fallback: Try Capital IQ transcript table.
    """
    print("\nAttempting Capital IQ alternative...")
    print("(This requires Capital IQ subscription)")
    
    try:
        db = wrds.Connection()
        
        # Try ciq.ciqtranscript
        test_query = """
        SELECT COUNT(*) 
        FROM ciq.ciqtranscript 
        LIMIT 1
        """
        
        db.raw_sql(test_query)
        print("✓ Capital IQ access confirmed")
        
        # Actual query would go here
        # But structure varies - need to map companyId to gvkey first
        
        db.close()
        print("\n⚠ Capital IQ table exists but mapping is complex")
        print("  Recommend using return-based inference instead")
        
    except Exception as e:
        print(f"✗ Capital IQ not accessible: {e}")
    
    return None


def analyze_timing_by_backtest(
    timing_file: str = "data/processed/earnings_timing.parquet"
):
    """
    Analyze how AMC vs BMO affects backtest results.
    """
    
    print("\n" + "="*70)
    print("TIMING IMPACT ON BACKTEST RESULTS")
    print("="*70)
    
    # Load timing
    timing_df = pd.read_parquet(timing_file)
    
    # Load backtest results
    backtest_files = {
        't=0': 'data/results/backtest_results_t0_h20.parquet',
        't=+1': 'data/results/backtest_results_t1_h20.parquet'
    }
    
    for entry, filepath in backtest_files.items():
        if not Path(filepath).exists():
            print(f"\n⚠ Backtest file not found: {filepath}")
            continue
        
        backtest = pd.read_parquet(filepath)
        merged = backtest.merge(
            timing_df[['event_id', 'timing_label']],
            on='event_id',
            how='left'
        )
        
        print(f"\n{entry} entry, 20-day horizon:")
        print("-" * 50)
        
        for timing in ['AMC', 'BMO', 'During_Market']:
            subset = merged[merged['timing_label'] == timing]
            if len(subset) > 0:
                mean_ret = subset['cum_return'].mean()
                n = len(subset)
                print(f"  {timing:<15}: {mean_ret:>8.4f} ({mean_ret*100:>6.2f}%) [n={n}]")
        
        # Extreme bins by timing
        print(f"\n  Top 15% vs Bottom 15% by timing:")
        
        top_thresh = merged['Pressure_Score'].quantile(0.85)
        bot_thresh = merged['Pressure_Score'].quantile(0.15)
        
        for timing in ['AMC', 'BMO']:
            timing_subset = merged[merged['timing_label'] == timing]
            if len(timing_subset) > 0:
                top = timing_subset[timing_subset['Pressure_Score'] >= top_thresh]['cum_return'].mean()
                bot = timing_subset[timing_subset['Pressure_Score'] <= bot_thresh]['cum_return'].mean()
                spread = top - bot
                
                print(f"    {timing}: {spread:>8.4f} ({spread*100:>6.2f}%)")


if __name__ == "__main__":
    
    # Get timing data
    timing_df = get_earnings_timing_fixed()
    
    if timing_df is not None:
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Review timing breakdown above")
        print("2. To analyze impact on backtest:")
        print("   from this_script import analyze_timing_by_backtest")
        print("   analyze_timing_by_backtest()")
        print("\n3. Consider re-running backtest split by AMC/BMO")
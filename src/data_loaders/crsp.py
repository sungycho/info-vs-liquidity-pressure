"""
Download CRSP Daily Returns
----------------------------
One-time script to download and save CRSP daily returns for backtesting.

Usage:
    python download_crsp_returns.py
"""

import pandas as pd
import wrds
from pathlib import Path
from datetime import datetime


def download_crsp_returns(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    permno_list: list = None,
    save_path: str = "data/processed/crsp_daily.parquet"
):
    """
    Download CRSP daily stock returns from WRDS.
    
    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    permno_list : list, optional
        List of permnos to download. If None, downloads all.
    save_path : str
        Where to save the parquet file
    """
    
    print("="*70)
    print("CRSP Daily Returns Downloader")
    print("="*70)
    
    # Connect to WRDS
    print("\nConnecting to WRDS...")
    try:
        db = wrds.Connection()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure you have:")
        print("  1. WRDS account credentials configured")
        print("  2. .pgpass file set up (or use wrds.Connection(wrds_username='...'))")
        return None
    
    # Build query
    print(f"\nDownloading CRSP data:")
    print(f"  Date range: {start_date} to {end_date}")
    
    if permno_list:
        print(f"  Permnos: {len(permno_list)} stocks")
        permno_filter = f"AND permno IN ({','.join(map(str, permno_list))})"
    else:
        print(f"  Permnos: ALL (this may take a while...)")
        permno_filter = ""
    
    query = f"""
    SELECT 
        permno,
        date,
        ret,
        prc,
        vol,
        shrout
    FROM crsp.dsf
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
        {permno_filter}
        AND ret IS NOT NULL
    ORDER BY permno, date
    """
    
    print("\nExecuting query...")
    print("(This may take 1-5 minutes depending on date range...)")
    
    try:
        df = db.raw_sql(query)
        db.close()
        print(f"✓ Downloaded {len(df):,} rows")
    except Exception as e:
        db.close()
        print(f"✗ Query failed: {e}")
        return None
    
    # Data summary
    print("\n" + "-"*70)
    print("Data Summary")
    print("-"*70)
    print(f"Total observations: {len(df):,}")
    print(f"Unique stocks (permno): {df['permno'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nReturn statistics:")
    print(f"  Mean: {df['ret'].mean():.6f}")
    print(f"  Std:  {df['ret'].std():.6f}")
    print(f"  Min:  {df['ret'].min():.6f}")
    print(f"  Max:  {df['ret'].max():.6f}")
    
    # Check for missing data
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Save to parquet
    print(f"\nSaving to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(save_path, index=False, compression='snappy')
    
    file_size_mb = Path(save_path).stat().st_size / (1024 * 1024)
    print(f"✓ Saved successfully ({file_size_mb:.1f} MB)")
    
    print("\n" + "="*70)
    print("Download Complete!")
    print("="*70)
    
    return df


def download_for_event_universe(
    event_file: str = "data/processed/event_features.parquet",
    buffer_days: int = 30,
    save_path: str = "data/processed/crsp_daily.parquet"
):
    """
    Download CRSP returns tailored to your event universe.
    
    Reads event_features.parquet to determine:
    - Which permnos are needed
    - What date range is needed (with buffer)
    
    This is more efficient than downloading all CRSP data.
    """
    
    print("="*70)
    print("CRSP Returns - Event Universe Download")
    print("="*70)
    
    # Load events to determine universe
    print(f"\nReading event file: {event_file}")
    
    try:
        events = pd.read_parquet(event_file)
        print(f"✓ Loaded {len(events)} rows")
    except FileNotFoundError:
        print(f"✗ File not found: {event_file}")
        print("  Run this after you have event_features.parquet from Week 2")
        return None
    
    # Deduplicate if needed
    if 'event_id' in events.columns and events['event_id'].duplicated().any():
        events = events.drop_duplicates(subset='event_id', keep='first')
        print(f"  Deduplicated to {len(events)} unique events")
    
    # Extract universe
    permnos = events['permno'].unique().tolist()
    
    # Find date column
    date_col = None
    for col in ['earnings_date', 'rdq', 'event_date', 'date']:
        if col in events.columns:
            date_col = col
            break
    
    if date_col is None:
        print("✗ Could not find date column in event file")
        return None
    
    # Determine date range with buffer
    min_date = pd.to_datetime(events[date_col].min())
    max_date = pd.to_datetime(events[date_col].max())
    
    start_date = (min_date - pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    end_date = (max_date + pd.Timedelta(days=buffer_days)).strftime('%Y-%m-%d')
    
    print(f"\nEvent universe:")
    print(f"  Unique permnos: {len(permnos)}")
    print(f"  Event date range: {min_date.date()} to {max_date.date()}")
    print(f"  Download range (with {buffer_days}d buffer): {start_date} to {end_date}")
    
    # Download
    df = download_crsp_returns(
        start_date=start_date,
        end_date=end_date,
        permno_list=permnos,
        save_path=save_path
    )
    
    return df


if __name__ == "__main__":
    
    print("\nChoose download mode:")
    print("  1. Event universe (recommended - faster, smaller)")
    print("  2. Full date range (downloads all stocks)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Download based on event_features.parquet
        df = download_for_event_universe(
            event_file="data/processed/event_features.parquet",
            buffer_days=30,
            save_path="data/processed/crsp_daily.parquet"
        )
        
    elif choice == "2":
        # Download full range
        start = input("Start date (YYYY-MM-DD, default 2022-01-01): ").strip() or "2022-01-01"
        end = input("End date (YYYY-MM-DD, default 2024-12-31): ").strip() or "2024-12-31"
        
        df = download_crsp_returns(
            start_date=start,
            end_date=end,
            permno_list=None,  # All stocks
            save_path="data/processed/crsp_daily.parquet"
        )
    
    else:
        print("Invalid choice. Exiting.")
        exit(1)
    
    if df is not None:
        print("\n✓ Ready to run backtester!")
        print("  Next step: python src/backtest/event_backtester.py")
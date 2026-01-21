"""
Earnings calendar loader from Compustat + CRSP linking
"""
import wrds
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def load_earnings_calendar(
    start_year: int = 2023,
    end_year: int = 2024,
    min_volume_usd: float = 5_000_000,
    skip_volume_filter: bool = True,
    save_path: str = "data/processed/event_table.parquet"
) -> pd.DataFrame:
    """
    Load earnings announcement dates for S&P 500 constituents.
    
    Args:
        start_year: First year of earnings announcements (by rdq date)
        end_year: Last year of earnings announcements (by rdq date)
        min_volume_usd: Minimum average daily dollar volume filter
        skip_volume_filter: Skip volume filter (default True for S&P 500 - already liquid)
        save_path: Output path for parquet file
        
    Returns:
        DataFrame with columns: event_id, permno, gvkey, ticker, rdq, datadate, fyearq, fqtr
    """
    
    try:
        db = wrds.Connection()
    except Exception as e:
        print(f"✗ Error connecting to WRDS: {e}")
        raise
    
    # Main query with proper S&P 500 membership timing
    query = f"""
    WITH sp500 AS (
        -- S&P 500 constituents during the sample period
        -- Include stocks that were in S&P 500 at any point during our window
        SELECT DISTINCT permno
        FROM crsp.dsp500list
        WHERE start <= '{end_year}-12-31'
            AND ending >= '{start_year}-01-01'
    ),
    
    earnings AS (
        SELECT 
            gvkey,
            datadate,
            rdq,
            fyearq,
            fqtr
        FROM comp.fundq
        WHERE rdq BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
            AND rdq IS NOT NULL
            AND gvkey IS NOT NULL
            AND datadate IS NOT NULL
    ),
    
    linked AS (
        SELECT 
            e.gvkey,
            e.datadate,
            e.rdq,
            e.fyearq,
            e.fqtr,
            lnk.lpermno as permno
        FROM earnings e
        INNER JOIN crsp.ccmxpf_lnkhist lnk
            ON e.gvkey = lnk.gvkey
            AND lnk.linktype IN ('LU', 'LC')
            AND lnk.linkprim IN ('P', 'C')
            AND e.rdq BETWEEN lnk.linkdt AND COALESCE(lnk.linkenddt, CURRENT_DATE)
        INNER JOIN sp500 s 
            ON lnk.lpermno = s.permno
    ),
    
    with_ticker AS (
        SELECT 
            l.gvkey,
            l.permno,
            l.datadate,
            l.rdq,
            l.fyearq,
            l.fqtr,
            n.ticker
        FROM linked l
        INNER JOIN crsp.stocknames n
            ON l.permno = n.permno
            AND l.rdq BETWEEN n.namedt AND n.nameenddt
    )
    
    SELECT DISTINCT
        gvkey,
        permno,
        ticker,
        rdq,
        datadate,
        fyearq,
        fqtr
    FROM with_ticker
    ORDER BY rdq, permno
    """
    
    try:
        print("Fetching earnings calendar from WRDS...")
        df = db.raw_sql(query)
        
        if len(df) == 0:
            print("⚠ Warning: No earnings events found for specified period")
            db.close()
            return pd.DataFrame()
        
        print(f"  Found {len(df)} raw earnings events")
        
        # Volume filter: Skip for S&P 500 (already liquid large-caps)
        # Only apply when extending to broader universe (e.g., Russell 1000)
        if skip_volume_filter:
            print("  Skipping volume filter (S&P 500 constituents are already liquid)")
        else:
            print("Filtering by volume threshold...")
            
            # PERFORMANCE FIX: Filter S&P 500 permnos FIRST, then calculate volume
            # This reduces CRSP daily table scan from ~50M rows to ~375K rows
            sp500_permnos = tuple(df['permno'].unique())
            permno_list = ','.join(map(str, sp500_permnos))
            
            volume_filter_query = f"""
            SELECT 
                permno,
                AVG(abs(prc) * vol) as avg_dollar_volume
            FROM crsp.dsf
            WHERE permno IN ({permno_list})
                AND date BETWEEN '{start_year-1}-01-01' AND '{end_year}-12-31'
            GROUP BY permno
            HAVING AVG(abs(prc) * vol) >= {min_volume_usd}
            """
            
            volume_df = db.raw_sql(volume_filter_query)
            n_before = len(df)
            df = df.merge(volume_df[['permno']], on='permno', how='inner')
            n_filtered = n_before - len(df)
            
            if n_filtered > 0:
                print(f"  Filtered out {n_filtered} events due to low volume")

        
        db.close()
        
        # Clean up dates
        df['rdq'] = pd.to_datetime(df['rdq'])
        df['datadate'] = pd.to_datetime(df['datadate'])
        
        # Create event_id with zero-padding for sorting
        df = df.sort_values(['rdq', 'permno']).reset_index(drop=True)
        df['event_id'] = ['E' + str(i).zfill(4) for i in range(len(df))]
        
        # Reorder columns for clarity
        df = df[['event_id', 'permno', 'gvkey', 'ticker', 'rdq', 'datadate', 'fyearq', 'fqtr']]
        
        # Save to parquet
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(save_path, index=False)
        
        # Summary statistics
        print(f"\n✓ Saved {len(df)} events to {save_path}")
        print(f"  Date range: {df['rdq'].min().date()} to {df['rdq'].max().date()}")
        print(f"  Unique stocks: {df['permno'].nunique()}")
        print(f"  Events per year:")
        for year in sorted(df['rdq'].dt.year.unique()):
            year_count = len(df[df['rdq'].dt.year == year])
            year_stocks = df[df['rdq'].dt.year == year]['permno'].nunique()
            print(f"    {year}: {year_count:,} events across {year_stocks} stocks")
        
        return df
        
    except Exception as e:
        db.close()
        print(f"✗ Error loading earnings data: {e}")
        raise


if __name__ == "__main__":
    # Test run: Load 2023-2024 S&P 500 earnings
    df = load_earnings_calendar(
        start_year=2023,
        end_year=2024,
        skip_volume_filter=True,  # S&P 500 already liquid, skip for speed
        save_path="data/processed/event_table_2023_2024.parquet"
    )
    
    if len(df) > 0:
        print("\n=== Sample Events ===")
        print(df.head(10))
        
        print("\n=== Data Quality Checks ===")
        print(f"Null values:\n{df.isnull().sum()}")
        print(f"\nDuplicate event_ids: {df['event_id'].duplicated().sum()}")
        print(f"Duplicate (permno, rdq) pairs: {df[['permno', 'rdq']].duplicated().sum()}")
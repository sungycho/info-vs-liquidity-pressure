"""
Liquidity friction metrics from TAQ quotes

Responsibility: Spread level and stability only
- Quoted spread (relative and absolute)
- Spread stability (coefficient of variation)
- Tail metrics (p95 for liquidity shocks)

NOT responsible for:
- Trade direction/OFI (see order_flow.py)
- Volume patterns (see volume.py)
- Realized spreads (Phase 2 - requires trade-quote merge)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_daily_liquidity(
    quotes_df: pd.DataFrame,
    trades_df: pd.DataFrame = None
) -> dict:
    """
    Compute daily liquidity metrics from TAQ quotes.
    
    Args:
        quotes_df: From taq.load_quotes() with timestamp, bid, ask, midpoint, spread, permno, date
        trades_df: Optional, not used in Phase 1 (reserved for Phase 2 realized spreads)
    
    Returns:
        dict with keys (frozen contract for build_daily_features.py):
            - permno: int
            - date: datetime
            - spread_mean: float, average relative spread
            - spread_std: float, spread volatility
            - spread_stability: float, CV (lower = more stable)
            - quoted_spread_mean: float, average absolute spread ($)
            - spread_p95: float, 95th percentile (tail liquidity shock)
            - num_quotes: int, original quote count
            - num_quotes_1s: int, after 1-second resampling
            
    Note:
        - Quotes are resampled to 1-second buckets for performance and noise reduction
        - Empty quotes_df returns NaN/0 with warning
        - Phase 1 ignores trades_df; Phase 2 will add realized spreads
    """
    
    if len(quotes_df) == 0:
        logger.warning("Empty quotes_df in compute_daily_liquidity")
        return {
            'permno': np.nan,
            'date': pd.NaT,
            'spread_mean': np.nan,
            'spread_std': np.nan,
            'spread_stability': np.nan,
            'quoted_spread_mean': np.nan,
            'spread_p95': np.nan,
            'num_quotes': 0,
            'num_quotes_1s': 0
        }
    
    # Extract identifiers (set by taq.py)
    permno = int(quotes_df['permno'].iloc[0])
    date = pd.to_datetime(quotes_df['date'].iloc[0])
    num_quotes_raw = len(quotes_df)
    
    # Resample to 1-second buckets for performance and stability
    quotes_1s = _resample_quotes_1s(quotes_df)
    
    if len(quotes_1s) == 0:
        logger.warning(f"No valid quotes after resampling for permno {permno} on {date.date()}")
        return {
            'permno': permno,
            'date': date,
            'spread_mean': np.nan,
            'spread_std': np.nan,
            'spread_stability': np.nan,
            'quoted_spread_mean': np.nan,
            'spread_p95': np.nan,
            'num_quotes': num_quotes_raw,
            'num_quotes_1s': 0
        }
    
    num_quotes_1s = len(quotes_1s)
    
    # Calculate spread metrics
    spread_mean = float(quotes_1s['spread'].mean())
    spread_std = float(quotes_1s['spread'].std())
    
    # Spread stability (coefficient of variation)
    if spread_mean > 0:
        spread_stability = spread_std / spread_mean
    else:
        spread_stability = np.nan
    
    # Absolute quoted spread (in dollars)
    quoted_spread_mean = float(quotes_1s['quoted_spread'].mean())
    
    # Tail metric (95th percentile for liquidity shocks)
    spread_p95 = float(quotes_1s['spread'].quantile(0.95))
    
    result = {
        'permno': permno,
        'date': date,
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'spread_stability': spread_stability,
        'quoted_spread_mean': quoted_spread_mean,
        'spread_p95': spread_p95,
        'num_quotes': num_quotes_raw,
        'num_quotes_1s': num_quotes_1s
    }
    
    # Format logging conditionally
    stability_str = f"CV={spread_stability:.3f}" if pd.notna(spread_stability) else "CV=N/A"
    
    logger.info(f"Computed liquidity for permno {permno} on {date.date()}: "
                f"spread={spread_mean:.4%} ({stability_str}), "
                f"p95={spread_p95:.4%}, quotes={num_quotes_raw:,}→{num_quotes_1s:,}")
    
    return result


def _resample_quotes_1s(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal: Resample quotes to 1-second buckets.
    
    Resampling strategy:
        - Use last() to get the prevailing quote at end of each second
        - Recalculate spread and quoted_spread from resampled bid/ask
        - Filter out extreme spreads (>50%) as additional defense
        
    Args:
        quotes_df: Raw quotes with timestamp (tz-aware), bid, ask, midpoint
        
    Returns:
        Resampled quotes_df with same columns
    """
    
    if len(quotes_df) == 0:
        return pd.DataFrame()
    
    df = quotes_df.copy()
    
    # Convert decimal.Decimal columns to float to avoid type mismatch
    df['bid'] = df['bid'].astype(float)
    df['ask'] = df['ask'].astype(float)
    
    # Set timestamp as index for resampling
    df = df.set_index('timestamp')
    
    # Resample to 1-second buckets, taking last quote in each bucket
    # (last quote = prevailing quote at end of second)
    df_1s = df.resample('1s').last()
    
    # Drop NaN rows (seconds with no quotes)
    df_1s = df_1s.dropna(subset=['bid', 'ask'])
    
    if len(df_1s) == 0:
        return pd.DataFrame()
    
    # Recalculate spread metrics from resampled bid/ask
    # (midpoint may have been resampled incorrectly if we just took last())
    df_1s['midpoint'] = (df_1s['bid'] + df_1s['ask']) / 2
    df_1s['spread'] = (df_1s['ask'] - df_1s['bid']) / df_1s['midpoint']
    df_1s['quoted_spread'] = df_1s['ask'] - df_1s['bid']
    
    # Additional defensive filter: remove extreme spreads (>50%)
    # (taq.py already filters, but double-check after resampling)
    n_before = len(df_1s)
    df_1s = df_1s[df_1s['spread'] < 0.50]
    n_removed = n_before - len(df_1s)
    
    if n_removed > 0:
        logger.debug(f"Removed {n_removed} quotes with spread > 50% after resampling")
    
    # Reset index to get timestamp back as column
    df_1s = df_1s.reset_index()
    
    return df_1s


def compute_spread_stability_intraday(
    quotes_df: pd.DataFrame,
    window_minutes: int = 30
) -> dict:
    """
    Optional Phase 2: Compute intraday spread stability over rolling windows.
    
    Useful for detecting regime shifts (calm → stressed liquidity).
    
    Args:
        quotes_df: Raw quotes
        window_minutes: Rolling window size
        
    Returns:
        dict with:
            - spread_autocorr: float, lag-1 autocorrelation of spread
            - spread_regime_changes: int, number of volatility regime breaks
    """
    
    if len(quotes_df) == 0:
        return {
            'spread_autocorr': np.nan,
            'spread_regime_changes': 0
        }
    
    # Resample to reduce noise
    df = _resample_quotes_1s(quotes_df)
    
    if len(df) < 10:
        return {
            'spread_autocorr': np.nan,
            'spread_regime_changes': 0
        }
    
    # Lag-1 autocorrelation (persistence measure)
    spread_autocorr = float(df['spread'].autocorr(lag=1))
    
    # Regime changes: rolling std breaks threshold
    # (Simple heuristic: count times rolling std > 2x median)
    df = df.set_index('timestamp')
    rolling_std = df['spread'].rolling(f'{window_minutes}min').std()
    median_std = rolling_std.median()
    
    if pd.notna(median_std) and median_std > 0:
        regime_breaks = (rolling_std > 2 * median_std).sum()
    else:
        regime_breaks = 0
    
    result = {
        'spread_autocorr': spread_autocorr,
        'spread_regime_changes': int(regime_breaks)
    }
    
    logger.debug(f"Intraday spread: autocorr={spread_autocorr:.3f}, "
                f"regime_changes={regime_breaks}")
    
    return result


if __name__ == "__main__":
    # Test with taq.py integration
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    import wrds
    from src.data_loaders.taq import load_quotes
    
    db = wrds.Connection()
    
    try:
        # AAPL = permno 14593
        print("Loading quotes for AAPL...")
        quotes = load_quotes(14593, "2023-05-03", db)
        
        print("\n" + "="*60)
        print("Test 1: Liquidity metrics (Phase 1)")
        print("="*60)
        
        result = compute_daily_liquidity(quotes)
        
        print("\nLiquidity Features:")
        for k, v in result.items():
            if isinstance(v, float):
                if pd.notna(v):
                    if 'spread' in k and k != 'quoted_spread_mean':
                        print(f"  {k:25s}: {v:.4%}")
                    elif k == 'quoted_spread_mean':
                        print(f"  {k:25s}: ${v:.4f}")
                    else:
                        print(f"  {k:25s}: {v:.4f}")
                else:
                    print(f"  {k:25s}: NaN")
            elif isinstance(v, int):
                print(f"  {k:25s}: {v:,}")
            else:
                print(f"  {k:25s}: {v}")
        
        print("\n" + "="*60)
        print("Test 2: Intraday spread stability (Phase 2 feature)")
        print("="*60)
        
        intraday_result = compute_spread_stability_intraday(quotes, window_minutes=30)
        
        print("\nIntraday Spread Metrics:")
        print(f"  Spread autocorrelation:  {intraday_result['spread_autocorr']:.3f}")
        print(f"  Regime changes (30min):  {intraday_result['spread_regime_changes']}")
        
        print("\n" + "="*60)
        print("Interpretation")
        print("="*60)
        
        spread_mean = result['spread_mean']
        spread_stability = result['spread_stability']
        spread_p95 = result['spread_p95']
        
        print(f"AAPL on 2023-05-03:")
        print(f"  - Average spread: {spread_mean:.4%}")
        print(f"  - Spread stability (CV): {spread_stability:.3f}")
        
        if spread_stability < 0.5:
            print(f"    → Highly stable spreads (low friction volatility)")
        elif spread_stability < 1.0:
            print(f"    → Moderate spread stability")
        else:
            print(f"    → Volatile spreads (liquidity stress)")
        
        print(f"  - 95th percentile spread: {spread_p95:.4%}")
        
        if spread_p95 > 3 * spread_mean:
            print(f"    → Significant tail risk (liquidity shocks present)")
        else:
            print(f"    → Tail risk contained")
        
        print(f"  - Quote reduction: {result['num_quotes']:,} → {result['num_quotes_1s']:,}")
        reduction_pct = (1 - result['num_quotes_1s'] / result['num_quotes']) * 100
        print(f"    → {reduction_pct:.1f}% compression via 1s resampling")
        
    finally:
        db.close()
        print("\n✓ Test complete")
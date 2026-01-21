"""
Volume intensity, abnormality, and concentration metrics

Responsibility: Trading intensity and intraday patterns only
- Dollar/share volume calculation
- Abnormal volume (relative to ADV benchmark)
- Intraday concentration (morning vs full day)

NOT responsible for:
- Trade direction (see order_flow.py)
- Spread/liquidity measures (see liquidity.py)
- ADV calculation (external input)
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_daily_volume(
    trades_df: pd.DataFrame,
    adv_usd: float = None
) -> dict:
    """
    Compute daily volume intensity and pattern metrics.
    
    Args:
        trades_df: From taq.load_trades() with timestamp, price, size, permno, date
        adv_usd: Optional, Average Daily Dollar Volume benchmark
                 (60-day trailing mean, computed externally)
    
    Returns:
        dict with keys (frozen contract for build_daily_features.py):
            - permno: int
            - date: datetime
            - dollar_volume: float, Σ(price × size)
            - share_volume: int, Σ(size)
            - num_trades: int
            - avg_trade_size: float
            - abnormal_vol_ratio: float or NaN, dollar_volume / adv_usd
            - morning_share: float [0-1], fraction of volume before noon
            
    Note:
        - If trades_df empty: returns NaN/0 with warning
        - If adv_usd not provided: abnormal_vol_ratio = NaN
        - Morning cutoff: 12:00 ET (arbitrary but standard)
    """
    
    if len(trades_df) == 0:
        logger.warning("Empty trades_df in compute_daily_volume")
        return {
            'permno': np.nan,
            'date': pd.NaT,
            'dollar_volume': 0.0,
            'share_volume': 0,
            'num_trades': 0,
            'avg_trade_size': np.nan,
            'abnormal_vol_ratio': np.nan,
            'morning_share': np.nan
        }
    
    # Extract identifiers (set by taq.py)
    permno = int(trades_df['permno'].iloc[0])
    date = pd.to_datetime(trades_df['date'].iloc[0])
    
    df = trades_df.copy()
    
    # Basic volume metrics
    df['dollar_value'] = df['price'] * df['size']
    
    dollar_volume = float(df['dollar_value'].sum())
    share_volume = int(df['size'].sum())
    num_trades = len(df)
    avg_trade_size = float(df['size'].mean())
    
    # Abnormal volume (relative to benchmark)
    if adv_usd is not None and adv_usd > 0:
        abnormal_vol_ratio = dollar_volume / adv_usd
    else:
        abnormal_vol_ratio = np.nan
        if adv_usd is None:
            logger.debug(f"No ADV provided for permno {permno} on {date.date()}")
    
    # Intraday pattern: morning concentration
    morning_share = _compute_morning_share(df)
    
    result = {
        'permno': permno,
        'date': date,
        'dollar_volume': dollar_volume,
        'share_volume': share_volume,
        'num_trades': num_trades,
        'avg_trade_size': avg_trade_size,
        'abnormal_vol_ratio': abnormal_vol_ratio,
        'morning_share': morning_share
    }
    
    # Format logging message conditionally
    adv_str = f"{abnormal_vol_ratio:.2f}x ADV" if pd.notna(abnormal_vol_ratio) else "no ADV"
    morning_str = f"{morning_share:.1%}" if pd.notna(morning_share) else "N/A"
    
    logger.info(f"Computed volume metrics for permno {permno} on {date.date()}: "
                f"${dollar_volume:,.0f} ({adv_str}), "
                f"{morning_str} morning")
    
    return result


def _compute_morning_share(trades_df: pd.DataFrame) -> float:
    """
    Internal: Compute fraction of dollar volume occurring before noon.
    
    Morning cutoff: 12:00 ET (arbitrary but fixed for consistency)
    Note: Literature varies between 10:30-12:00; we use 12:00 as a reasonable
    midpoint that captures pre-lunch trading patterns.
    
    Returns:
        float [0-1], morning_dollar_volume / total_dollar_volume
        NaN if total volume is zero
    """
    
    if len(trades_df) == 0:
        return np.nan
    
    df = trades_df.copy()
    
    # Define morning: before 12:00 ET
    # timestamp is already timezone-aware from taq.py
    df['is_morning'] = df['timestamp'].dt.time < pd.Timestamp('12:00:00').time()
    
    # Calculate dollar volumes
    df['dollar_value'] = df['price'] * df['size']
    
    morning_dollar_vol = df[df['is_morning']]['dollar_value'].sum()
    total_dollar_vol = df['dollar_value'].sum()
    
    if total_dollar_vol > 0:
        return float(morning_dollar_vol / total_dollar_vol)
    else:
        return np.nan


def compute_volume_burst(
    trades_df: pd.DataFrame,
    window_minutes: int = 5
) -> dict:
    """
    Optional: Detect volume bursts (spikes significantly above average).
    
    Phase 2 feature - useful for liquidity pressure classification.
    
    Args:
        trades_df: Intraday trades
        window_minutes: Rolling window for burst detection
        
    Returns:
        dict with:
            - max_burst_ratio: float, max(window_vol / avg_window_vol)
            - burst_count: int, number of windows > 2x average
            - burst_timestamps: list of burst window start times
    """
    
    if len(trades_df) == 0:
        return {
            'max_burst_ratio': np.nan,
            'burst_count': 0,
            'burst_timestamps': []
        }
    
    df = trades_df.copy()
    df['dollar_value'] = df['price'] * df['size']
    
    # Resample to window_minutes buckets
    df = df.set_index('timestamp')
    window_vol = df['dollar_value'].resample(f'{window_minutes}min').sum()
    
    # Calculate average and burst threshold
    avg_vol = window_vol.mean()
    burst_threshold = 2.0 * avg_vol  # Heuristic threshold (can be parameterized in Phase 2)
    
    # Detect bursts
    bursts = window_vol[window_vol > burst_threshold]
    
    if avg_vol > 0:
        max_burst_ratio = float(window_vol.max() / avg_vol)
    else:
        max_burst_ratio = np.nan
    
    result = {
        'max_burst_ratio': max_burst_ratio,
        'burst_count': len(bursts),
        'burst_timestamps': bursts.index.tolist() if len(bursts) > 0 else []
    }
    
    logger.debug(f"Detected {len(bursts)} volume bursts (>{burst_threshold:,.0f})")
    
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
    from src.data_loaders.taq import load_trades
    
    db = wrds.Connection()
    
    try:
        # AAPL = permno 14593
        print("Loading trades for AAPL...")
        trades = load_trades(14593, "2023-05-03", db)
        
        print("\n" + "="*60)
        print("Test 1: Volume metrics without ADV benchmark")
        print("="*60)
        
        result_no_adv = compute_daily_volume(trades, adv_usd=None)
        
        print("\nVolume Features (no ADV):")
        for k, v in result_no_adv.items():
            if isinstance(v, float):
                if pd.notna(v):
                    print(f"  {k:25s}: {v:,.2f}")
                else:
                    print(f"  {k:25s}: NaN")
            elif isinstance(v, int):
                print(f"  {k:25s}: {v:,}")
            else:
                print(f"  {k:25s}: {v}")
        
        print("\n" + "="*60)
        print("Test 2: Volume metrics with mock ADV")
        print("="*60)
        
        # Mock ADV: assume AAPL trades ~$15B/day on average
        mock_adv = 15_000_000_000
        
        result_with_adv = compute_daily_volume(trades, adv_usd=mock_adv)
        
        print(f"\nVolume Features (ADV = ${mock_adv:,.0f}):")
        for k, v in result_with_adv.items():
            if isinstance(v, float):
                if pd.notna(v):
                    print(f"  {k:25s}: {v:,.2f}")
                else:
                    print(f"  {k:25s}: NaN")
            elif isinstance(v, int):
                print(f"  {k:25s}: {v:,}")
            else:
                print(f"  {k:25s}: {v}")
        
        print("\n" + "="*60)
        print("Test 3: Volume burst detection (Phase 2 feature)")
        print("="*60)
        
        burst_result = compute_volume_burst(trades, window_minutes=5)
        
        print("\nVolume Burst Metrics:")
        print(f"  Max burst ratio:     {burst_result['max_burst_ratio']:.2f}x")
        print(f"  Burst count:         {burst_result['burst_count']}")
        if len(burst_result['burst_timestamps']) > 0:
            print(f"  First burst at:      {burst_result['burst_timestamps'][0]}")
        
        print("\n" + "="*60)
        print("Interpretation")
        print("="*60)
        morning = result_with_adv['morning_share']
        abnormal = result_with_adv['abnormal_vol_ratio']
        
        print(f"AAPL on 2023-05-03:")
        print(f"  - {morning:.1%} of volume occurred before noon")
        if morning > 0.6:
            print(f"    → Morning-heavy trading (potential information flow)")
        elif morning < 0.4:
            print(f"    → Afternoon-heavy trading (potential closing imbalance)")
        else:
            print(f"    → Balanced intraday distribution")
        
        print(f"  - {abnormal:.2f}x normal daily volume")
        if abnormal > 1.5:
            print(f"    → Abnormally high volume (potential liquidity event)")
        elif abnormal < 0.7:
            print(f"    → Below-average volume")
        else:
            print(f"    → Normal volume range")
        
    finally:
        db.close()
        print("\n✓ Test complete")
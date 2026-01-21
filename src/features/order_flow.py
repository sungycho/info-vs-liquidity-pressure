"""
Order Flow Imbalance calculation using Lee-Ready classification

Responsibility: Trade direction classification and order flow imbalance only
- Lee-Ready algorithm (quote rule + tick test fallback)
- Order Flow Imbalance (OFI) calculation
- OFI persistence/stability metrics
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def classify_trades(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Classify trades as buyer-initiated or seller-initiated.
    
    Uses Lee-Ready algorithm if quotes available, otherwise falls back to tick test.
    
    Lee-Ready algorithm:
        1. Quote Rule: Compare trade price to prevailing midpoint
        2. Tick Test: Use price change as fallback for at-midpoint trades
        
    Args:
        trades_df: From taq.load_trades() with timestamp, price, size
        quotes_df: Optional, from taq.load_quotes() with timestamp, midpoint
        
    Returns:
        trades_df with 'direction' column: 1 (buy), -1 (sell), 0 (unknown)
        Also adds 'classification_method' column for diagnostics
    """
    
    if len(trades_df) == 0:
        logger.warning("Empty trades_df, cannot classify")
        df = trades_df.copy()
        df['direction'] = 0
        df['classification_method'] = 'none'
        return df
    
    df = trades_df.copy()
    
    # Use Lee-Ready if quotes available
    if quotes_df is not None and len(quotes_df) > 0:
        df = _classify_lee_ready(df, quotes_df)
        df['classification_method'] = 'lee_ready'
    else:
        logger.info("No quotes provided, using tick test fallback")
        df = _classify_tick_test(df)
        df['classification_method'] = 'tick_test'
    
    n_buys = (df['direction'] == 1).sum()
    n_sells = (df['direction'] == -1).sum()
    n_unknown = (df['direction'] == 0).sum()
    
    logger.debug(f"Classified {len(df)} trades: {n_buys} buys, {n_sells} sells, {n_unknown} unknown")
    
    return df


def _classify_lee_ready(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Internal: Lee-Ready classification implementation.
    
    Uses quote rule with 5-second backward lookup, then tick test fallback.
    """
    
    df = trades_df.copy()
    
    # Step 1: Merge each trade with prevailing quote (5s before trade)
    # Use merge_asof for efficient time-based join
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        quotes_df[['timestamp', 'midpoint']].sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        tolerance=pd.Timedelta('5s')
    )
    
    # Step 2: Quote Rule
    df['direction'] = 0
    df.loc[df['price'] > df['midpoint'], 'direction'] = 1   # Above mid = buy
    df.loc[df['price'] < df['midpoint'], 'direction'] = -1  # Below mid = sell
    
    # Step 3: Tick Test (fallback for at-midpoint trades)
    at_mid = df['direction'] == 0
    
    if at_mid.sum() > 0:
        df['price_change'] = df['price'].diff()
        df.loc[at_mid & (df['price_change'] > 0), 'direction'] = 1
        df.loc[at_mid & (df['price_change'] < 0), 'direction'] = -1
        
        # Forward fill remaining zero ticks
        still_unknown = df['direction'] == 0
        if still_unknown.sum() > 0:
            df.loc[still_unknown, 'direction'] = np.nan
            df['direction'] = df['direction'].ffill().fillna(0).astype(int)
    
    return df


def _classify_tick_test(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal: Fallback tick test classification.
    
    Less accurate (~75%) but works when quotes unavailable.
    """
    
    df = trades_df.copy()
    
    # Price change from previous trade
    df['price_change'] = df['price'].diff()
    
    # Initialize direction
    df['direction'] = 0
    df.loc[df['price_change'] > 0, 'direction'] = 1
    df.loc[df['price_change'] < 0, 'direction'] = -1
    
    # Forward fill zero ticks
    df.loc[df['direction'] == 0, 'direction'] = np.nan
    df['direction'] = df['direction'].ffill().fillna(0).astype(int)
    
    return df


def compute_daily_order_flow(
    trades_df: pd.DataFrame,
    quotes_df: pd.DataFrame = None
) -> dict:
    """
    Compute daily order flow imbalance and related metrics.
    
    This is the main entry point for order flow feature calculation.
    
    Args:
        trades_df: From taq.load_trades() - must have permno, date columns
        quotes_df: Optional, from taq.load_quotes() - enables Lee-Ready
    
    Returns:
        dict with keys:
            - permno: int
            - date: datetime
            - ofi: float, Order Flow Imbalance in [-1, 1]
            - ofi_abs: float, Absolute OFI
            - ofi_std: float, Intraday OFI dispersion (computed over 30-min buckets)
            - buy_volume: int
            - sell_volume: int
            - volume: int, total volume
            - num_trades: int
            - avg_trade_size: float
            - classification_method: str, 'lee_ready' or 'tick_test'
            
    Contract:
        This output format is frozen for build_daily_features.py integration.
        Any additions must maintain backward compatibility.
    """
    
    if len(trades_df) == 0:
        logger.warning("Empty trades_df in compute_daily_order_flow")
        return {
            'permno': np.nan,
            'date': pd.NaT,
            'ofi': np.nan,
            'ofi_abs': np.nan,
            'ofi_std': np.nan,
            'buy_volume': 0,
            'sell_volume': 0,
            'volume': 0,
            'num_trades': 0,
            'avg_trade_size': np.nan,
            'classification_method': 'none'
        }
    
    # Extract identifiers from trades_df (set by taq.py)
    permno = int(trades_df['permno'].iloc[0])
    date = pd.to_datetime(trades_df['date'].iloc[0])
    
    # Classify trades
    df = classify_trades(trades_df, quotes_df)
    method = df['classification_method'].iloc[0]
    
    # Calculate volumes
    buy_volume = int(df[df['direction'] == 1]['size'].sum())
    sell_volume = int(df[df['direction'] == -1]['size'].sum())
    total_volume = int(df['size'].sum())
    
    # Order Flow Imbalance
    ofi = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0
    ofi_abs = abs(ofi)
    
    # OFI stability: compute intraday dispersion using 30-min buckets
    ofi_std = _compute_ofi_stability(df)
    
    result = {
        'permno': permno,
        'date': date,
        'ofi': ofi,
        'ofi_abs': ofi_abs,
        'ofi_std': ofi_std,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'volume': total_volume,
        'num_trades': len(df),
        'avg_trade_size': float(df['size'].mean()),
        'classification_method': method
    }
    
    logger.info(f"Computed OFI={ofi:.4f} (std={ofi_std:.4f}) for permno {permno} "
                f"on {date.date()} using {method}")
    
    return result


def _compute_ofi_stability(trades_df: pd.DataFrame) -> float:
    """
    Internal: Compute intraday OFI dispersion as stability metric.
    
    Divides trading day into 30-minute buckets, computes OFI for each,
    returns standard deviation across buckets.
    
    Higher std = more volatile order flow (less persistent)
    Lower std = more stable order flow (more persistent)
    """
    
    if len(trades_df) == 0:
        return np.nan
    
    df = trades_df.copy()
    
    # Create 30-min time buckets
    df['bucket'] = df['timestamp'].dt.floor('30min')
    
    # Compute OFI per bucket
    bucket_ofis = []
    for bucket, group in df.groupby('bucket'):
        buy_vol = group[group['direction'] == 1]['size'].sum()
        sell_vol = group[group['direction'] == -1]['size'].sum()
        total_vol = group['size'].sum()
        
        if total_vol > 0:
            bucket_ofi = (buy_vol - sell_vol) / total_vol
            bucket_ofis.append(bucket_ofi)
    
    # Return std across buckets
    if len(bucket_ofis) > 1:
        return float(np.std(bucket_ofis))
    else:
        return 0.0


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
    from src.data_loaders.taq import load_trades, load_quotes
    
    db = wrds.Connection()
    
    try:
        # AAPL = permno 14593
        print("Loading trades and quotes for AAPL...")
        trades = load_trades(14593, "2023-05-03", db)
        quotes = load_quotes(14593, "2023-05-03", db)
        
        print("\n" + "="*60)
        print("Test 1: Lee-Ready classification (with quotes)")
        print("="*60)
        result_lr = compute_daily_order_flow(trades, quotes)
        
        print("\nOrder Flow Features (Lee-Ready):")
        for k, v in result_lr.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.6f}")
            else:
                print(f"  {k:25s}: {v}")
        
        print("\n" + "="*60)
        print("Test 2: Tick test fallback (no quotes)")
        print("="*60)
        result_tt = compute_daily_order_flow(trades, None)
        
        print("\nOrder Flow Features (Tick Test):")
        for k, v in result_tt.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.6f}")
            else:
                print(f"  {k:25s}: {v}")
        
        print("\n" + "="*60)
        print("Comparison")
        print("="*60)
        print(f"OFI difference: {abs(result_lr['ofi'] - result_tt['ofi']):.6f}")
        print(f"  Lee-Ready OFI: {result_lr['ofi']:.6f}")
        print(f"  Tick Test OFI: {result_tt['ofi']:.6f}")
        
    finally:
        db.close()
        print("\n✓ Test complete")
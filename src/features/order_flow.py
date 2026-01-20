"""
Order Flow Imbalance (OFI) calculation using Lee-Ready classification
"""
import pandas as pd
import numpy as np


def classify_trades_tick_test(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple tick test: compare price to previous price.
    
    Returns:
        trades_df with 'direction' column: 1 (buy), -1 (sell), 0 (unknown)
    """
    
    df = trades_df.copy()
    
    # Price change
    df['price_change'] = df['price'].diff()
    
    # Initialize direction
    df['direction'] = 0
    
    # Classify (handle NA explicitly)
    df.loc[df['price_change'] > 0, 'direction'] = 1
    df.loc[df['price_change'] < 0, 'direction'] = -1
    
    # Forward fill zero ticks
    df.loc[df['direction'] == 0, 'direction'] = np.nan
    df['direction'] = df['direction'].ffill().fillna(0).astype(int)
    
    return df


def calculate_daily_ofi(
    trades_df: pd.DataFrame,
    permno: int,
    date: str
) -> pd.Series:
    """
    Calculate daily Order Flow Imbalance and summary stats.
    
    Args:
        trades_df: Must have 'price', 'size', 'direction' columns
        permno: CRSP PERMNO
        date: Trading date
    
    Returns:
        Series with: permno, date, ofi, volume, num_trades, avg_trade_size
    """
    
    if len(trades_df) == 0:
        return pd.Series({
            'permno': permno,
            'date': pd.to_datetime(date),
            'ofi': np.nan,
            'volume': 0,
            'num_trades': 0,
            'avg_trade_size': np.nan
        })
    
    df = trades_df.copy()
    
    # Classify if not already done
    if 'direction' not in df.columns:
        df = classify_trades_tick_test(df)
    
    # Calculate volumes
    buy_volume = (df[df['direction'] == 1]['size']).sum()
    sell_volume = (df[df['direction'] == -1]['size']).sum()
    total_volume = df['size'].sum()
    
    # OFI
    if total_volume > 0:
        ofi = (buy_volume - sell_volume) / total_volume
    else:
        ofi = 0.0
    
    return pd.Series({
        'permno': permno,
        'date': pd.to_datetime(date),
        'ofi': ofi,
        'volume': total_volume,
        'num_trades': len(df),
        'avg_trade_size': df['size'].mean(),
        'buy_volume': buy_volume,
        'sell_volume': sell_volume
    })


def calculate_daily_spread(quotes_df: pd.DataFrame) -> dict:
    """
    Calculate daily average spread metrics.
    
    Returns:
        dict with: avg_spread, avg_quoted_spread, time_weighted_spread
    """
    
    if len(quotes_df) == 0:
        return {
            'avg_spread': np.nan,
            'avg_quoted_spread': np.nan
        }
    
    return {
        'avg_spread': quotes_df['spread'].mean(),
        'avg_quoted_spread': (quotes_df['ask'] - quotes_df['bid']).mean()
    }


if __name__ == "__main__":
    # Test with mock data
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.data_loaders.taq import load_trades, load_quotes
    
    trades = load_trades("AAPL", "2023-05-03")
    
    if len(trades) > 0:
        daily_features = calculate_daily_ofi(
            trades_df=trades,
            permno=14593,  # AAPL
            date="2023-05-03"
        )
        
        print("\nDaily features:")
        print(daily_features)
# test_taq.py
import sys
sys.path.append('src')

from src.data_loaders.taq import load_trades
import wrds
import logging

logging.basicConfig(level=logging.INFO)

db = wrds.Connection()

try:
    # AAPL = permno 14593, 작은 날짜로 테스트
    trades = load_trades(14593, "2023-05-03", db)
    
    print(f"\n✓ Success!")
    print(f"  Loaded {len(trades):,} trades")
    print(f"  Columns: {list(trades.columns)}")
    print(f"\nFirst 3 rows:")
    print(trades.head(3))
    
finally:
    db.close()
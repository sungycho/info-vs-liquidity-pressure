"""
Quick test: Verify CRSP dsf only contains NYSE trading days.

This validates that rolling(20) on CRSP data = 20 trading days, not calendar days.
"""

import pandas as pd
import pandas_market_calendars as mcal
import wrds
from sqlalchemy import text

# Query CRSP for sample period
db = wrds.Connection()

query = text("""
SELECT DISTINCT date
FROM crsp.dsf
WHERE date BETWEEN '2023-01-01' AND '2023-12-31'
ORDER BY date
""")

result = db.connection.execute(query)
crsp_dates = pd.to_datetime([row[0] for row in result.fetchall()])
db.close()

print(f"CRSP dates in 2023: {len(crsp_dates)}")

# Get NYSE trading days for 2023
nyse = mcal.get_calendar('NYSE')
schedule = nyse.schedule(start_date='2023-01-01', end_date='2023-12-31')
trading_days = pd.to_datetime(schedule.index)

print(f"NYSE trading days in 2023: {len(trading_days)}")

# Check if CRSP ⊆ NYSE trading days
crsp_set = set(crsp_dates.date)
trading_set = set(trading_days.date)

non_trading = crsp_set - trading_set
missing_trading = trading_set - crsp_set

print(f"\nValidation:")
print(f"  CRSP dates NOT in NYSE calendar: {len(non_trading)}")
print(f"  NYSE days NOT in CRSP: {len(missing_trading)}")

if len(non_trading) == 0:
    print("\n✅ PASS: All CRSP dates are NYSE trading days")
    print("   → rolling(20) on CRSP = 20 trading days (no calendar needed)")
else:
    print(f"\n❌ FAIL: Found non-trading dates in CRSP: {list(non_trading)[:5]}")
    print("   → Need explicit trading day reindexing")

# Bonus: Check for gaps (e.g., missing trading days in CRSP)
if len(missing_trading) > 0:
    print(f"\n⚠️  Note: {len(missing_trading)} trading days missing from CRSP")
    print(f"   Examples: {sorted(missing_trading)[:5]}")
    print("   (Normal for early data or data gaps)")
# test_earnings_loader.py
from src.data_loaders.earnings import load_earnings_calendar

# Quick test: 2024 Jan-Feb only
df = load_earnings_calendar(
    start_year=2024,
    end_year=2024,
    min_volume_usd=5_000_000,
    save_path="data/processed/event_table_sample.parquet"
)

print("\n=== Test Passed ===")
print(f"Total events: {len(df)}")
print(f"\nFirst event for TAQ testing:")
print(df.iloc[0])

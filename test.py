"""
Diagnose why 1,520 events (36%) are being dropped.
"""

import pandas as pd
import numpy as np

# Load data
daily_df = pd.read_parquet("data/processed/daily_features_full_v2.parquet")
events_df = pd.read_parquet("data/processed/event_table_2023_2024.parquet")

print("="*60)
print("Event Loss Diagnostic")
print("="*60)

# Add event_day
events_meta = events_df[['event_id', 'permno', 'rdq']].copy()
events_meta = events_meta.rename(columns={'rdq': 'announce_date'})
daily_df = daily_df.merge(events_meta, on=['event_id', 'permno'], how='left')
daily_df['announce_date'] = pd.to_datetime(daily_df['announce_date'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df['event_day'] = (daily_df['date'] - daily_df['announce_date']).dt.days

print(f"\n1. Event day distribution (all data):")
print(daily_df['event_day'].value_counts().sort_index())

print(f"\n2. Events by day count (before filtering):")
event_day_counts = daily_df.groupby('event_id')['event_day'].apply(lambda x: len(x.unique()))
print(event_day_counts.value_counts().sort_index())

# Filter [-10, -1]
pre_window = daily_df[(daily_df['event_day'] >= -10) & (daily_df['event_day'] <= -1)]

print(f"\n3. After [-10, -1] filter:")
print(f"   Events before: {daily_df['event_id'].nunique()}")
print(f"   Events after: {pre_window['event_id'].nunique()}")
print(f"   Lost: {daily_df['event_id'].nunique() - pre_window['event_id'].nunique()}")

print(f"\n4. Days per event in [-10, -1] window:")
event_counts = pre_window.groupby('event_id').size()
print(event_counts.value_counts().sort_index())

print(f"\n5. Events with < 7 days:")
insufficient = event_counts[event_counts < 7]
print(f"   Count: {len(insufficient)}")
print(f"   Percentage: {len(insufficient) / len(event_counts) * 100:.1f}%")

print(f"\n6. Sample events with < 7 days:")
sample_insufficient = insufficient.head(5)
for event_id in sample_insufficient.index:
    event_data = pre_window[pre_window['event_id'] == event_id]
    print(f"\n   {event_id}:")
    print(f"     Permno: {event_data['permno'].iloc[0]}")
    print(f"     Announce: {event_data['announce_date'].iloc[0].date()}")
    print(f"     Days in window: {len(event_data)}")
    print(f"     Event days: {sorted(event_data['event_day'].unique())}")
    print(f"     Dates: {sorted(event_data['date'].dt.date.unique())}")

print(f"\n7. Hypothesis check - are these holiday-heavy periods?")
insufficient_events = pre_window[pre_window['event_id'].isin(insufficient.index)]
date_dist = insufficient_events['date'].dt.to_period('M').value_counts().sort_index()
print("   Month distribution of insufficient events:")
print(date_dist)

print("\n" + "="*60)
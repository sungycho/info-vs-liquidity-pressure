import pandas as pd

df = pd.read_parquet("data/processed/daily_features_full.parquet")

event_table = pd.read_parquet(
    "data/processed/event_table_2023_2024.parquet"
)


print(event_table.loc[
    event_table["event_id"] == "E0022",
    ["event_id", "permno", "rdq"]
])


"""
days_per_event = df.groupby("event_id").size()
eid = days_per_event[days_per_event == 6].index[0]

sub = (
    df[df["event_id"] == "E0022"]
    .merge(
        event_table[["event_id", "rdq"]],
        on="event_id",
        how="left"
    )
    .sort_values("date")
)

print(sub[["date", "rdq"]])


sub[["date", "rdq"]]

six_day_events = days_per_event[days_per_event == 6]

print("Number of 6-day events:", len(six_day_events))
print(six_day_events.head())
"""

import pandas as pd

df = pd.read_parquet("data/processed/daily_features_full.parquet")

key = ["event_id", "permno", "date"]
dup_mask = df.duplicated(subset=key, keep=False)
dup_df = df[dup_mask].copy()

event_conc = (
    dup_df
    .groupby("event_id")
    .size()
    .sort_values(ascending=False)
)

print(event_conc.head(15))

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('SC/B_event_features.parquet')

print(df.columns)

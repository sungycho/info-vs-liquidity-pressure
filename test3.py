import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('data/processed/daily_features.parquet')

print(df.columns)

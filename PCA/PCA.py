import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted successfully. You can now access your files via '/content/drive/MyDrive/'")

"""After executing the cell above, you will see a prompt to authorize Google Drive access. Follow the instructions to complete the mounting process.

Once your Drive is mounted, provide the full path to your `.parquet` file in the next code cell. Remember that files in your Google Drive will typically be located under `/content/drive/MyDrive/` followed by your file's specific path within Drive.
"""

gdrive_parquet_path = '/content/drive/MyDrive/Quant Project/Data/event_features.parquet'

try:
    df_gdrive = pd.read_parquet(gdrive_parquet_path)
    print(f"\nSuccessfully loaded Parquet file from Google Drive: {gdrive_parquet_path}")
    print("Displaying the first 5 rows:")
    display(df_gdrive.head())
    print("\nDataFrame Info:")
    df_gdrive.info()
except FileNotFoundError:
    print(f"Error: The file '{gdrive_parquet_path}' was not found.\n" \
          "Please ensure the path is correct and your Google Drive is mounted successfully.")
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")
    print("This might be due to an incorrect file format, a corrupted file, or other read errors.")

# -------------------------------
# 1. DIAGNOSTIC CHECKS
# -------------------------------

print("Original summary statistics:")
print(df_gdrive[["volume_mean", "num_trades_mean"]].describe())

# Check cross-sectional dispersion by stock
stock_baseline = df_gdrive.groupby("permno")[["volume_mean", "num_trades_mean"]].mean()

print("\nCross-sectional dispersion across stocks (baseline means):")
print(stock_baseline.describe())

# -------------------------------
# 2. NORMALIZATION (WITHIN-STOCK RELATIVE SCALE)
# -------------------------------

# Compute per-stock baseline means
df_gdrive["volume_stock_mean"] = df_gdrive.groupby("permno")["volume_mean"].transform("mean")
df_gdrive["trades_stock_mean"] = df_gdrive.groupby("permno")["num_trades_mean"].transform("mean")

# Relative (event vs own-stock baseline)
df_gdrive["volume_rel"] = df_gdrive["volume_mean"] / df_gdrive["volume_stock_mean"]
df_gdrive["trades_rel"] = df_gdrive["num_trades_mean"] / df_gdrive["trades_stock_mean"]

# Log transform to stabilize heavy tails
df_gdrive["volume_rel_log"] = np.log(df_gdrive["volume_rel"])
df_gdrive["trades_rel_log"] = np.log(df_gdrive["trades_rel"])

# -------------------------------
# 3. CROSS-SECTIONAL STANDARDIZATION (FOR PCA)
# -------------------------------

df_gdrive["volume_rel_z"] = (
    df_gdrive["volume_rel_log"] - df_gdrive["volume_rel_log"].mean()
) / df_gdrive["volume_rel_log"].std()

df_gdrive["trades_rel_z"] = (
    df_gdrive["trades_rel_log"] - df_gdrive["trades_rel_log"].mean()
) / df_gdrive["trades_rel_log"].std()

# -------------------------------
# 4. POST-NORMALIZATION CHECK
# -------------------------------

print("\nAfter normalization:")
print(df_gdrive[["volume_rel_z", "trades_rel_z"]].describe())

# Optional: check correlation between raw and normalized
print("\nCorrelation check:")
print(df_gdrive[[
    "volume_mean", "volume_rel_z",
    "num_trades_mean", "trades_rel_z"
]].corr())

#Data refinement since distributions are too skewed
df_gdrive["volume_rel_log"] = np.clip(df_gdrive["volume_rel_log"], -5, 5)
df_gdrive["trades_rel_log"] = np.clip(df_gdrive["trades_rel_log"], -5, 5)

# -----------------------------
# 1. Construct PCA feature matrix
# -----------------------------

drop_cols = [
    "event_id",
    "permno",
    "event_date",
    "info_score",
    "liq_score",
    "pressure_score",
    "volume_mean",
    "num_trades_mean",
    "volume_stock_mean",
    "trades_stock_mean",
    "volume_rel",
    "trades_rel",
    "volume_rel_z",
    "trades_rel_z"
]

X = df_gdrive.drop(columns=drop_cols)

# Ensure only numeric columns remain
X = X.select_dtypes(include=[np.number])

print("Feature columns used in PCA:")
print(X.columns.tolist())

# -----------------------------
# 2. Standardize
# -----------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. Run PCA
# -----------------------------

pca = PCA()
components = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

# -----------------------------
# 4. Explained variance table
# -----------------------------

ev_df = pd.DataFrame({
    "PC": [f"PC{i+1}" for i in range(len(explained_variance))],
    "Explained Variance Ratio": explained_variance,
    "Cumulative Variance": np.cumsum(explained_variance)
})

print("\nExplained Variance:")
print(ev_df)

# -----------------------------
# 5. Loadings
# -----------------------------

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(X.columns))],
    index=X.columns
)

print("\nLoadings (first 5 PCs):")
print(loadings.iloc[:, :5])

# -----------------------------
# 6. Sorted loadings per PC (top drivers)
# -----------------------------

for i in range(3):  # First 3 PCs usually most important
    print(f"\nTop contributors to PC{i+1}:")
    print(loadings[f"PC{i+1}"].sort_values(key=abs, ascending=False))

print(df_gdrive.columns)

gdrive_crsp_daily_path = '/content/drive/MyDrive/Quant Project/Data/crsp_daily.parquet'

try:
    df_crsp_daily = pd.read_parquet(gdrive_crsp_daily_path)
    print(f"\nSuccessfully loaded Parquet file from Google Drive: {gdrive_crsp_daily_path}")
    print("Displaying the first 5 rows of df_crsp_daily:")
    display(df_crsp_daily.head())
    print("\ndf_crsp_daily DataFrame Info:")
    df_crsp_daily.info()
except FileNotFoundError:
    print(f"Error: The file '{gdrive_crsp_daily_path}' was not found.\n" \
          "Please ensure the path is correct and your Google Drive is mounted successfully.")
except Exception as e:
    print(f"An error occurred while reading the Parquet file: {e}")
    print("This might be due to an incorrect file format, a corrupted file, or other read errors.")

# ----------------------------------------
# 1. Clean CRSP daily data
# ----------------------------------------

df_crsp_daily["date"] = pd.to_datetime(df_crsp_daily["date"])
df_crsp_daily = df_crsp_daily.sort_values(["permno", "date"])

# Remove missing returns
df_crsp_daily = df_crsp_daily.dropna(subset=["ret"])

# ----------------------------------------
# 2. Prepare event dataframe
# ----------------------------------------

df_gdrive["event_date"] = pd.to_datetime(df_gdrive["event_date"])

# ----------------------------------------
# 3. Merge event dates with CRSP
# ----------------------------------------

# Merge to align event with CRSP date index
df_event_ret = pd.merge(
    df_gdrive[["event_id", "permno", "event_date"]],
    df_crsp_daily[["permno", "date", "ret"]],
    left_on=["permno", "event_date"],
    right_on=["permno", "date"],
    how="left"
)

# ----------------------------------------
# 4. Compute forward 20-day return
# ----------------------------------------

# Create forward return container
forward_returns = []

for idx, row in df_gdrive.iterrows():
    perm = row["permno"]
    event_date = row["event_date"]

    stock_data = df_crsp_daily[df_crsp_daily["permno"] == perm]
    stock_data = stock_data.sort_values("date").reset_index(drop=True)

    # Find first trading day after event
    future_data = stock_data[stock_data["date"] > event_date]

    if len(future_data) >= 20:
        window = future_data.iloc[:20]
        cum_ret = (1 + window["ret"].astype(float)).prod() - 1
    else:
        cum_ret = np.nan

    forward_returns.append(cum_ret)

df_gdrive["ret_20d"] = forward_returns

print("Forward returns added.")
print(df_gdrive["ret_20d"].describe())

print("Number of NA data.")
print(df_gdrive["ret_20d"].isna().sum())

#Regression on return based on PC

# ---------------------------------------
# 1. Recompute PCA (clean version)
# ---------------------------------------

drop_cols = [
    "event_id",
    "permno",
    "event_date",
    "info_score",
    "liq_score",
    "pressure_score",
    "volume_mean",
    "num_trades_mean",
    "ret_20d"
]

X = df_gdrive.drop(columns=drop_cols)
X = X.select_dtypes(include=[float, int])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pcs = pca.fit_transform(X_scaled)

pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
pcs_df = pd.DataFrame(pcs, columns=pc_cols, index=df_gdrive.index)

df_reg = pd.concat([df_gdrive, pcs_df], axis=1)

# ---------------------------------------
# 2. Define factors
# ---------------------------------------

df_reg["Info_factor"] = df_reg["PC5"]
df_reg["Liquidity_factor"] = df_reg["PC2"] + df_reg["PC3"]

# ---------------------------------------
# 3. Run regression
# ---------------------------------------

df_model = df_reg[["ret_20d", "Info_factor", "Liquidity_factor"]].dropna()

Y = df_model["ret_20d"]
X_reg = df_model[["Info_factor", "Liquidity_factor"]]
X_reg = sm.add_constant(X_reg)

model = sm.OLS(Y, X_reg).fit(cov_type="HC3")

print(model.summary())

#Quintile Sort to Check Monotonicity

# Work on regression dataframe with PCs
df_q = df_reg[["ret_20d", "Info_factor"]].dropna().copy()

# Create quintiles
df_q["Info_quintile"] = pd.qcut(df_q["Info_factor"], 5, labels=False) + 1

# Compute average returns per quintile
quintile_returns = df_q.groupby("Info_quintile")["ret_20d"].mean()

print("Mean 20-day return by Info_factor quintile:")
print(quintile_returns)

# Q5 - Q1 spread
q_spread = quintile_returns.loc[5] - quintile_returns.loc[1]
print("\nQ5 - Q1 spread:", q_spread)

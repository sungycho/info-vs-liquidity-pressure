"""
Continuous Cross-Sectional Regressions (Methods A–D)

Pools all events into a single OLS regression:
    forward_return_20d = α + β * pressure_score + ε

Reports three SE variants per method:
    - Firm-clustered (by permno)
    - Date-clustered (by event_date)
    - Two-way clustered (CGM 2011: permno + date − intersection)

Output: data/results/continuous_regression_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "Methods_Sung_Cho"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = RESULTS_DIR / "continuous_regression_summary.csv"
FWD_RETURNS_PATH = RESULTS_DIR / "backtest_results_t0_h20.parquet"


# ---------------------------------------------------------------------------
# Helper: two-way clustered SE (Cameron–Gelbach–Miller 2011)
# ---------------------------------------------------------------------------
def _get_vcov_cluster(fit, groups: pd.Series) -> np.ndarray:
    """Return the variance-covariance matrix clustered on `groups`."""
    result = fit.get_robustcov_results(cov_type="cluster", groups=groups)
    return result.cov_params()


def two_way_clustered_tstat(fit, groups_a: pd.Series, groups_b: pd.Series) -> tuple[float, float]:
    """
    CGM (2011) two-way clustered SE.

    vcov_2way = vcov_A + vcov_B - vcov_AB
    where vcov_AB uses a combined cluster label (str(A) + "_" + str(B)).

    Returns (beta_pressure_score, t_stat) for the pressure_score coefficient.
    """
    vcov_a = _get_vcov_cluster(fit, groups_a)
    vcov_b = _get_vcov_cluster(fit, groups_b)
    groups_ab = groups_a.astype(str) + "_" + groups_b.astype(str)
    vcov_ab = _get_vcov_cluster(fit, groups_ab)

    vcov_2way = vcov_a + vcov_b - vcov_ab

    # coefficient index for pressure_score (index 1 after constant at index 0)
    beta_idx = 1
    beta = float(fit.params.iloc[beta_idx])
    se = np.sqrt(np.diag(vcov_2way)[beta_idx])
    tstat = beta / se
    return float(beta), float(tstat)


# ---------------------------------------------------------------------------
# Step 1 — Load forward returns once
# ---------------------------------------------------------------------------
fwd = pd.read_parquet(FWD_RETURNS_PATH)[["event_id", "cum_return"]].rename(
    columns={"cum_return": "forward_return_20d"}
)
print(f"Loaded forward returns: {len(fwd)} events\n")


# ---------------------------------------------------------------------------
# Step 2–6 — Per-method loop
# ---------------------------------------------------------------------------
METHOD_CONFIG = {
    "A": {"date_col": "event_date"},
    "B": {"date_col": "event_date"},
    "C": {"date_col": "event_date"},
    "D": {"date_col": "rdq"},   # rename rdq → event_date
}

rows = []

for method, cfg in METHOD_CONFIG.items():
    print(f"=== Method {method} ===")

    # Load features
    feat_path = METHODS_DIR / f"{method}_event_features.parquet"
    feat = pd.read_parquet(feat_path)

    # Rename date column to event_date if needed
    date_col = cfg["date_col"]
    if date_col != "event_date":
        feat = feat.rename(columns={date_col: "event_date"})

    # Keep only needed columns
    feat = feat[["event_id", "permno", "event_date", "pressure_score"]]

    # Merge with forward returns
    df = feat.merge(fwd, on="event_id", how="inner")

    # Drop NaN rows
    df = df.dropna(subset=["pressure_score", "forward_return_20d"])

    n = len(df)

    # Standardize pressure_score
    ps_mean = df["pressure_score"].mean()
    ps_std  = df["pressure_score"].std(ddof=1)
    df["pressure_score_std"] = (df["pressure_score"] - ps_mean) / ps_std

    # ------------------------------------------------------------------
    # Step 3 — OLS with raw pressure_score
    # ------------------------------------------------------------------
    X_raw = sm.add_constant(df["pressure_score"])
    y     = df["forward_return_20d"]
    fit_raw = sm.OLS(y, X_raw).fit()

    # index 0 = const, index 1 = pressure_score
    beta_raw = float(fit_raw.params.iloc[1])
    r2       = float(fit_raw.rsquared)

    # ------------------------------------------------------------------
    # Step 5 — OLS with standardized pressure_score
    # ------------------------------------------------------------------
    X_std    = sm.add_constant(df["pressure_score_std"])
    fit_std  = sm.OLS(y, X_std).fit()
    beta_std = float(fit_std.params.iloc[1])

    # ------------------------------------------------------------------
    # Step 4 — Three SE variants (from raw fit)
    # ------------------------------------------------------------------
    groups_permno = df["permno"].astype(str)
    groups_date   = df["event_date"].astype(str)

    se_variants = {}

    # coefficient index: 0 = const, 1 = pressure_score
    BETA_IDX = 1

    # A. Firm-clustered
    res_firm = fit_raw.get_robustcov_results(cov_type="cluster", groups=groups_permno)
    tstat_firm = float(res_firm.tvalues[BETA_IDX])
    se_variants["firm"] = ("Firm-clustered", tstat_firm)

    # B. Date-clustered
    res_date = fit_raw.get_robustcov_results(cov_type="cluster", groups=groups_date)
    tstat_date = float(res_date.tvalues[BETA_IDX])
    se_variants["date"] = ("Date-clustered", tstat_date)

    # C. Two-way clustered (CGM 2011)
    _, tstat_2way = two_way_clustered_tstat(fit_raw, groups_permno, groups_date)
    se_variants["twoway"] = ("Two-way (CGM)", tstat_2way)

    # ------------------------------------------------------------------
    # Print summary for this method
    # ------------------------------------------------------------------
    header = f"  {'SE type':<20} {'beta_raw':>10} {'beta_std':>10} {'t-stat':>10} {'R2':>10} {'N':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for se_key, (se_label, tstat) in se_variants.items():
        print(f"  {se_label:<20} {beta_raw:>10.6f} {beta_std:>10.6f} {tstat:>10.4f} {r2:>10.6f} {n:>6}")
        rows.append(
            {
                "method":    method,
                "se_type":   se_label,
                "beta_raw":  beta_raw,
                "beta_std":  beta_std,
                "tstat":     tstat,
                "r2":        r2,
                "n":         n,
            }
        )
    print()

# ---------------------------------------------------------------------------
# Step 6 — Save results
# ---------------------------------------------------------------------------
summary = pd.DataFrame(rows, columns=["method", "se_type", "beta_raw", "beta_std", "tstat", "r2", "n"])
summary.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to: {OUTPUT_CSV}")
print(f"Total rows: {len(summary)}")

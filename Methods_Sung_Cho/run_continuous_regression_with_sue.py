"""
Continuous Cross-Sectional Regressions with SUE (Methods A–D)

Estimates two OLS specifications per method:

  Spec 1 (baseline + SUE):
      forward_return_20d = α + β1*pressure_std + β2*sue_std + ε

  Spec 2 (interaction):
      forward_return_20d = α + β1*pressure_std + β2*sue_std
                             + β3*(pressure_std × sue_std) + ε

SUE construction — seasonal random walk (Bernard & Thomas 1989):
    delta_q      = EPS_q − EPS_{q−4}
    sue          = delta_q / rolling_std(delta, 8 quarters)
    Source: comp.fundq, column epsfxq (diluted EPS excl. extraordinary items)

SE variants per specification:
    - Firm-clustered   (by permno)
    - Date-clustered   (by event_date)
    - Two-way clustered (CGM 2011)

Output: data/results/continuous_regression_with_sue.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "Methods_Sung_Cho"
PROCESSED   = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

SUE_CACHE        = PROCESSED / "sue.parquet"
FWD_RETURNS_PATH = RESULTS_DIR / "backtest_results_t0_h20.parquet"
OUTPUT_CSV       = RESULTS_DIR / "continuous_regression_with_sue.csv"

METHOD_CONFIG = {
    "A": {"date_col": "event_date"},
    "B": {"date_col": "event_date"},
    "C": {"date_col": "event_date"},
    "D": {"date_col": "rdq"},
}


# ---------------------------------------------------------------------------
# SUE: fetch from WRDS and cache, or load from cache
# ---------------------------------------------------------------------------
def _compute_sue_from_eps(eps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Seasonal random walk SUE from a long EPS panel.

    Input columns: gvkey, datadate, epsfxq  (sorted by gvkey, datadate)
    Output adds: sue
    """
    eps_df = eps_df.sort_values(["gvkey", "datadate"]).copy()

    # Seasonal difference (same quarter last year)
    eps_df["delta"] = eps_df.groupby("gvkey")["epsfxq"].diff(4)

    # Rolling std of last 8 seasonal differences (min 4 observations required)
    eps_df["delta_std"] = (
        eps_df.groupby("gvkey")["delta"]
        .transform(lambda s: s.rolling(8, min_periods=4).std())
    )

    # SUE = delta / rolling_std  (winsorize at ±10 to handle near-zero std)
    eps_df["sue"] = eps_df["delta"] / eps_df["delta_std"]
    eps_df["sue"] = eps_df["sue"].clip(-10, 10)

    return eps_df[["gvkey", "datadate", "sue"]].dropna(subset=["sue"])


def _load_wrds_credentials() -> tuple[str, str]:
    """
    Resolve WRDS credentials in priority order:
      1. WRDS_USERNAME / WRDS_PASSWORD environment variables
      2. .env file in project root (supports both YAML and shell formats)
    Raises RuntimeError if credentials cannot be found.
    """
    import re

    username = os.environ.get("WRDS_USERNAME", "")
    password = os.environ.get("WRDS_PASSWORD", "")

    if not username or not password:
        env_path = ROOT / ".env"
        if env_path.exists():
            text = env_path.read_text()
            for line in text.splitlines():
                # accept both  KEY=value  and  KEY: "value"  formats
                m = re.match(r'WRDS_USERNAME[=:\s]+"?([^"\s]+)"?', line.strip())
                if m:
                    username = m.group(1)
                m = re.match(r'WRDS_PASSWORD[=:\s]+"?([^"\s]+)"?', line.strip())
                if m:
                    password = m.group(1)

    if not username or not password:
        raise RuntimeError(
            "WRDS credentials not found.\n"
            "Set WRDS_USERNAME and WRDS_PASSWORD as environment variables,\n"
            "or add them to the project .env file."
        )
    return username, password


def _wrds_connect():
    """Return a psycopg2 connection to WRDS PostgreSQL."""
    import psycopg2

    username, password = _load_wrds_credentials()
    return psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        dbname="wrds",
        user=username,
        password=password,
        sslmode="require",
        connect_timeout=60,
    )


def fetch_sue(event_table: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with columns [gvkey, datadate, sue].

    Tries the cache first; falls back to WRDS (comp.fundq) via direct
    SQLAlchemy connection (no interactive prompts).
    The event_table must have columns: gvkey, datadate.
    """
    if SUE_CACHE.exists():
        print(f"Loading cached SUE from {SUE_CACHE}")
        return pd.read_parquet(SUE_CACHE)

    print("SUE cache not found — fetching from WRDS (comp.fundq)...")

    conn = _wrds_connect()

    gvkeys    = tuple(event_table["gvkey"].dropna().unique().tolist())
    min_date  = (event_table["datadate"].min() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    max_date  = event_table["datadate"].max().strftime("%Y-%m-%d")
    gvkey_str = "', '".join(gvkeys)

    query = f"""
        SELECT gvkey, datadate, epsfxq
        FROM comp.fundq
        WHERE gvkey IN ('{gvkey_str}')
          AND datadate BETWEEN '{min_date}' AND '{max_date}'
          AND epsfxq IS NOT NULL
        ORDER BY gvkey, datadate
    """

    print(f"  Querying {len(gvkeys)} gvkeys, {min_date} to {max_date} ...")
    raw = pd.read_sql(query, conn)
    conn.close()
    print(f"  Fetched {len(raw):,} firm-quarter rows from comp.fundq")

    raw["datadate"] = pd.to_datetime(raw["datadate"])

    sue_df = _compute_sue_from_eps(raw)
    print(f"  Computed SUE for {len(sue_df):,} firm-quarters")

    sue_df.to_parquet(SUE_CACHE, index=False)
    print(f"  Saved SUE cache to {SUE_CACHE}")

    return sue_df


# ---------------------------------------------------------------------------
# Shared regression helpers (reused from run_continuous_regression.py)
# ---------------------------------------------------------------------------
def _get_vcov_cluster(fit, groups: pd.Series) -> np.ndarray:
    return fit.get_robustcov_results(cov_type="cluster", groups=groups).cov_params()


def _two_way_tstat(fit, groups_a: pd.Series, groups_b: pd.Series, beta_idx: int) -> float:
    """CGM (2011) two-way clustered t-stat for coefficient at beta_idx."""
    vcov_a  = _get_vcov_cluster(fit, groups_a)
    vcov_b  = _get_vcov_cluster(fit, groups_b)
    vcov_ab = _get_vcov_cluster(fit, groups_a.astype(str) + "_" + groups_b.astype(str))
    vcov_2w = vcov_a + vcov_b - vcov_ab
    beta    = float(fit.params.iloc[beta_idx])
    se      = np.sqrt(np.diag(vcov_2w)[beta_idx])
    return beta / se


def _clustered_tstats(fit, groups_permno: pd.Series, groups_date: pd.Series,
                      beta_indices: list[int]) -> dict[str, list[float]]:
    """
    Compute t-stats for all beta_indices under three SE schemes.
    Returns dict keyed by SE label, value = list of t-stats (one per beta_idx).
    """
    res_firm = fit.get_robustcov_results(cov_type="cluster", groups=groups_permno)
    res_date = fit.get_robustcov_results(cov_type="cluster", groups=groups_date)

    out = {}
    for label, res, two_way in [
        ("Firm-clustered", res_firm, False),
        ("Date-clustered", res_date, False),
        ("Two-way (CGM)",  None,     True),
    ]:
        if two_way:
            tstats = [_two_way_tstat(fit, groups_permno, groups_date, idx)
                      for idx in beta_indices]
        else:
            tstats = [float(res.tvalues[idx]) for idx in beta_indices]
        out[label] = tstats

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ------------------------------------------------------------------
    # Load forward returns
    # ------------------------------------------------------------------
    fwd = pd.read_parquet(FWD_RETURNS_PATH)[["event_id", "cum_return"]].rename(
        columns={"cum_return": "forward_return_20d"}
    )
    print(f"Loaded forward returns: {len(fwd):,} events")

    # ------------------------------------------------------------------
    # Load / build SUE — merge key: (gvkey, datadate)
    # We pull gvkey + datadate from the canonical event_table
    # ------------------------------------------------------------------
    event_table = pd.read_parquet(PROCESSED / "event_table.parquet")[
        ["event_id", "gvkey", "datadate"]
    ]
    event_table["datadate"] = pd.to_datetime(event_table["datadate"])

    sue_df = fetch_sue(event_table)

    # Map SUE onto events via (gvkey, datadate)
    sue_events = event_table.merge(sue_df, on=["gvkey", "datadate"], how="left")
    sue_events = sue_events[["event_id", "sue"]].dropna(subset=["sue"])
    print(f"SUE matched for {len(sue_events):,} / {len(event_table):,} events\n")

    # Standardize SUE once across the full matched sample
    sue_mean = sue_events["sue"].mean()
    sue_std  = sue_events["sue"].std(ddof=1)
    sue_events["sue_std"] = (sue_events["sue"] - sue_mean) / sue_std

    # ------------------------------------------------------------------
    # Per-method loop
    # ------------------------------------------------------------------
    all_rows = []

    for method, cfg in METHOD_CONFIG.items():
        print(f"{'='*60}")
        print(f" Method {method}")
        print(f"{'='*60}")

        # Load method features
        feat = pd.read_parquet(METHODS_DIR / f"{method}_event_features.parquet")
        date_col = cfg["date_col"]
        if date_col != "event_date":
            feat = feat.rename(columns={date_col: "event_date"})
        feat = feat[["event_id", "permno", "event_date", "pressure_score"]]

        # Merge: features + forward returns + SUE
        df = (
            feat
            .merge(fwd,        on="event_id", how="inner")
            .merge(sue_events,  on="event_id", how="inner")
            .dropna(subset=["pressure_score", "forward_return_20d", "sue_std"])
        )

        # Standardize pressure_score within this method's sample
        ps_mean = df["pressure_score"].mean()
        ps_std  = df["pressure_score"].std(ddof=1)
        df["pressure_std"] = (df["pressure_score"] - ps_mean) / ps_std

        # Interaction term
        df["interaction"] = df["pressure_std"] * df["sue_std"]

        n               = len(df)
        y               = df["forward_return_20d"]
        groups_permno   = df["permno"].astype(str)
        groups_date     = df["event_date"].astype(str)

        # ----------------------------------------------------------
        # Spec 1: pressure_std + sue_std
        # Coeff order: const(0), pressure_std(1), sue_std(2)
        # ----------------------------------------------------------
        X1   = sm.add_constant(df[["pressure_std", "sue_std"]])
        fit1 = sm.OLS(y, X1).fit()

        betas1      = fit1.params.values          # numpy array
        r2_1        = float(fit1.rsquared)
        beta_idx1   = [1, 2]                      # pressure, sue
        tstats1     = _clustered_tstats(fit1, groups_permno, groups_date, beta_idx1)

        print(f"\n  -- Spec 1: pressure_std + sue_std --  (N={n:,}, R²={r2_1:.6f})")
        _print_spec(tstats1, ["beta_pressure", "beta_sue"],
                    [betas1[1], betas1[2]])

        for se_label, ts in tstats1.items():
            all_rows.append({
                "method":           method,
                "spec":             "Spec1",
                "se_type":          se_label,
                "beta_pressure":    float(betas1[1]),
                "tstat_pressure":   ts[0],
                "beta_sue":         float(betas1[2]),
                "tstat_sue":        ts[1],
                "beta_interaction": np.nan,
                "tstat_interaction":np.nan,
                "r2":               r2_1,
                "n":                n,
            })

        # ----------------------------------------------------------
        # Spec 2: pressure_std + sue_std + interaction
        # Coeff order: const(0), pressure_std(1), sue_std(2), interaction(3)
        # ----------------------------------------------------------
        X2   = sm.add_constant(df[["pressure_std", "sue_std", "interaction"]])
        fit2 = sm.OLS(y, X2).fit()

        betas2      = fit2.params.values
        r2_2        = float(fit2.rsquared)
        beta_idx2   = [1, 2, 3]
        tstats2     = _clustered_tstats(fit2, groups_permno, groups_date, beta_idx2)

        print(f"\n  -- Spec 2: + interaction term --  (N={n:,}, R²={r2_2:.6f})")
        _print_spec(tstats2, ["beta_pressure", "beta_sue", "beta_interaction"],
                    [betas2[1], betas2[2], betas2[3]])

        for se_label, ts in tstats2.items():
            all_rows.append({
                "method":           method,
                "spec":             "Spec2",
                "se_type":          se_label,
                "beta_pressure":    float(betas2[1]),
                "tstat_pressure":   ts[0],
                "beta_sue":         float(betas2[2]),
                "tstat_sue":        ts[1],
                "beta_interaction": float(betas2[3]),
                "tstat_interaction":ts[2],
                "r2":               r2_2,
                "n":                n,
            })

        print()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    cols = [
        "method", "spec", "se_type",
        "beta_pressure", "tstat_pressure",
        "beta_sue",      "tstat_sue",
        "beta_interaction", "tstat_interaction",
        "r2", "n",
    ]
    summary = pd.DataFrame(all_rows, columns=cols)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}  ({len(summary)} rows)")


# ---------------------------------------------------------------------------
# Console print helper
# ---------------------------------------------------------------------------
def _print_spec(tstats_by_se: dict, labels: list[str], betas: list[float]) -> None:
    """Print a compact table: one column per beta, one row per SE type."""
    # Header
    hdr_parts = [f"  {'SE type':<22}"]
    for lbl, b in zip(labels, betas):
        hdr_parts.append(f"  {lbl:>18} (β={b:+.6f})")
    print("".join(hdr_parts))
    print("  " + "-" * (22 + 28 * len(labels)))
    for se_label, ts in tstats_by_se.items():
        row = f"  {se_label:<22}"
        for t in ts:
            row += f"  {'t='+f'{t:+.4f}':>28}"
        print(row)


if __name__ == "__main__":
    main()

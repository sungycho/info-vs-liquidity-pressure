"""
Cross-sectional size-loading check for Methods A-D:

    pressure_score = alpha + beta * log(market_cap) + error

Market cap source:
    data/processed/crsp_daily.parquet
    market_cap = abs(prc) * shrout

Join logic:
    For each event, attach the latest available CRSP market cap on or before
    event_date using an as-of merge by permno.

Outputs:
    data/results/pressure_vs_log_mcap/pressure_vs_log_mcap_summary.csv
    data/results/pressure_vs_log_mcap/pressure_vs_log_mcap_samples.parquet
    data/results/pressure_vs_log_mcap/pressure_vs_log_mcap_report.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "Methods_Sung_Cho"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "data" / "results" / "pressure_vs_log_mcap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRSP_DAILY_PATH = PROCESSED_DIR / "crsp_daily.parquet"

OUTPUT_SUMMARY_CSV = OUTPUT_DIR / "pressure_vs_log_mcap_summary.csv"
OUTPUT_SAMPLES_PARQUET = OUTPUT_DIR / "pressure_vs_log_mcap_samples.parquet"
OUTPUT_REPORT_TXT = OUTPUT_DIR / "pressure_vs_log_mcap_report.txt"

METHOD_CONFIG = {
    "A": {"path": METHODS_DIR / "A_event_features.parquet", "date_col": "event_date"},
    "B": {"path": METHODS_DIR / "B_event_features.parquet", "date_col": "event_date"},
    "C": {"path": METHODS_DIR / "C_event_features.parquet", "date_col": "event_date"},
    "D": {"path": METHODS_DIR / "D_event_features.parquet", "date_col": "rdq"},
}


def load_crsp_market_cap(crsp_path: Path) -> pd.DataFrame:
    """Load CRSP daily and compute market cap + log market cap."""
    crsp = pd.read_parquet(crsp_path, columns=["permno", "date", "prc", "shrout"]).copy()
    crsp["date"] = pd.to_datetime(crsp["date"]).astype("datetime64[ns]")
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce")

    prc = pd.to_numeric(crsp["prc"], errors="coerce").abs()
    shrout = pd.to_numeric(crsp["shrout"], errors="coerce")

    crsp["market_cap"] = prc * shrout
    crsp["log_market_cap"] = np.log(crsp["market_cap"].where(crsp["market_cap"] > 0))

    crsp = crsp.dropna(subset=["permno", "date", "market_cap", "log_market_cap"])
    crsp["permno"] = crsp["permno"].astype("int64")
    crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True)
    return crsp[["permno", "date", "market_cap", "log_market_cap"]]


def load_method_events(method: str, path: Path, date_col: str) -> pd.DataFrame:
    """Load event-level pressure scores for one method."""
    feat = pd.read_parquet(path).copy()

    required_cols = {"event_id", "permno", "pressure_score", date_col}
    missing = required_cols - set(feat.columns)
    if missing:
        raise ValueError(f"Method {method}: missing columns {sorted(missing)} in {path}")

    events = feat[["event_id", "permno", date_col, "pressure_score"]].copy()
    if date_col != "event_date":
        events = events.rename(columns={date_col: "event_date"})

    events["event_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    events["permno"] = pd.to_numeric(events["permno"], errors="coerce")
    events = events.dropna(subset=["event_date", "permno", "pressure_score"])
    events["permno"] = events["permno"].astype("int64")
    events = events.sort_values(["permno", "event_date"]).reset_index(drop=True)
    return events


def attach_market_cap_asof(events: pd.DataFrame, crsp_mcap: pd.DataFrame) -> pd.DataFrame:
    """
    Attach latest available market cap on or before event_date by permno.
    """
    out = []

    for permno, evt_grp in events.groupby("permno", sort=False):
        evt_grp = evt_grp.sort_values("event_date").copy()
        mcap_grp = (
            crsp_mcap.loc[crsp_mcap["permno"] == permno, ["date", "market_cap", "log_market_cap"]]
            .sort_values("date")
            .copy()
        )

        if mcap_grp.empty:
            evt_grp["mcap_date"] = pd.NaT
            evt_grp["market_cap"] = np.nan
            evt_grp["log_market_cap"] = np.nan
            evt_grp["mcap_lag_days"] = np.nan
            out.append(evt_grp)
            continue

        merged_grp = pd.merge_asof(
            evt_grp,
            mcap_grp,
            left_on="event_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=True,
        ).rename(columns={"date": "mcap_date"})

        merged_grp["mcap_lag_days"] = (
            merged_grp["event_date"] - merged_grp["mcap_date"]
        ).dt.days
        out.append(merged_grp)

    return pd.concat(out, ignore_index=True)


def run_pressure_on_log_mcap(df: pd.DataFrame) -> tuple[dict, sm.regression.linear_model.RegressionResultsWrapper]:
    """Run OLS: pressure_score ~ log_market_cap (with intercept)."""
    reg_df = df.copy()
    reg_df["pressure_score"] = pd.to_numeric(reg_df["pressure_score"], errors="coerce")
    reg_df["log_market_cap"] = pd.to_numeric(reg_df["log_market_cap"], errors="coerce")
    reg_df = reg_df.dropna(subset=["pressure_score", "log_market_cap"]).copy()
    n = len(reg_df)
    if n < 10:
        raise ValueError(f"Not enough rows for regression (n={n})")

    y = reg_df["pressure_score"].astype(float)
    x = sm.add_constant(reg_df["log_market_cap"].astype(float))
    fit = sm.OLS(y, x).fit()
    fit_hc3 = fit.get_robustcov_results(cov_type="HC3")

    beta_idx = 1
    summary = {
        "n": int(n),
        "beta_log_mcap": float(fit.params.iloc[beta_idx]),
        "tstat_ols": float(fit.tvalues.iloc[beta_idx]),
        "pvalue_ols": float(fit.pvalues.iloc[beta_idx]),
        "tstat_hc3": float(fit_hc3.tvalues[beta_idx]),
        "pvalue_hc3": float(fit_hc3.pvalues[beta_idx]),
        "intercept": float(fit.params.iloc[0]),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "corr_pressure_log_mcap": float(reg_df["pressure_score"].corr(reg_df["log_market_cap"])),
        "mean_log_mcap": float(reg_df["log_market_cap"].mean()),
        "std_log_mcap": float(reg_df["log_market_cap"].std(ddof=1)),
        "mean_pressure": float(reg_df["pressure_score"].mean()),
        "std_pressure": float(reg_df["pressure_score"].std(ddof=1)),
    }
    return summary, fit


def main() -> None:
    print("Loading CRSP market cap panel...")
    crsp_mcap = load_crsp_market_cap(CRSP_DAILY_PATH)
    print(f"  CRSP rows: {len(crsp_mcap):,}")

    summary_rows: list[dict] = []
    sample_frames: list[pd.DataFrame] = []
    report_lines: list[str] = []

    for method, cfg in METHOD_CONFIG.items():
        print(f"\n=== Method {method} ===")
        events = load_method_events(method, cfg["path"], cfg["date_col"])
        merged = attach_market_cap_asof(events, crsp_mcap)

        total_events = len(merged)
        matched = merged["log_market_cap"].notna().sum()
        coverage = matched / total_events if total_events else np.nan

        print(f"  Events loaded: {total_events:,}")
        print(f"  Market-cap matched: {matched:,} ({coverage:.2%})")

        reg_input = merged.dropna(subset=["pressure_score", "log_market_cap"]).copy()
        stats, fit = run_pressure_on_log_mcap(reg_input)
        stats.update(
            {
                "method": method,
                "events_loaded": int(total_events),
                "mcap_matched": int(matched),
                "mcap_match_rate": float(coverage),
                "avg_mcap_lag_days": float(reg_input["mcap_lag_days"].mean()),
                "median_mcap_lag_days": float(reg_input["mcap_lag_days"].median()),
            }
        )
        summary_rows.append(stats)

        sample = reg_input[
            [
                "event_id",
                "permno",
                "event_date",
                "pressure_score",
                "mcap_date",
                "mcap_lag_days",
                "market_cap",
                "log_market_cap",
            ]
        ].copy()
        sample["method"] = method
        sample_frames.append(sample)

        report_lines.append(f"Method {method}")
        report_lines.append("-" * 80)
        report_lines.append(fit.summary().as_text())
        report_lines.append("\n")

        print(
            f"  beta(log_mcap)={stats['beta_log_mcap']:.6f}, "
            f"t(HC3)={stats['tstat_hc3']:.3f}, p(HC3)={stats['pvalue_hc3']:.4g}, "
            f"R2={stats['r2']:.4f}"
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("method").reset_index(drop=True)
    samples_df = pd.concat(sample_frames, ignore_index=True)

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    samples_df.to_parquet(OUTPUT_SAMPLES_PARQUET, index=False)
    OUTPUT_REPORT_TXT.write_text("\n".join(report_lines))

    print("\nSaved outputs:")
    print(f"  {OUTPUT_SUMMARY_CSV}")
    print(f"  {OUTPUT_SAMPLES_PARQUET}")
    print(f"  {OUTPUT_REPORT_TXT}")


if __name__ == "__main__":
    main()

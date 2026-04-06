#!/usr/bin/env python3
"""
Size-falsification diagnostic suite for feature-by-feature CAR tests.

Outputs are saved under:
    Methods_June_Lee/feature_car_tests2/size_diagnostics/
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


np.random.seed(0)


@dataclass
class FitResult:
    params: dict[str, float]
    tvals_two_way: dict[str, float]
    r2: float
    n: int


def _safe_autocorr(series: pd.Series, lag: int = 1) -> float:
    x = series.dropna()
    if len(x) < 4:
        return np.nan
    if x.std(ddof=1) == 0:
        return np.nan
    return float(x.autocorr(lag=lag))


def _parse_wrds_credentials(root: Path) -> tuple[str, str]:
    username = os.environ.get("WRDS_USERNAME", "")
    password = os.environ.get("WRDS_PASSWORD", "")

    if username and password:
        return username, password

    env_path = root / ".env"
    if env_path.exists():
        text = env_path.read_text()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m_user = re.match(r'WRDS_USERNAME[=:\s]+"?([^"\s]+)"?', line)
            if m_user:
                username = m_user.group(1)
            m_pass = re.match(r'WRDS_PASSWORD[=:\s]+"?([^"\s]+)"?', line)
            if m_pass:
                password = m_pass.group(1)

    if not username or not password:
        raise RuntimeError("WRDS credentials not found in env or .env")
    return username, password


def _wrds_connect(root: Path):
    import psycopg2

    username, password = _parse_wrds_credentials(root)
    return psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        dbname="wrds",
        user=username,
        password=password,
        sslmode="require",
        connect_timeout=60,
    )


def _chunked(items: Iterable[int], size: int) -> Iterable[list[int]]:
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fetch_wrds_mcap_panel(
    root: Path,
    permnos: list[int],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Pull market cap panel from WRDS CRSP daily stock file:
        market_cap = abs(prc) * shrout
        log_market_cap = log(market_cap)
    """
    conn = _wrds_connect(root)
    frames: list[pd.DataFrame] = []
    try:
        for chunk in _chunked(sorted(set(permnos)), 400):
            permnos_sql = ",".join(str(int(x)) for x in chunk)
            query = f"""
                SELECT
                    permno::bigint AS permno,
                    date,
                    ABS(prc) * shrout AS market_cap
                FROM crsp.dsf
                WHERE permno IN ({permnos_sql})
                  AND date BETWEEN '{min_date.date()}' AND '{max_date.date()}'
                  AND prc IS NOT NULL
                  AND shrout IS NOT NULL
            """
            frames.append(pd.read_sql(query, conn))
    finally:
        conn.close()

    if not frames:
        return pd.DataFrame(columns=["permno", "date", "market_cap", "log_market_cap"])

    mcap = pd.concat(frames, ignore_index=True)
    mcap["permno"] = pd.to_numeric(mcap["permno"], errors="coerce").astype("Int64")
    mcap["date"] = pd.to_datetime(mcap["date"])
    mcap["market_cap"] = pd.to_numeric(mcap["market_cap"], errors="coerce")
    mcap = mcap.dropna(subset=["permno", "date", "market_cap"])
    mcap = mcap[mcap["market_cap"] > 0].copy()
    mcap["log_market_cap"] = np.log(mcap["market_cap"])
    return mcap[["permno", "date", "market_cap", "log_market_cap"]]


def build_local_mcap_panel(crsp_daily_path: Path) -> pd.DataFrame:
    """Fallback mcap from local CRSP if prc/shrout columns are available."""
    crsp = pd.read_parquet(crsp_daily_path)
    needed = {"permno", "date", "prc", "shrout"}
    if not needed.issubset(crsp.columns):
        raise RuntimeError("Local CRSP file lacks prc/shrout needed for fallback mcap.")

    out = crsp[["permno", "date", "prc", "shrout"]].copy()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"])
    out["prc"] = pd.to_numeric(out["prc"], errors="coerce").abs()
    out["shrout"] = pd.to_numeric(out["shrout"], errors="coerce")
    out["market_cap"] = out["prc"] * out["shrout"]
    out = out.dropna(subset=["permno", "date", "market_cap"])
    out = out[out["market_cap"] > 0].copy()
    out["log_market_cap"] = np.log(out["market_cap"])
    return out[["permno", "date", "market_cap", "log_market_cap"]]


def attach_mcap_asof(events: pd.DataFrame, mcap: pd.DataFrame) -> pd.DataFrame:
    merged_parts: list[pd.DataFrame] = []
    events = events.sort_values(["permno", "event_date"]).copy()
    mcap = mcap.sort_values(["permno", "date"]).copy()
    events["event_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    mcap["date"] = pd.to_datetime(mcap["date"]).astype("datetime64[ns]")

    for permno, egrp in events.groupby("permno", sort=False):
        sgrp = mcap[mcap["permno"] == permno][["date", "market_cap", "log_market_cap"]]
        if sgrp.empty:
            tmp = egrp.copy()
            tmp["mcap_date"] = pd.NaT
            tmp["market_cap"] = np.nan
            tmp["log_market_cap"] = np.nan
            tmp["mcap_lag_days"] = np.nan
            merged_parts.append(tmp)
            continue

        tmp = pd.merge_asof(
            egrp.sort_values("event_date"),
            sgrp.sort_values("date"),
            left_on="event_date",
            right_on="date",
            direction="backward",
        ).rename(columns={"date": "mcap_date"})

        tmp["mcap_lag_days"] = (tmp["event_date"] - tmp["mcap_date"]).dt.days
        merged_parts.append(tmp)

    return pd.concat(merged_parts, ignore_index=True)


def compute_event_car(
    event_table: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    start_offset: int = 2,
    end_offset: int = 20,
) -> pd.DataFrame:
    """
    CAR[+2,+20], where event time 0 is first CRSP trading day on/after rdq.
    CAR = prod(1 + ret_t) - 1 over the inclusive return window.
    """
    events = event_table[["event_id", "permno", "rdq"]].copy()
    events["permno"] = pd.to_numeric(events["permno"], errors="coerce").astype("Int64")
    events["rdq"] = pd.to_datetime(events["rdq"])
    events = events.dropna(subset=["event_id", "permno", "rdq"])

    crsp = crsp_daily[["permno", "date", "ret"]].copy()
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64")
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["permno", "date", "ret"]).sort_values(["permno", "date"])

    by_permno = {
        int(p): g[["date", "ret"]].reset_index(drop=True)
        for p, g in crsp.groupby("permno", sort=False)
    }

    rows: list[dict] = []
    for row in events.itertuples(index=False):
        permno = int(row.permno)
        rdq = pd.Timestamp(row.rdq)
        panel = by_permno.get(permno)
        if panel is None:
            continue

        idx0 = int(panel["date"].searchsorted(rdq, side="left"))
        start_idx = idx0 + start_offset
        end_idx = idx0 + end_offset
        if idx0 >= len(panel) or end_idx >= len(panel):
            continue

        ret_window = panel["ret"].iloc[start_idx : end_idx + 1]
        if ret_window.isna().any():
            continue

        car = float(np.prod(1.0 + ret_window.to_numpy(dtype=float)) - 1.0)
        rows.append(
            {
                "event_id": row.event_id,
                "permno": permno,
                "event_date": rdq.normalize(),
                "car_2_20": car,
                "car_num_days": int(end_idx - start_idx + 1),
            }
        )

    out = pd.DataFrame(rows)
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out["event_date"] = pd.to_datetime(out["event_date"])
    return out


def build_event_features_from_daily(daily_features: pd.DataFrame) -> pd.DataFrame:
    """Fallback builder when data/processed/event_features.parquet is unavailable."""
    df = daily_features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["event_id", "date"])

    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_cols = [c for c in numeric if c != "permno"]

    event_means = df.groupby("event_id")[mean_cols].mean().add_suffix("_mean")
    event_meta = df.groupby("event_id", as_index=True).agg(permno=("permno", "first"))
    out = event_meta.join(event_means)

    if "ofi" in df.columns:
        out["ofi_autocorr"] = df.groupby("event_id")["ofi"].apply(_safe_autocorr)
    if "abnormal_vol_ratio" in df.columns:
        out["volume_burst_fraction"] = df.groupby("event_id")["abnormal_vol_ratio"].apply(
            lambda x: float((x > 1.5).mean()) if x.notna().any() else np.nan
        )

    out = out.reset_index()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out


def get_feature_list_from_regression_summary(reg_path: Path) -> list[str]:
    reg = pd.read_csv(reg_path)
    return sorted(reg["feature"].dropna().unique().tolist())


def _cluster_cov(fit, groups: pd.Series) -> np.ndarray:
    return fit.get_robustcov_results(cov_type="cluster", groups=groups).cov_params()


def fit_two_way(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    firm_col: str = "permno",
    date_col: str = "event_date",
) -> FitResult | None:
    use_cols = [y_col, firm_col, date_col] + x_cols
    work = df[use_cols].copy().dropna()
    n = int(len(work))
    if n <= len(x_cols) + 1:
        return None

    y = pd.to_numeric(work[y_col], errors="coerce")
    X = sm.add_constant(work[x_cols], has_constant="add")
    model_df = pd.concat([y, X, work[[firm_col, date_col]]], axis=1).dropna()
    n = int(len(model_df))
    if n <= len(x_cols) + 1:
        return None

    y = model_df[y_col].astype(float)
    X = model_df[X.columns].astype(float)
    try:
        fit = sm.OLS(y, X).fit()
    except Exception:
        return None

    groups_firm = model_df[firm_col].astype(str)
    groups_date = pd.to_datetime(model_df[date_col]).dt.strftime("%Y-%m-%d")
    params = {k: float(v) for k, v in fit.params.items()}

    tvals: dict[str, float] = {k: np.nan for k in fit.params.index.tolist()}
    try:
        cov_firm = _cluster_cov(fit, groups_firm)
        cov_date = _cluster_cov(fit, groups_date)
        cov_inter = _cluster_cov(fit, groups_firm + "_" + groups_date)
        cov_2w = cov_firm + cov_date - cov_inter
        variances = np.diag(cov_2w)
        for idx, name in enumerate(fit.params.index):
            var = float(variances[idx])
            if np.isfinite(var) and var > 0:
                tvals[name] = float(fit.params.iloc[idx] / np.sqrt(var))
    except Exception:
        pass

    return FitResult(params=params, tvals_two_way=tvals, r2=float(fit.rsquared), n=n)


def run_size_control_regressions(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i, feature in enumerate(features, start=1):
        if i % 25 == 0 or i == len(features):
            print(f"Step 1 regressions: {i}/{len(features)}")

        work = df[["car_2_20", "permno", "event_date", "sue", "log_market_cap", feature]].copy()
        work = work.rename(columns={feature: "feature"})
        work["interaction"] = work["feature"] * work["sue"]

        for spec_name, x_cols in [
            ("A_with_size", ["feature", "sue", "log_market_cap"]),
            ("B_with_size_interaction", ["feature", "sue", "log_market_cap", "interaction"]),
        ]:
            res = fit_two_way(work, y_col="car_2_20", x_cols=x_cols)
            if res is None:
                rows.append(
                    {
                        "feature": feature,
                        "spec": spec_name,
                        "beta_X": np.nan,
                        "t_X": np.nan,
                        "beta_interaction": np.nan,
                        "t_interaction": np.nan,
                        "beta_log_mcap": np.nan,
                        "t_log_mcap": np.nan,
                        "r2": np.nan,
                        "n": 0,
                    }
                )
                continue

            rows.append(
                {
                    "feature": feature,
                    "spec": spec_name,
                    "beta_X": res.params.get("feature", np.nan),
                    "t_X": res.tvals_two_way.get("feature", np.nan),
                    "beta_interaction": res.params.get("interaction", np.nan),
                    "t_interaction": res.tvals_two_way.get("interaction", np.nan),
                    "beta_log_mcap": res.params.get("log_market_cap", np.nan),
                    "t_log_mcap": res.tvals_two_way.get("log_market_cap", np.nan),
                    "r2": res.r2,
                    "n": res.n,
                }
            )
    return pd.DataFrame(rows)


def run_size_bin_subsample(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    median_size = float(df["log_market_cap"].median())
    large = df[df["log_market_cap"] >= median_size].copy()
    small = df[df["log_market_cap"] < median_size].copy()

    rows: list[dict] = []
    for i, feature in enumerate(features, start=1):
        if i % 25 == 0 or i == len(features):
            print(f"Step 2 subsample: {i}/{len(features)}")

        large_work = large[["car_2_20", "permno", "event_date", "sue", feature]].rename(columns={feature: "feature"})
        small_work = small[["car_2_20", "permno", "event_date", "sue", feature]].rename(columns={feature: "feature"})

        r_large = fit_two_way(large_work, y_col="car_2_20", x_cols=["feature", "sue"])
        r_small = fit_two_way(small_work, y_col="car_2_20", x_cols=["feature", "sue"])

        b_large = np.nan if r_large is None else r_large.params.get("feature", np.nan)
        t_large = np.nan if r_large is None else r_large.tvals_two_way.get("feature", np.nan)
        b_small = np.nan if r_small is None else r_small.params.get("feature", np.nan)
        t_small = np.nan if r_small is None else r_small.tvals_two_way.get("feature", np.nan)

        sig_large = bool(np.isfinite(t_large) and abs(t_large) > 1.96)
        sig_small = bool(np.isfinite(t_small) and abs(t_small) > 1.96)

        if sig_large and sig_small:
            pattern = "both"
        elif sig_large and not sig_small:
            pattern = "large_only"
        elif (not sig_large) and sig_small:
            pattern = "small_only"
        else:
            pattern = "neither"

        rows.append(
            {
                "feature": feature,
                "median_log_mcap_split": median_size,
                "beta_large": b_large,
                "t_large": t_large,
                "n_large": 0 if r_large is None else r_large.n,
                "beta_small": b_small,
                "t_small": t_small,
                "n_small": 0 if r_small is None else r_small.n,
                "sig_large_5": sig_large,
                "sig_small_5": sig_small,
                "sig_pattern": pattern,
            }
        )
    return pd.DataFrame(rows)


def run_residualization(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i, feature in enumerate(features, start=1):
        if i % 25 == 0 or i == len(features):
            print(f"Step 3 residualization: {i}/{len(features)}")

        work = df[["car_2_20", "permno", "event_date", "sue", "log_market_cap", feature]].copy()
        work = work.rename(columns={feature: "feature"})
        work["feature"] = pd.to_numeric(work["feature"], errors="coerce")
        work["log_market_cap"] = pd.to_numeric(work["log_market_cap"], errors="coerce")
        work = work.dropna(subset=["feature", "log_market_cap"])
        if work.empty:
            rows.append(
                {
                    "feature": feature,
                    "beta_resid": np.nan,
                    "t_resid": np.nan,
                    "r2_resid_model": np.nan,
                    "n": 0,
                    "beta_x_on_size": np.nan,
                    "t_x_on_size": np.nan,
                    "r2_x_on_size": np.nan,
                }
            )
            continue

        X_x = sm.add_constant(work["log_market_cap"].astype(float), has_constant="add")
        fit_x = sm.OLS(work["feature"].astype(float), X_x).fit()
        work["feature_resid"] = work["feature"] - fit_x.fittedvalues

        reg = fit_two_way(work, y_col="car_2_20", x_cols=["feature_resid", "sue"])
        rows.append(
            {
                "feature": feature,
                "beta_resid": np.nan if reg is None else reg.params.get("feature_resid", np.nan),
                "t_resid": np.nan if reg is None else reg.tvals_two_way.get("feature_resid", np.nan),
                "r2_resid_model": np.nan if reg is None else reg.r2,
                "n": 0 if reg is None else reg.n,
                "beta_x_on_size": float(fit_x.params.get("log_market_cap", np.nan)),
                "t_x_on_size": float(fit_x.tvalues.get("log_market_cap", np.nan)),
                "r2_x_on_size": float(fit_x.rsquared),
            }
        )

    return pd.DataFrame(rows)


def infer_conclusion(
    n_orig: int,
    n_robust_all: int,
) -> str:
    if n_orig == 0:
        return "No originally significant features, so size-driven conclusion is not identifiable."
    share = n_robust_all / n_orig
    if share == 0:
        return "Liquidity/microstructure effects appear entirely size-driven under these diagnostics."
    if share < 0.5:
        return "Liquidity/microstructure effects appear partially size-driven."
    return "Liquidity/microstructure effects appear mostly independent of size."


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    out_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Size falsification diagnostic suite")
    p.add_argument("--event-features", type=Path, default=root / "data/processed/event_features.parquet")
    p.add_argument("--daily-features", type=Path, default=root / "data/processed/daily_features.parquet")
    p.add_argument("--event-table", type=Path, default=root / "data/processed/event_table.parquet")
    p.add_argument("--sue-events", type=Path, default=root / "data/processed/sue_event_level.parquet")
    p.add_argument("--crsp-daily", type=Path, default=root / "data/processed/crsp_daily.parquet")
    p.add_argument("--regression-summary", type=Path, default=root / "Methods_June_Lee/feature_car_tests/regression_summary.csv")
    p.add_argument("--out-dir", type=Path, default=out_dir)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading base inputs...")
    event_table = pd.read_parquet(args.event_table)
    crsp = pd.read_parquet(args.crsp_daily)
    sue = pd.read_parquet(args.sue_events)[["event_id", "sue"]].copy()
    sue["sue"] = pd.to_numeric(sue["sue"], errors="coerce")

    features_from_summary = get_feature_list_from_regression_summary(args.regression_summary)

    if args.event_features.exists():
        print(f"Using event features file: {args.event_features}")
        event_feat = pd.read_parquet(args.event_features)
        if "event_date" not in event_feat.columns:
            if "rdq" in event_feat.columns:
                event_feat = event_feat.rename(columns={"rdq": "event_date"})
            else:
                rdq_map = event_table[["event_id", "rdq"]].rename(columns={"rdq": "event_date"})
                event_feat = event_feat.merge(rdq_map, on="event_id", how="left")
        event_feat["event_date"] = pd.to_datetime(event_feat["event_date"])
        event_feat["permno"] = pd.to_numeric(event_feat["permno"], errors="coerce").astype("Int64")
    else:
        print("event_features.parquet not found. Reconstructing event features from daily_features.parquet.")
        daily = pd.read_parquet(args.daily_features)
        event_feat = build_event_features_from_daily(daily)
        rdq_map = event_table[["event_id", "rdq"]].rename(columns={"rdq": "event_date"})
        event_feat = event_feat.merge(rdq_map, on="event_id", how="left")
        event_feat["event_date"] = pd.to_datetime(event_feat["event_date"])

    feature_cols = [f for f in features_from_summary if f in event_feat.columns]
    print(f"Feature columns available: {len(feature_cols)}")
    if not feature_cols:
        raise RuntimeError("No feature columns from regression summary found in event feature dataframe.")

    print("Step 0: Reconstructing CAR[+2,+20]...")
    car = compute_event_car(event_table=event_table, crsp_daily=crsp, start_offset=2, end_offset=20)
    base = (
        car.merge(event_feat[["event_id", "permno", "event_date"] + feature_cols], on=["event_id", "permno", "event_date"], how="inner")
        .merge(sue, on="event_id", how="left")
    )
    base["event_date"] = pd.to_datetime(base["event_date"])

    print("Fetching/constructing log_market_cap...")
    events_for_mcap = base[["event_id", "permno", "event_date"]].drop_duplicates().copy()
    events_for_mcap["permno"] = pd.to_numeric(events_for_mcap["permno"], errors="coerce").astype("Int64")
    min_date = pd.to_datetime(events_for_mcap["event_date"].min()) - pd.Timedelta(days=31)
    max_date = pd.to_datetime(events_for_mcap["event_date"].max())
    permnos = events_for_mcap["permno"].dropna().astype(int).unique().tolist()

    mcap_source = "wrds"
    try:
        mcap_panel = fetch_wrds_mcap_panel(root, permnos, min_date, max_date)
        if mcap_panel.empty:
            raise RuntimeError("WRDS market cap query returned empty panel.")
        print(f"Market cap source: WRDS CRSP (rows={len(mcap_panel):,})")
    except Exception as e:
        mcap_source = "local_crsp_fallback"
        print(f"WRDS fetch failed ({e}). Falling back to local CRSP for market cap.")
        mcap_panel = build_local_mcap_panel(args.crsp_daily)
        print(f"Market cap source: local CRSP fallback (rows={len(mcap_panel):,})")

    with_mcap = attach_mcap_asof(events_for_mcap, mcap_panel)
    base = base.merge(
        with_mcap[["event_id", "log_market_cap", "market_cap", "mcap_date", "mcap_lag_days"]],
        on="event_id",
        how="left",
    )

    car_event_path = args.out_dir / "car_event_level.parquet"
    base.to_parquet(car_event_path, index=False)
    print(f"Saved Step 0 output: {car_event_path}")

    analysis = base.dropna(subset=["car_2_20", "sue", "log_market_cap"]).copy()
    print(f"Analysis events after sue + size merge: {len(analysis):,}")

    # No-control baseline from prior run: Two-way, B_with_sue
    reg0 = pd.read_csv(args.regression_summary)
    no_ctrl = reg0[(reg0["se_type"] == "two_way") & (reg0["spec"] == "B_with_sue")][
        ["feature", "beta_feature", "tstat_feature"]
    ].rename(columns={"beta_feature": "beta_no_control", "tstat_feature": "t_no_control"})

    print("Step 1: Size-control regressions...")
    with_size = run_size_control_regressions(analysis, feature_cols)
    with_size_path = args.out_dir / "with_size_control.csv"
    with_size.to_csv(with_size_path, index=False)

    compare = no_ctrl.merge(
        with_size[with_size["spec"] == "A_with_size"][["feature", "beta_X", "t_X"]],
        on="feature",
        how="inner",
    ).rename(columns={"beta_X": "beta_with_control", "t_X": "t_with_control"})
    compare["pct_change_abs_t"] = (
        (compare["t_with_control"].abs() - compare["t_no_control"].abs())
        / compare["t_no_control"].abs().replace(0, np.nan)
        * 100.0
    )
    compare["remains_sig_5"] = compare["t_with_control"].abs() > 1.96
    compare_path = args.out_dir / "size_control_comparison.csv"
    compare.to_csv(compare_path, index=False)

    print("Step 2: Within-size-bin subsample regressions...")
    subsample = run_size_bin_subsample(analysis, feature_cols)
    subsample_path = args.out_dir / "size_bin_subsample_results.csv"
    subsample.to_csv(subsample_path, index=False)

    print("Step 3: Residualization on size...")
    resid = run_residualization(analysis, feature_cols)
    resid = resid.merge(no_ctrl, on="feature", how="left")
    resid_path = args.out_dir / "residualized_feature_results.csv"
    resid.to_csv(resid_path, index=False)

    resid_comp = resid[["feature", "t_no_control", "t_resid"]].copy()
    resid_comp = resid_comp.rename(columns={"t_no_control": "t_original"})
    resid_comp["survives_5"] = resid_comp["t_resid"].abs() > 1.96
    resid_comp_path = args.out_dir / "residualization_comparison.csv"
    resid_comp.to_csv(resid_comp_path, index=False)

    print("\nStep 4: Diagnostic summary")
    originally_sig = set(no_ctrl.loc[no_ctrl["t_no_control"].abs() > 1.96, "feature"])
    after_size_ctrl = set(compare.loc[compare["remains_sig_5"], "feature"])
    after_subsample = set(subsample.loc[subsample["sig_pattern"] == "both", "feature"])
    after_resid = set(resid_comp.loc[resid_comp["survives_5"], "feature"])

    robust_all = originally_sig & after_size_ctrl & after_subsample & after_resid

    print(f"- Originally significant features (Two-way, Spec1=B_with_sue): {len(originally_sig)}")
    print(f"- Remaining significant after size control: {len(after_size_ctrl & originally_sig)}")
    print(f"- Remaining significant after size-bin split (both large & small): {len(after_subsample & originally_sig)}")
    print(f"- Remaining significant after residualization: {len(after_resid & originally_sig)}")
    print(f"- Robust across all three tests: {len(robust_all)}")
    print(f"- Robust features: {sorted(robust_all)}")
    print(f"- Size conclusion: {infer_conclusion(len(originally_sig), len(robust_all))}")

    print("\nSaved outputs:")
    print(f"- {car_event_path}")
    print(f"- {with_size_path}")
    print(f"- {compare_path}")
    print(f"- {subsample_path}")
    print(f"- {resid_path}")
    print(f"- {resid_comp_path}")
    print(f"- mcap_source={mcap_source}")


if __name__ == "__main__":
    main()

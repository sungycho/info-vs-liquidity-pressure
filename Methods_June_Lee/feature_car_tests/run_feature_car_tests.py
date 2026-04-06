#!/usr/bin/env python3
"""
Feature-by-feature predictive tests for post-earnings CAR[+2,+20].

Workflow:
1. Build event-level features from `daily_features.parquet`.
2. Build event-level CAR[+2,+20] from `crsp_daily.parquet`.
3. Merge SUE from `sue_event_level.parquet`.
4. Run three regressions per feature:
   A) CAR ~ feature
   B) CAR ~ feature + sue
   C) CAR ~ feature + sue + feature*sue
   with firm/date/two-way clustered SE.
5. Compute quintile tests.
6. Save:
   - regression_summary.csv
   - quintile_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def safe_lag1_autocorr(series: pd.Series) -> float:
    """Lag-1 autocorrelation with guardrails for short or constant series."""
    x = series.dropna()
    if len(x) < 4:
        return np.nan
    if x.std(ddof=1) == 0:
        return np.nan
    return float(x.autocorr(lag=1))


def build_event_features(
    daily_features: pd.DataFrame,
    volume_burst_threshold: float = 1.5,
) -> pd.DataFrame:
    """Aggregate stock-day features to one row per event."""
    df = daily_features.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["event_id", "date"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_cols = [c for c in numeric_cols if c != "permno"]

    event_means = df.groupby("event_id")[mean_cols].mean().add_suffix("_mean")
    event_meta = df.groupby("event_id", as_index=True).agg(permno=("permno", "first"))

    out = event_meta.join(event_means)

    if "ofi" in df.columns:
        out["ofi_autocorr"] = df.groupby("event_id")["ofi"].apply(safe_lag1_autocorr)

    if "abnormal_vol_ratio" in df.columns:
        out["volume_burst_fraction"] = df.groupby("event_id")["abnormal_vol_ratio"].apply(
            lambda x: float((x > volume_burst_threshold).mean()) if x.notna().any() else np.nan
        )

    out = out.reset_index()
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out


def compute_event_car(
    event_table: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    start_offset: int = 2,
    end_offset: int = 20,
) -> pd.DataFrame:
    """
    Compute CAR[+2,+20] using simple cumulative returns:
        CAR = prod(1 + r_t) - 1, for t in [+2, +20].

    Anchor date is RDQ aligned to first trading day on/after RDQ.
    """
    events = event_table[["event_id", "permno", "rdq"]].copy()
    events["rdq"] = pd.to_datetime(events["rdq"])
    events["permno"] = pd.to_numeric(events["permno"], errors="coerce").astype("Int64")
    events = events.dropna(subset=["event_id", "permno", "rdq"])

    crsp = crsp_daily[["permno", "date", "ret"]].copy()
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64")
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["permno", "date", "ret"]).sort_values(["permno", "date"])

    by_permno = {
        int(permno): g[["date", "ret"]].reset_index(drop=True)
        for permno, g in crsp.groupby("permno", sort=False)
    }

    rows: list[dict] = []
    for event in events.itertuples(index=False):
        permno = int(event.permno)
        rdq = pd.Timestamp(event.rdq)
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
                "event_id": event.event_id,
                "permno": permno,
                "event_date": rdq.normalize(),
                "car_2_20": car,
                "car_num_days": int(end_idx - start_idx + 1),
            }
        )

    out = pd.DataFrame(rows)
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    return out


def _cluster_cov(fit, groups: pd.Series) -> np.ndarray:
    return fit.get_robustcov_results(cov_type="cluster", groups=groups).cov_params()


def compute_clustered_tstats(
    fit,
    groups_firm: pd.Series,
    groups_date: pd.Series,
    beta_indices: dict[str, int | None],
) -> dict[str, dict[str, float]]:
    """Compute firm/date/two-way clustered t-stats for selected coefficients."""
    out = {
        "firm": {key: np.nan for key in beta_indices},
        "date": {key: np.nan for key in beta_indices},
        "two_way": {key: np.nan for key in beta_indices},
    }

    try:
        res_firm = fit.get_robustcov_results(cov_type="cluster", groups=groups_firm)
        for name, idx in beta_indices.items():
            if idx is not None:
                out["firm"][name] = float(res_firm.tvalues[idx])
    except Exception:
        pass

    try:
        res_date = fit.get_robustcov_results(cov_type="cluster", groups=groups_date)
        for name, idx in beta_indices.items():
            if idx is not None:
                out["date"][name] = float(res_date.tvalues[idx])
    except Exception:
        pass

    try:
        cov_firm = _cluster_cov(fit, groups_firm)
        cov_date = _cluster_cov(fit, groups_date)
        cov_intersection = _cluster_cov(
            fit,
            groups_firm.astype(str) + "_" + groups_date.astype(str),
        )
        cov_2w = cov_firm + cov_date - cov_intersection
        var_2w = np.diag(cov_2w)
        for name, idx in beta_indices.items():
            if idx is None:
                continue
            variance = float(var_2w[idx])
            if np.isfinite(variance) and variance > 0:
                out["two_way"][name] = float(fit.params.iloc[idx] / np.sqrt(variance))
    except Exception:
        pass

    return out


def run_feature_regressions(analysis_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Run specs A/B/C per feature and return long-form regression summary."""
    rows: list[dict] = []
    y_col = "car_2_20"

    spec_defs = [
        ("A_univariate", ["feature"]),
        ("B_with_sue", ["feature", "sue"]),
        ("C_with_interaction", ["feature", "sue", "interaction"]),
    ]

    for i, feature in enumerate(feature_cols, start=1):
        if i % 25 == 0 or i == len(feature_cols):
            print(f"Running regressions: {i}/{len(feature_cols)} features")

        for spec_name, rhs in spec_defs:
            needed = [y_col, "permno", "event_date", feature]
            if "sue" in rhs:
                needed.append("sue")

            df = analysis_df[needed].copy().dropna()
            if df.empty:
                continue

            x_df = pd.DataFrame(index=df.index)
            x_df["feature"] = pd.to_numeric(df[feature], errors="coerce")
            if "sue" in rhs:
                x_df["sue"] = pd.to_numeric(df["sue"], errors="coerce")
            if "interaction" in rhs:
                x_df["interaction"] = x_df["feature"] * x_df["sue"]

            model_df = pd.concat(
                [df[[y_col, "permno", "event_date"]], x_df],
                axis=1,
            ).dropna()
            n_obs = int(len(model_df))

            if n_obs <= len(rhs) + 1:
                continue

            y = pd.to_numeric(model_df[y_col], errors="coerce")
            X = sm.add_constant(model_df[rhs], has_constant="add")

            try:
                fit = sm.OLS(y, X).fit()
            except Exception:
                continue

            param_index = {name: idx for idx, name in enumerate(fit.params.index)}
            beta_feature = float(fit.params.get("feature", np.nan))
            beta_interaction = float(fit.params.get("interaction", np.nan))
            r2 = float(fit.rsquared)

            beta_indices = {
                "feature": param_index.get("feature"),
                "interaction": param_index.get("interaction"),
            }

            groups_firm = model_df["permno"].astype(str)
            groups_date = pd.to_datetime(model_df["event_date"]).dt.strftime("%Y-%m-%d")
            tstats = compute_clustered_tstats(fit, groups_firm, groups_date, beta_indices)

            for se_type in ("firm", "date", "two_way"):
                rows.append(
                    {
                        "feature": feature,
                        "spec": spec_name,
                        "se_type": se_type,
                        "beta_feature": beta_feature,
                        "tstat_feature": tstats[se_type]["feature"],
                        "beta_interaction": beta_interaction,
                        "tstat_interaction": tstats[se_type]["interaction"],
                        "r2": r2,
                        "n": n_obs,
                    }
                )

    return pd.DataFrame(rows)


def run_quintile_tests(analysis_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Compute quintile mean CAR, Q5-Q1 spread, and strict monotonicity per feature."""
    rows: list[dict] = []
    y_col = "car_2_20"

    for feature in feature_cols:
        df = analysis_df[[feature, y_col]].dropna().copy()
        n_obs = int(len(df))
        if n_obs < 5:
            continue

        try:
            df["quintile"] = pd.qcut(df[feature], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        except ValueError:
            continue

        means = df.groupby("quintile", observed=True)[y_col].mean()
        q_means = {}
        for q in (1, 2, 3, 4, 5):
            q_means[q] = float(means.loc[q]) if q in means.index else np.nan

        spread = np.nan
        if np.isfinite(q_means[1]) and np.isfinite(q_means[5]):
            spread = float(q_means[5] - q_means[1])

        monotonic = False
        quintile_values = [q_means[q] for q in (1, 2, 3, 4, 5)]
        if all(np.isfinite(v) for v in quintile_values):
            monotonic = bool(
                quintile_values[0] < quintile_values[1] < quintile_values[2] < quintile_values[3] < quintile_values[4]
            )

        rows.append(
            {
                "feature": feature,
                "n": n_obs,
                "q1_mean_car": q_means[1],
                "q2_mean_car": q_means[2],
                "q3_mean_car": q_means[3],
                "q4_mean_car": q_means[4],
                "q5_mean_car": q_means[5],
                "q5_minus_q1": spread,
                "strictly_increasing": monotonic,
            }
        )

    return pd.DataFrame(rows)


def prepare_analysis_dataframe(
    daily_features: pd.DataFrame,
    crsp_daily: pd.DataFrame,
    event_table: pd.DataFrame,
    sue_events: pd.DataFrame,
    volume_burst_threshold: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Create event-level analysis dataset and list of numeric feature columns."""
    event_features = build_event_features(
        daily_features=daily_features,
        volume_burst_threshold=volume_burst_threshold,
    )
    event_cars = compute_event_car(event_table=event_table, crsp_daily=crsp_daily)
    sue = sue_events[["event_id", "sue"]].copy()
    sue["sue"] = pd.to_numeric(sue["sue"], errors="coerce")

    analysis_df = event_cars.merge(event_features, on=["event_id", "permno"], how="inner")
    analysis_df = analysis_df.merge(sue, on="event_id", how="left")
    analysis_df["event_date"] = pd.to_datetime(analysis_df["event_date"])

    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"permno", "car_2_20", "car_num_days", "sue"}
    feature_cols = sorted([c for c in numeric_cols if c not in exclude])

    return analysis_df, feature_cols


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_out_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Feature-by-feature CAR regression tests")
    parser.add_argument("--daily-features", type=Path, default=root / "data/processed/daily_features.parquet")
    parser.add_argument("--crsp-daily", type=Path, default=root / "data/processed/crsp_daily.parquet")
    parser.add_argument("--event-table", type=Path, default=root / "data/processed/event_table.parquet")
    parser.add_argument("--sue-events", type=Path, default=root / "data/processed/sue_event_level.parquet")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir)
    parser.add_argument("--volume-burst-threshold", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    daily_features = pd.read_parquet(args.daily_features)
    crsp_daily = pd.read_parquet(args.crsp_daily)
    event_table = pd.read_parquet(args.event_table)
    sue_events = pd.read_parquet(args.sue_events)

    print("Preparing analysis dataset...")
    analysis_df, feature_cols = prepare_analysis_dataframe(
        daily_features=daily_features,
        crsp_daily=crsp_daily,
        event_table=event_table,
        sue_events=sue_events,
        volume_burst_threshold=args.volume_burst_threshold,
    )
    print(f"Events in analysis dataset: {len(analysis_df)}")
    print(f"Numeric features to test: {len(feature_cols)}")

    print("Running feature regressions...")
    regression_df = run_feature_regressions(analysis_df, feature_cols)

    print("Running quintile tests...")
    quintile_df = run_quintile_tests(analysis_df, feature_cols)

    regression_out = args.out_dir / "regression_summary.csv"
    quintile_out = args.out_dir / "quintile_summary.csv"
    regression_df.to_csv(regression_out, index=False)
    quintile_df.to_csv(quintile_out, index=False)

    print(f"\nSaved regression summary: {regression_out}")
    print(f"Saved quintile summary:   {quintile_out}")

    print("\nTop 10 features by |tstat_feature| (spec A, two-way clustered):")
    top10 = (
        regression_df[
            (regression_df["spec"] == "A_univariate")
            & (regression_df["se_type"] == "two_way")
        ]
        .dropna(subset=["tstat_feature"])
        .assign(abs_t=lambda d: d["tstat_feature"].abs())
        .sort_values("abs_t", ascending=False)
        .head(10)
    )

    if top10.empty:
        print("No valid two-way clustered t-stats available for ranking.")
    else:
        print(
            top10[["feature", "beta_feature", "tstat_feature", "r2", "n"]].to_string(
                index=False,
                float_format=lambda x: f"{x:0.4f}",
            )
        )


if __name__ == "__main__":
    main()

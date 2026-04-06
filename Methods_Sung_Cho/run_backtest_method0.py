"""
Run event backtest for Method 0 and save a markdown report.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest.event_backtester import EventBacktester

EVENT_FILE = ROOT / "Methods_Sung_Cho" / "0_event_features.parquet"
CRSP_FILE = ROOT / "data" / "processed" / "crsp_daily.parquet"
REPORT_FILE = ROOT / "Methods_Sung_Cho" / "0_backtest_result.md"
OUTPUT_DIR = ROOT / "data" / "results" / "method0"


def _load_event_df(path: Path) -> pd.DataFrame:
    events_raw = pd.read_parquet(path).copy()

    date_col = None
    for col in ["earnings_date", "rdq", "event_date", "date"]:
        if col in events_raw.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError("No date column found in event file.")

    if date_col != "earnings_date":
        events_raw = events_raw.rename(columns={date_col: "earnings_date"})

    if events_raw["event_id"].duplicated().any():
        events_raw = events_raw.drop_duplicates(subset="event_id", keep="first")

    event_df = events_raw[["event_id", "permno", "earnings_date", "pressure_score"]].copy()
    event_df["earnings_date"] = pd.to_datetime(event_df["earnings_date"])
    return event_df


def _load_returns(event_df: pd.DataFrame, path: Path) -> pd.DataFrame:
    crsp_raw = pd.read_parquet(path).copy()
    crsp_raw["date"] = pd.to_datetime(crsp_raw["date"])

    permnos = event_df["permno"].unique()
    min_date = event_df["earnings_date"].min() - pd.Timedelta(days=30)
    max_date = event_df["earnings_date"].max() + pd.Timedelta(days=30)

    returns_df = crsp_raw[
        crsp_raw["permno"].isin(permnos)
        & (crsp_raw["date"] >= min_date)
        & (crsp_raw["date"] <= max_date)
    ][["permno", "date", "ret"]].copy()

    return returns_df


def _summary_table_markdown(df: pd.DataFrame) -> str:
    out = ["| quintile | n_events | mean_return | std_return |", "|---:|---:|---:|---:|"]
    for _, row in df.iterrows():
        out.append(
            f"| {int(row['quintile'])} | {int(row['n_events'])} | {row['mean_return']:.6f} | {row['std_return']:.6f} |"
        )
    return "\n".join(out)


def _run_extreme_slices(
    event_df: pd.DataFrame, results: dict
) -> list[dict]:
    rows: list[dict] = []
    for pct in [10, 15, 20]:
        n_extreme = int(len(event_df) * pct / 100)
        top_events = event_df.nlargest(n_extreme, "pressure_score")["event_id"]
        bottom_events = event_df.nsmallest(n_extreme, "pressure_score")["event_id"]

        top_thr = event_df.nlargest(n_extreme, "pressure_score")["pressure_score"].min()
        bot_thr = event_df.nsmallest(n_extreme, "pressure_score")["pressure_score"].max()

        for entry in ["t=0", "t=+1"]:
            backtest_df = results[(entry, 20)]["backtest"]
            top_returns = backtest_df[backtest_df["event_id"].isin(top_events)]["cum_return"]
            bottom_returns = backtest_df[backtest_df["event_id"].isin(bottom_events)]["cum_return"]

            top_mean = float(top_returns.mean())
            bottom_mean = float(bottom_returns.mean())
            rows.append(
                {
                    "pct": pct,
                    "entry": entry,
                    "top_n": int(top_returns.notna().sum()),
                    "bottom_n": int(bottom_returns.notna().sum()),
                    "top_threshold": float(top_thr),
                    "bottom_threshold": float(bot_thr),
                    "top_return": top_mean,
                    "bottom_return": bottom_mean,
                    "spread": top_mean - bottom_mean,
                }
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_df = _load_event_df(EVENT_FILE)
    returns_df = _load_returns(event_df, CRSP_FILE)

    bt = EventBacktester(event_df, returns_df)
    results = bt.run_full_analysis(entry_timings=["t=0", "t=+1"], horizons=[5, 20])

    # Save event-level outputs for each configuration
    for (entry, horizon), res in results.items():
        fname = f"method0_backtest_{entry.replace('=', '').replace('+', 'plus')}_h{horizon}.parquet"
        res["backtest"].to_parquet(OUTPUT_DIR / fname, index=False)

    # Prepare markdown report
    lines: list[str] = []
    lines.append("# Method 0 Backtest Result")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- Event file: `{EVENT_FILE.relative_to(ROOT)}`")
    lines.append(f"- CRSP file: `{CRSP_FILE.relative_to(ROOT)}`")
    lines.append(f"- Events: {len(event_df):,}")
    lines.append(f"- Event date range: {event_df['earnings_date'].min().date()} to {event_df['earnings_date'].max().date()}")
    lines.append(
        f"- Pressure score range: [{event_df['pressure_score'].min():.3f}, {event_df['pressure_score'].max():.3f}]"
    )
    lines.append(f"- NaN pressure scores: {int(event_df['pressure_score'].isna().sum())}")
    lines.append("")
    lines.append("## Quintile Backtest (Q5 - Q1)")
    lines.append("")

    for (entry, horizon), res in results.items():
        summary = res["summary"]
        ls = res["long_short"]
        backtest_df = res["backtest"]
        valid = int(backtest_df["cum_return"].notna().sum())
        total = int(len(backtest_df))
        monotonic = summary["mean_return"].is_monotonic_increasing

        lines.append(f"### entry={entry}, horizon={horizon}d")
        lines.append(f"- Data availability: {valid}/{total} ({100 * (total - valid) / total:.2f}% loss)")
        lines.append(
            f"- Q1 return: {ls['q1_return']:.4f} ({ls['q1_return']*100:.2f}%), "
            f"Q5 return: {ls['q5_return']:.4f} ({ls['q5_return']*100:.2f}%), "
            f"Q5-Q1: {ls['long_short_return']:.4f} ({ls['long_short_return']*100:.2f}%)"
        )
        lines.append(f"- Monotonic Q1→Q5: {'Yes' if monotonic else 'No'}")
        lines.append("")
        lines.append(_summary_table_markdown(summary))
        lines.append("")

    extreme_rows = _run_extreme_slices(event_df, results)
    lines.append("## Extreme Bin Test (20d Horizon)")
    lines.append("")
    lines.append("| top/bottom pct | entry | top_n | bottom_n | top_return | bottom_return | spread |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    for r in extreme_rows:
        lines.append(
            f"| {r['pct']}% | {r['entry']} | {r['top_n']} | {r['bottom_n']} | "
            f"{r['top_return']:.4f} | {r['bottom_return']:.4f} | {r['spread']:.4f} |"
        )

    # Short interpretation block
    lines.append("")
    lines.append("## Interpretation")
    t0_20 = results[("t=0", 20)]["long_short"]["long_short_return"]
    t1_20 = results[("t=+1", 20)]["long_short"]["long_short_return"]
    t0_5 = results[("t=0", 5)]["long_short"]["long_short_return"]
    t1_5 = results[("t=+1", 5)]["long_short"]["long_short_return"]

    if t0_20 > 0 and t1_20 > 0:
        dir_text = "The main 20-day long-short spread is positive under both entry conventions."
    elif t0_20 < 0 and t1_20 < 0:
        dir_text = "The main 20-day long-short spread is negative under both entry conventions."
    else:
        dir_text = "The 20-day long-short spread is mixed across entry conventions."

    lines.append(f"- {dir_text}")
    lines.append(
        f"- 5-day spreads: t=0 `{t0_5:.4f}`, t=+1 `{t1_5:.4f}`. "
        f"20-day spreads: t=0 `{t0_20:.4f}`, t=+1 `{t1_20:.4f}`."
    )
    lines.append(
        "- If Q5-Q1 is persistently negative, the current pressure direction is likely inverted for return prediction."
    )
    lines.append(
        "- Saved event-level backtest outputs in `data/results/method0/` for follow-up diagnostics."
    )

    REPORT_FILE.write_text("\n".join(lines))
    print(f"Saved report: {REPORT_FILE}")
    print(f"Saved detailed outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

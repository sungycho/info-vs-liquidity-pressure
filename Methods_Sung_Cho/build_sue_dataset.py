"""
Build standalone SUE datasets for downstream modeling.

Outputs:
1) Firm-quarter SUE panel (gvkey, datadate, sue, ...)
2) Event-level SUE file keyed by event_id for easy merge into feature pipelines

SUE definition (seasonal random walk):
    delta_q = EPS_q - EPS_{q-4}
    sue     = delta_q / rolling_std(delta_q, 8 quarters)

Data source:
    WRDS Compustat Quarterly Fundamentals (comp.fundq.epsfxq)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_EVENT_TABLE = PROCESSED / "event_table.parquet"
DEFAULT_SUE_QUARTERLY = PROCESSED / "sue_firm_quarter.parquet"
DEFAULT_SUE_EVENT = PROCESSED / "sue_event_level.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch EPS from WRDS, compute SUE, and save standalone SUE datasets."
    )
    parser.add_argument("--event-table", type=Path, default=DEFAULT_EVENT_TABLE)
    parser.add_argument("--quarterly-output", type=Path, default=DEFAULT_SUE_QUARTERLY)
    parser.add_argument("--event-output", type=Path, default=DEFAULT_SUE_EVENT)
    parser.add_argument("--rolling-window", type=int, default=8)
    parser.add_argument("--min-history", type=int, default=4)
    parser.add_argument("--winsor-cap", type=float, default=10.0)
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=3,
        help="Extra years before min event datadate when querying fundq.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore existing quarterly file and fetch from WRDS again.",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Also save CSV versions next to parquet outputs.",
    )
    return parser.parse_args()


def _load_wrds_credentials() -> tuple[str, str]:
    username = os.environ.get("WRDS_USERNAME", "")
    password = os.environ.get("WRDS_PASSWORD", "")

    if username and password:
        return username, password

    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            keyval = line.strip()
            if not keyval or keyval.startswith("#"):
                continue
            m_user = re.match(r'WRDS_USERNAME[=:\s]+"?([^"\s]+)"?', keyval)
            if m_user:
                username = m_user.group(1)
            m_pass = re.match(r'WRDS_PASSWORD[=:\s]+"?([^"\s]+)"?', keyval)
            if m_pass:
                password = m_pass.group(1)

    if not username or not password:
        raise RuntimeError(
            "WRDS credentials not found. Set WRDS_USERNAME/WRDS_PASSWORD in env "
            "or .env at project root."
        )
    return username, password


def _wrds_connect():
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


def _compute_sue_from_eps(
    eps_df: pd.DataFrame, rolling_window: int, min_history: int, winsor_cap: float
) -> pd.DataFrame:
    req = {"gvkey", "datadate", "epsfxq"}
    missing = req - set(eps_df.columns)
    if missing:
        raise ValueError(f"EPS input missing columns: {sorted(missing)}")

    out = eps_df.copy()
    out["datadate"] = pd.to_datetime(out["datadate"])
    out = out.sort_values(["gvkey", "datadate"])

    out["delta_q"] = out.groupby("gvkey")["epsfxq"].diff(4)
    out["delta_std"] = out.groupby("gvkey")["delta_q"].transform(
        lambda s: s.rolling(rolling_window, min_periods=min_history).std()
    )
    out["sue"] = out["delta_q"] / out["delta_std"]
    out["sue"] = out["sue"].clip(-winsor_cap, winsor_cap)

    return out.dropna(subset=["sue"])[
        ["gvkey", "datadate", "epsfxq", "delta_q", "delta_std", "sue"]
    ]


def _fetch_eps_from_wrds(event_table: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
    gvkeys = sorted(event_table["gvkey"].dropna().astype(str).unique().tolist())
    if not gvkeys:
        raise ValueError("No gvkeys found in event table.")

    min_date = (event_table["datadate"].min() - pd.DateOffset(years=lookback_years)).date()
    max_date = event_table["datadate"].max().date()

    placeholders = ", ".join(["%s"] * len(gvkeys))
    query = f"""
        SELECT gvkey, datadate, epsfxq
        FROM comp.fundq
        WHERE gvkey IN ({placeholders})
          AND datadate BETWEEN %s AND %s
          AND epsfxq IS NOT NULL
        ORDER BY gvkey, datadate
    """
    params = [*gvkeys, min_date, max_date]

    print(
        f"Fetching EPS from WRDS for {len(gvkeys):,} gvkeys "
        f"between {min_date} and {max_date}..."
    )
    conn = _wrds_connect()
    try:
        eps = pd.read_sql(query, conn, params=params)
    finally:
        conn.close()

    eps["datadate"] = pd.to_datetime(eps["datadate"])
    print(f"Fetched {len(eps):,} firm-quarter rows from comp.fundq")
    return eps


def main() -> None:
    args = parse_args()
    event_table_path = args.event_table.resolve()
    quarterly_output = args.quarterly_output.resolve()
    event_output = args.event_output.resolve()

    if not event_table_path.exists():
        raise FileNotFoundError(f"Event table not found: {event_table_path}")

    event_table = pd.read_parquet(event_table_path)
    required = {"event_id", "gvkey", "datadate"}
    missing = required - set(event_table.columns)
    if missing:
        raise ValueError(f"Event table missing columns: {sorted(missing)}")

    event_table = event_table.copy()
    event_table["gvkey"] = event_table["gvkey"].astype(str)
    event_table["datadate"] = pd.to_datetime(event_table["datadate"])

    should_fetch = args.force_refresh or (not quarterly_output.exists())
    if not should_fetch:
        sue_quarterly = pd.read_parquet(quarterly_output)
        sue_quarterly["gvkey"] = sue_quarterly["gvkey"].astype(str)
        sue_quarterly["datadate"] = pd.to_datetime(sue_quarterly["datadate"])
        print(f"Using cached quarterly SUE: {quarterly_output}")

    if should_fetch:
        eps = _fetch_eps_from_wrds(event_table, lookback_years=args.lookback_years)
        sue_quarterly = _compute_sue_from_eps(
            eps_df=eps,
            rolling_window=args.rolling_window,
            min_history=args.min_history,
            winsor_cap=args.winsor_cap,
        )
        quarterly_output.parent.mkdir(parents=True, exist_ok=True)
        sue_quarterly.to_parquet(quarterly_output, index=False)
        print(f"Saved quarterly SUE to {quarterly_output}")

    event_sue = event_table.merge(
        sue_quarterly[["gvkey", "datadate", "sue"]],
        on=["gvkey", "datadate"],
        how="left",
    )

    matched = event_sue["sue"].notna().sum()
    total = len(event_sue)
    if matched == 0:
        raise RuntimeError("No SUE values matched onto events. Check event datadate/gvkey mapping.")

    sue_mean = event_sue["sue"].mean(skipna=True)
    sue_std = event_sue["sue"].std(ddof=1, skipna=True)
    event_sue["sue_std"] = (event_sue["sue"] - sue_mean) / sue_std

    event_output.parent.mkdir(parents=True, exist_ok=True)
    event_sue.to_parquet(event_output, index=False)
    print(f"Saved event-level SUE to {event_output}")
    print(f"SUE matched events: {matched:,} / {total:,} ({matched / total:.1%})")

    if args.save_csv:
        q_csv = quarterly_output.with_suffix(".csv")
        e_csv = event_output.with_suffix(".csv")
        sue_quarterly.to_csv(q_csv, index=False)
        event_sue.to_csv(e_csv, index=False)
        print(f"Saved CSV copies: {q_csv}, {e_csv}")


if __name__ == "__main__":
    main()

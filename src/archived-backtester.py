from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Backtester
# ============================================================

@dataclass(frozen=True)
class BacktestConfig:
    execution_lag_days: int = 1   # target[t] -> position[t+lag]
    cost_bps: float = 0.0         # linear cost on turnover: (bps/10000) * abs(trade)
    events_df: Optional[pd.DataFrame] = None  # optional: asset, entry_date, exit_date (+ optional event_id/group/quintile/side)


def run_backtest(
    targets_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    targets_df:
      index: MultiIndex(['date','asset']), col: 'target' (float weights)
    returns_df:
      index: MultiIndex(['date','asset']), col: 'ret' (float returns)
    outputs:
      positions_df: index ['date','asset'], col 'position'
      trades_df:    index ['date','asset'], cols ['trade','turnover_abs','cost']
      pnl_df:       index ['date'], cols ['gross_pnl','cost','net_pnl','turnover','cum_net_pnl']
      event_summary_df (optional)
    """

    targets = targets_df[["target"]].copy()
    rets = returns_df[["ret"]].copy()

    if targets.index.names != ["date", "asset"] or rets.index.names != ["date", "asset"]:
        raise ValueError("targets_df and returns_df must have MultiIndex names ['date','asset']")

    # normalize date dtype + sort
    targets = targets.reset_index()
    rets = rets.reset_index()
    targets["date"] = pd.to_datetime(targets["date"])
    rets["date"] = pd.to_datetime(rets["date"])
    targets = targets.set_index(["date", "asset"]).sort_index()
    rets = rets.set_index(["date", "asset"]).sort_index().dropna(subset=["ret"])

    # align targets to returns grid (missing targets -> 0)
    targets = targets.reindex(rets.index).fillna({"target": 0.0})

    # 1) targets -> positions (apply execution lag)
    positions_df = targets.rename(columns={"target": "position"}).copy()
    if cfg.execution_lag_days > 0:
        positions_df["position"] = positions_df.groupby(level="asset")["position"].shift(cfg.execution_lag_days)
    positions_df["position"] = positions_df["position"].fillna(0.0)

    # 2) positions -> trades (delta positions)
    prev_pos = positions_df.groupby(level="asset")["position"].shift(1).fillna(0.0)
    trades_df = positions_df.copy()
    trades_df["trade"] = trades_df["position"] - prev_pos
    trades_df["turnover_abs"] = trades_df["trade"].abs()
    trades_df["cost"] = (cfg.cost_bps / 10000.0) * trades_df["turnover_abs"]
    trades_df = trades_df[["trade", "turnover_abs", "cost"]]

    # 3) PnL accounting: gross_pnl[t] = sum_a position[a, t-1] * ret[a, t]
    pos_for_pnl = positions_df.groupby(level="asset")["position"].shift(1).fillna(0.0)
    pnl_panel = rets.copy()
    pnl_panel["pos_for_pnl"] = pos_for_pnl.reindex(rets.index).fillna(0.0)
    pnl_panel["gross_contrib"] = pnl_panel["pos_for_pnl"] * pnl_panel["ret"]

    gross_pnl = pnl_panel.groupby(level="date")["gross_contrib"].sum()
    daily_cost = trades_df.groupby(level="date")["cost"].sum()
    daily_turnover = trades_df.groupby(level="date")["turnover_abs"].sum()

    dates = gross_pnl.index.union(daily_cost.index).union(daily_turnover.index)
    pnl_df = pd.DataFrame(
        {
            "gross_pnl": gross_pnl.reindex(dates, fill_value=0.0),
            "cost": daily_cost.reindex(dates, fill_value=0.0),
            "turnover": daily_turnover.reindex(dates, fill_value=0.0),
        },
        index=dates,
    )
    pnl_df.index.name = "date"
    pnl_df["net_pnl"] = pnl_df["gross_pnl"] - pnl_df["cost"]
    pnl_df["cum_net_pnl"] = pnl_df["net_pnl"].cumsum()

    # 4) optional event summary
    event_summary_df = None
    if cfg.events_df is not None and len(cfg.events_df) > 0:
        event_summary_df = _event_summary(cfg.events_df, pnl_panel, trades_df)

    return positions_df, trades_df, pnl_df, event_summary_df


def _event_summary(
    events_df: pd.DataFrame,
    pnl_panel: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    events_df required: asset, entry_date, exit_date
    optional: event_id, group/quintile/side

    gross_return = sum gross_contrib over (entry_date..exit_date) for that asset
    cost         = sum cost over (entry_date..exit_date) for that asset
    net_return   = gross_return - cost
    """
    ev = events_df.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    ev["exit_date"] = pd.to_datetime(ev["exit_date"])

    if "event_id" not in ev.columns:
        ev = ev.reset_index(drop=True)
        ev["event_id"] = range(len(ev))
    ev = ev.set_index("event_id", drop=False)

    out_rows = []
    for _, r in ev.iterrows():
        asset = r["asset"]
        start = r["entry_date"]
        end = r["exit_date"]

        pnl_slice = pnl_panel.xs(asset, level="asset", drop_level=False)
        pnl_slice = pnl_slice.loc[
            (pnl_slice.index.get_level_values("date") >= start)
            & (pnl_slice.index.get_level_values("date") <= end)
        ]
        gross = float(pnl_slice["gross_contrib"].sum())

        tr_slice = trades_df.xs(asset, level="asset", drop_level=False)
        tr_slice = tr_slice.loc[
            (tr_slice.index.get_level_values("date") >= start)
            & (tr_slice.index.get_level_values("date") <= end)
        ]
        cost = float(tr_slice["cost"].sum())

        row = {
            "asset": asset,
            "entry_date": start,
            "exit_date": end,
            "gross_return": gross,
            "net_return": gross - cost,
            "cost": cost,
        }
        for c in ["group", "quintile", "side"]:
            if c in ev.columns:
                row[c] = r[c]
        out_rows.append((r["event_id"], row))

    out = pd.DataFrame.from_dict(dict(out_rows), orient="index")
    out.index.name = "event_id"
    return out


# ============================================================
# Utilities: returns from prices + random test data
# ============================================================

def make_returns_df_from_prices(
    targets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    price_col: str = "close",
    return_col: str = "ret",
) -> pd.DataFrame:
    """
    prices_df: index ['date','asset'] with column price_col
    output returns_df: index same as targets_df, column return_col
    """
    if targets_df.index.names != ["date", "asset"]:
        raise ValueError("targets_df index must be MultiIndex ['date','asset']")
    if prices_df.index.names != ["date", "asset"]:
        raise ValueError("prices_df index must be MultiIndex ['date','asset']")

    t = targets_df.reset_index()
    p = prices_df[[price_col]].reset_index()
    t["date"] = pd.to_datetime(t["date"])
    p["date"] = pd.to_datetime(p["date"])
    t = t.set_index(["date", "asset"]).sort_index()
    p = p.set_index(["date", "asset"]).sort_index()

    ret = p[price_col].groupby(level="asset").pct_change()
    return ret.to_frame(return_col).reindex(t.index)


def make_random_test_data(
    *,
    assets: list[str] = None,
    start: str = "2024-01-02",
    periods: int = 20,
    freq: str = "B",
    seed: int = 42,
    ret_sigma: float = 0.01,
    target_levels: tuple[float, ...] = (-0.02, 0.0, 0.02),
    target_probs: tuple[float, ...] = (0.25, 0.50, 0.25),
    market_neutralize: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (targets_df, returns_df) on the same (date,asset) grid
    """
    if assets is None:
        assets = ["AAPL", "MSFT", "GOOG", "AMZN"]

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq=freq)
    index = pd.MultiIndex.from_product([dates, assets], names=["date", "asset"])

    returns_df = pd.DataFrame({"ret": rng.normal(0.0, ret_sigma, size=len(index))}, index=index)

    raw_targets = rng.choice(list(target_levels), size=len(index), p=list(target_probs))
    targets_df = pd.DataFrame({"target": raw_targets}, index=index)

    if market_neutralize:
        def neutralize(g: pd.DataFrame) -> pd.DataFrame:
            s = g["target"].sum()
            if s != 0:
                g["target"] -= s / len(g)
            return g
        targets_df = targets_df.groupby(level="date", group_keys=False).apply(neutralize)

    return targets_df.sort_index(), returns_df.sort_index()


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # 1) generate random targets + returns to test engine mechanics
    targets_df, returns_df = make_random_test_data()

    # 2) generate returns from prices 
    prices_df = targets_df.copy() #replace with real market prices
    prices_df["close"] = 100 * np.exp(np.random.default_rng(0).normal(0, 0.01, size=len(prices_df))).cumprod()
    returns_df = make_returns_df_from_prices(targets_df, prices_df, price_col="close")

    # 3) run backtest
    cfg = BacktestConfig(execution_lag_days=1, cost_bps=5.0)
    positions_df, trades_df, pnl_df, event_summary_df = run_backtest(targets_df, returns_df, cfg)

    print("targets_df head:")
    print(targets_df.head(8))
    print("\nreturns_df head:")
    print(returns_df.head(8))
    print("\npnl_df head:")
    print(pnl_df.head())
    print("\nTotal net PnL:", pnl_df["net_pnl"].sum())
    print("Total turnover:", pnl_df["turnover"].sum())

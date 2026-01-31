import numpy as np
import pandas as pd


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with safety checks."""
    if len(x) < 3:
        return np.nan
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if sx == 0 or sy == 0 or np.isnan(sx) or np.isnan(sy):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _event_pi_ri(ofi_series: pd.Series) -> dict:
    """
    Compute PI and RI for a single event from an OFI time series (ordered by time).
    Returns dict with PI, RI and component values for debugging.
    """
    x = ofi_series.to_numpy(dtype=float)
    T = len(x)

    # Sign: treat exact zero as 0
    s = np.sign(x)  # -1, 0, +1
    a = np.abs(x)

    # --- PI components ---
    # (1) Level autocorr rho^(1) = corr(x_t, x_{t-1})
    rho1 = _safe_corr(x[1:], x[:-1])
    rho1_01 = (rho1 + 1.0) / 2.0 if not np.isnan(rho1) else np.nan

    # (2) Directional consistency: fraction of consecutive days with same nonzero sign
    nonzero_pair = (s[1:] != 0) & (s[:-1] != 0)
    if nonzero_pair.any():
        C = float(np.mean((s[1:][nonzero_pair] == s[:-1][nonzero_pair])))
    else:
        C = np.nan

    # (3) Drift ratio: |sum x| / sum |x| in [0,1] (if denom>0)
    denom = float(np.sum(a))
    D = float(np.abs(np.sum(x)) / denom) if denom > 0 else np.nan

    # --- RI components ---
    # First differences
    dx = np.diff(x)

    # (1) Reversal in differences: Gamma = (-corr(dx_t, dx_{t-1}) + 1)/2
    # Need at least 3 diffs (=> T>=4) to get corr(dx[1:], dx[:-1]) with len>=3
    corr_dx = _safe_corr(dx[1:], dx[:-1]) if len(dx) >= 3 else np.nan
    gamma = (-corr_dx) if not np.isnan(corr_dx) else np.nan
    Gamma = (gamma + 1.0) / 2.0 if not np.isnan(gamma) else np.nan

    # (2) Flip rate: fraction of consecutive days with different sign
    F = float(np.mean(s[1:] != s[:-1])) if T >= 2 else np.nan

    # (3) Alternation intensity: OFI magnitude occurring on flip transitions
    if T >= 2:
        flip = (s[1:] != s[:-1])
        denom2 = float(np.sum(a[1:]))
        A = float(np.sum(a[1:][flip]) / denom2) if denom2 > 0 else np.nan
    else:
        A = np.nan

    # Average available components (ignore NaNs)
    PI_components = np.array([rho1_01, C, D], dtype=float)
    RI_components = np.array([Gamma, F, A], dtype=float)

    PI = float(np.nanmean(PI_components)) if np.isfinite(np.nanmean(PI_components)) else np.nan
    RI = float(np.nanmean(RI_components)) if np.isfinite(np.nanmean(RI_components)) else np.nan

    return {
        "PI": PI,
        "RI": RI,
        # keep components for sanity-check diagnostics
        "pi_rho1_01": rho1_01,
        "pi_consistency": C,
        "pi_drift_ratio": D,
        "ri_gamma_01": Gamma,
        "ri_flip_rate": F,
        "ri_alt_intensity": A,
        "T": T,
    }


def compute_pi_ri(
    daily_df: pd.DataFrame,
    event_col: str = "event_id",
    date_col: str = "date",
    ofi_col: str = "ofi",
    min_days: int = 7,
) -> pd.DataFrame:
    """
    Compute Persistence Index (PI) and Reversibility Index (RI) per event.

    Parameters
    ----------
    daily_df : DataFrame
        Must contain columns [event_col, date_col, ofi_col].
        Multiple rows per event; will be sorted by date within event.
    min_days : int
        Events with fewer than this many observations are dropped.

    Returns
    -------
    DataFrame with one row per event_id:
        [event_id, PI, RI, component columns..., T]
    """
    df = daily_df[[event_col, date_col, ofi_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([event_col, date_col])

    # Filter by min days
    counts = df.groupby(event_col)[ofi_col].size()
    keep = counts[counts >= min_days].index
    df = df[df[event_col].isin(keep)]

    rows = []
    for eid, g in df.groupby(event_col, sort=False):
        res = _event_pi_ri(g[ofi_col])
        res[event_col] = eid
        rows.append(res)

    out = pd.DataFrame(rows)
    # Put event_id first
    cols = [event_col] + [c for c in out.columns if c != event_col]
    return out[cols]

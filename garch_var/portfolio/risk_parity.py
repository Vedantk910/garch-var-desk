"""Rolling inverse-volatility risk-parity overlay: SPY + TLT, monthly.

The desk's existing overlay is a single-asset 15% vol-target (equity/cash).
Risk parity spreads exposure across two risk-bearing assets (equities +
long-duration TLT) and sizes each month to equal *risk contribution*:
weight_i = inv(sig_i) / sum(inv(sig)), with sig_i the trailing 126-day
annualized vol of asset i.  The overlay then scales the basket to a target
annual portfolio vol and reports the FULL trade-off surface (Sharpe vs
drawdown vs turnover) - the "distributions, not point estimates" upgrade.

Costs: 10 bps on |delta weight| per asset per rebalance, the house
convention (vol_target.py, quant-ml-factor-backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .vol_target import annualised_vol, _metrics


def _combo_vol(eq_vol: float, dur_vol: float, cov: float) -> float:
    """Annualized vol of the inverse-vol blend, from its covariance pieces."""
    inv_e, inv_d = 1.0 / eq_vol, 1.0 / dur_vol
    raw_e = inv_e / (inv_e + inv_d)
    raw_d = 1.0 - raw_e
    var = (raw_e ** 2) * eq_vol ** 2 + (raw_d ** 2) * dur_vol ** 2 \
        + 2 * raw_e * raw_d * cov
    return float(np.sqrt(max(var, 1e-12)))


def risk_parity_portfolio(
    eq_returns: pd.Series,
    dur_returns: pd.Series,
    target_vol: float = 0.15,
    rebalance: str = "M",
    window: int = 126,
    cost_bps: float = 10.0,
) -> dict:
    """Monthly-rebalanced two-asset inverse-vol portfolio scaled to a target
    annual vol.  Returns PnL, per-asset weights, turnover and the metrics
    block, so a caller can build the trade-off surface.

    No look-ahead: the basket in force on day t was chosen at the close of
    the last rebalance day strictly before t, using only that day's rolling
    stats.  The very first days hold cash (0/0) until the first rebalance.
    A 10 bps cost is charged on |delta weight| in the rebalance period.
    """
    idx = eq_returns.index.union(dur_returns.index)
    eq = eq_returns.reindex(idx).fillna(0.0)
    dur = dur_returns.reindex(idx).fillna(0.0)
    eq_vol = annualised_vol(eq, window)
    dur_vol = annualised_vol(dur, window)
    # rolling annualized covariance (same window as the vols)
    cov = ((eq - eq.rolling(window, min_periods=60).mean())
           * (dur - dur.rolling(window, min_periods=60).mean())
           ).rolling(window, min_periods=60).mean() * 252

    start = eq_vol.first_valid_index()
    dates = idx[idx >= start]
    eq_vol_w = eq_vol.loc[dates]
    dur_vol_w = dur_vol.loc[dates]
    cov_w = cov.loc[dates]

    period = dates.to_period(rebalance)
    last_of_period = dates.to_series().groupby(period).max().values
    is_rebal = np.isin(dates, last_of_period)

    w_eq_cur, w_dur_cur = 0.0, 0.0
    total_turn = 0.0
    pnl = []
    w_eq_out, w_dur_out = {}, {}
    active = False

    for i, d in enumerate(dates):
        w_eq_eff = w_eq_cur if active else 0.0
        w_dur_eff = w_dur_cur if active else 0.0
        gross = w_eq_eff * eq.loc[d] + w_dur_eff * dur.loc[d]
        cost = 0.0
        if is_rebal[i] and pd.notna(eq_vol_w.loc[d]) and pd.notna(dur_vol_w.loc[d]):
            ve, vd = float(eq_vol_w.loc[d]), float(dur_vol_w.loc[d])
            cv = float(cov_w.loc[d]) if pd.notna(cov_w.loc[d]) else 0.0
            if ve > 0 and vd > 0:
                combo = _combo_vol(ve, vd, cv)
                scale = min(2.0, target_vol / combo)
                new_eq, new_dur = (1.0 / ve) / (1.0 / ve + 1.0 / vd) * scale, \
                    (1.0 / vd) / (1.0 / ve + 1.0 / vd) * scale
                total_turn += abs(new_eq - w_eq_cur) + abs(new_dur - w_dur_cur)
                w_eq_cur, w_dur_cur = new_eq, new_dur
                active = True
                cost = abs(new_eq - w_eq_eff) * cost_bps / 10_000.0 \
                    + abs(new_dur - w_dur_eff) * cost_bps / 10_000.0
        pnl.append(gross - cost)
        w_eq_out[d], w_dur_out[d] = w_eq_eff, w_dur_eff

    s = pd.Series(pnl, index=dates, name="pnl")
    m = _metrics(s)
    m["avg_turnover"] = float(total_turn / max(len(dates), 1))
    return {
        "pnl": s,
        "eq_weight": pd.Series(w_eq_out, index=dates, name="eq_weight"),
        "dur_weight": pd.Series(w_dur_out, index=dates, name="dur_weight"),
        "metrics": m,
    }


def vol_target_surface(
    eq_returns: pd.Series,
    dur_returns: pd.Series,
    target_vols=(0.10, 0.15, 0.20),
    cost_bps: float = 10.0,
) -> dict:
    """The trade-off surface: risk-parity at several target vols.

    One entry per target vol - Sharpe, max drawdown, turnover - so the desk
    reports the shape of the frontier, not a single point estimate.
    """
    out = {}
    for tv in target_vols:
        key = f"risk_parity_{int(round(tv * 100))}pct"
        out[key] = risk_parity_portfolio(
            eq_returns, dur_returns, target_vol=tv, cost_bps=cost_bps)
    return out

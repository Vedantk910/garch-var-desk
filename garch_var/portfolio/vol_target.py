"""Monthly vol-targeted portfolio with cost-aware turnover.

Targets an annual vol of `target_vol` (default 15%): each month the equity
weight is set to target_vol / realized_annual_vol, capped at 1.0, with the
remainder in T-bills (risk-free).  A 10 bps cost is charged on the turnover
(|w_new - w_old|) — consistent with the quant-ml-factor-backtest convention.

Comparison benchmark: equal 50/50 equity/T-bill, same cost model, so the
drawdown comparison isolates the vol-targeting effect from the cost effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def annualised_vol(returns: pd.Series, window: int = 126) -> pd.Series:
    """Trailing 6-month annualized vol (only past data)."""
    return returns.rolling(window, min_periods=60).std() * np.sqrt(252)


def _metrics(pnl: pd.Series) -> dict:
    equity = (1.0 + pnl).cumprod()
    n = len(pnl)
    ann_ret = equity.iloc[-1] ** (252.0 / n) - 1.0
    vol = pnl.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.0) / vol if vol > 0 else float("nan")
    dd = (equity / equity.cummax() - 1.0).min()
    return {"annualised_return": ann_ret, "volatility": vol,
            "sharpe": sharpe, "max_drawdown": float(-dd), "final_equity": float(equity.iloc[-1])}


def vol_target_portfolio(
    eq_returns: pd.Series,
    rf_returns: pd.Series,
    target_vol: float = 0.15,
    cost_bps: float = 10.0,
    target_equal_weight: float = 0.5,
) -> dict:
    """Simulate the vol-targeted strategy vs a static equal-weight benchmark.

    Both are rebalanced monthly; the vol-target strategy sizes equity each
    month, the benchmark holds a fixed target_equal_weight.

    Returns a dict with `pnl`, `weights` and `metrics` per arm.
    """
    idx = eq_returns.index.union(rf_returns.index)
    eq = eq_returns.reindex(idx).fillna(0.0)
    rf = rf_returns.reindex(idx).fillna(0.0)
    vol = annualised_vol(eq)

    start = vol.first_valid_index()
    dates = idx[idx >= start]
    eq_w, rf_w = eq.loc[dates], rf.loc[dates]
    vol_w = vol.loc[dates]

    # last observed date in each calendar month
    month_last = dates.to_series().groupby(dates.to_period("M")).max()
    is_month_end = pd.Series(
        np.isin(dates, month_last.values), index=dates, dtype=bool
    )

    out_wt, out_bench = {}, {}
    w_cur: float | None = None  # cash until the first rebalance
    bench_built = False
    pnl_wt, pnl_bench = [], []

    for d in dates:
        is_rebal = bool(is_month_end.loc[d])

        # PnL for day t is earned by the basket in force BEFORE t (no look-ahead);
        # a rebalance at the close of t sets the basket for t+1 and charges cost
        # in the same period (consistent with the factor-backtest convention).
        w_eff = 0.0 if w_cur is None else w_cur
        b_eff = target_equal_weight if bench_built else 0.0
        r = w_eff * eq_w.loc[d] + (1.0 - w_eff) * rf_w.loc[d]
        r_b = b_eff * eq_w.loc[d] + (1.0 - b_eff) * rf_w.loc[d]

        cost = cost_b = 0.0
        if is_rebal:
            tv = vol_w.loc[d]
            new_w = min(1.0, target_vol / tv) if pd.notna(tv) and tv > 0 else 1.0
            cost = abs(new_w - w_eff) * cost_bps / 10_000.0
            w_cur = new_w
            if not bench_built:
                cost_b = target_equal_weight * cost_bps / 10_000.0
                bench_built = True

        pnl_wt.append(r - cost)
        pnl_bench.append(r_b - cost_b)

        out_wt[d] = w_eff
        out_bench[d] = b_eff

    s_wt = pd.Series(pnl_wt, index=dates, name="pnl")
    s_bench = pd.Series(pnl_bench, index=dates, name="pnl")
    return {
        "pnl": s_wt,
        "benchmark_pnl": s_bench,
        "weights": pd.Series(out_wt, index=dates, name="eq_weight"),
        "metrics": _metrics(s_wt),
        "benchmark_metrics": _metrics(s_bench),
    }
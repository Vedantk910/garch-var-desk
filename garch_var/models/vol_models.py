"""GARCH(1,1) / EGARCH(1,1) volatility models on arch 7.x (new API).

All fits are deterministic MLE with fixed starting values.  Returns are
fitted in percent units (x100) for numerical conditioning, then rescaled
to log-return units so the desk works in plain decimal returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def _scaled(returns: pd.Series) -> np.ndarray:
    return 100.0 * returns.to_numpy()


def fit_garch(returns: pd.Series, dist: str = "studentst") -> dict:
    """Fit GARCH(1,1) with a HAR mean; return persistence + cond vol.

    dist: 'studentst' (handles heavy tails) or 'normal'.
    """
    x = _scaled(returns)
    res = arch_model(
        x, mean="har", lags=[1, 5, 21],
        vol="GARCH", p=1, q=1, dist=dist,
    ).fit(update_freq=50, disp="off")
    p = res.params
    alpha = float(p.get("alpha[1]", np.nan))
    beta = float(p.get("beta[1]", np.nan))
    h_t = pd.Series(res.conditional_volatility, index=returns.index) / 100.0
    return {
        "omega": float(p.get("omega", np.nan)),
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "near_integrated": (alpha + beta) > 0.99,
        "dist": dist,
        "dof": float(p["nu"]) if "nu" in p.index else None,
        "h_t": h_t,
        "aicc": res.aic,
        "results": res,
    }


def fit_egarch(returns: pd.Series) -> dict:
    """EGARCH(1,1): isolates the asymmetric (leverage) effect.

    gamma[1] < 0 and delta[1] > 0 is the classic leverage signature:
    bad-news shocks raise volatility more than good-news shocks of the
    same size.
    """
    x = _scaled(returns)
    res = arch_model(
        x, mean="har", lags=[1, 5, 21],
        vol="EGARCH", p=1, q=1, dist="normal",
    ).fit(update_freq=50, disp="off")
    p = res.params
    # arch 7.2 names the EGARCH coefficients: omega (log level),
    # alpha[1] (asymmetry, a.k.a. gamma), beta[1] (magnitude, a.k.a. delta)
    gamma = float(p.get("alpha[1]", np.nan))
    delta = float(p.get("beta[1]", np.nan))
    h_t = pd.Series(res.conditional_volatility, index=returns.index) / 100.0
    return {
        "gamma": gamma,
        "delta": delta,
        "leverage_negative": gamma < 0,
        "h_t": h_t,
        "results": res,
    }


def realized_vol(returns: pd.Series, window: int = 21) -> pd.Series:
    """Annualized rolling realized-vol benchmark (last N days only)."""
    return returns.rolling(window).std() * np.sqrt(252)


def fit_panel(returns: pd.DataFrame, dist: str = "studentst") -> dict:
    """Fit GARCH to every column; returns {ticker: fit dict or error}."""
    out = {}
    for col in returns.columns:
        try:
            out[col] = fit_garch(returns[col], dist=dist)
        except Exception as e:  # keep the desk alive if one fit fails
            out[col] = {"error": repr(e)}
    return out

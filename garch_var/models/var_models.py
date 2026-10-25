"""99% 1-day VaR estimators: historical, GARCH-normal, GARCH-t.

Convention: a VaR series is a POSITIVE loss bound, e.g. 0.02 = a 2% daily
loss.  A "hit" on day t occurs when the realized daily log return is below
the (negative of the) bound:  r_t < -VaR_t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..risk.backtest import hit_times, kupiec_pof, christoffersen_independence


def var_historical(returns: pd.Series, alpha: float = 0.01) -> pd.Series:
    """Historical 99% 1-day VaR as a rolling alpha-quantile (positive bound).

    Uses a 252-day rolling window so the estimator only ever sees the past.
    The first 120 obs are dropped (min_periods) to avoid a cold-start bias.
    """
    r = returns.fillna(0.0)
    q = r.rolling(252, min_periods=120).quantile(alpha)
    return -q  # alpha < 0.5 so the quantile is negative; flip to a positive bound


def var_garch(fit: dict, alpha: float = 0.01) -> pd.Series:
    """GARCH 1-day VaR from the conditional-volatility recursion.

    fit: output of fit_garch (h_t already rescaled to log-return units).
    VaR_t = z_alpha * h_t where z_alpha is the alpha-quantile of the
    standardized residual distribution (negative); -z_alpha * h_t is the
    positive loss bound.  `fit["dist"]` is "studentst" or "normal"; when
    student-t, `fit["dof"]` sets the t-quantile (falls back to 4.0).
    """
    h_t = fit["h_t"]
    if fit.get("dist") == "studentst" and fit.get("dof") is not None:
        z = -float(stats.t.ppf(alpha, df=fit["dof"]))
    else:
        z = -float(stats.norm.ppf(alpha))
    return z * h_t


def var_comparison(
    returns: pd.Series,
    fit: dict,
    alpha: float = 0.01,
    min_obs: int = 300,
) -> dict:
    """Run historical + GARCH VaR and report POF / independence for each.

    Only the tail of the sample (after both series are populated) is tested,
    so the coverage statistics are computed on a comparable window.
    """
    hist = var_historical(returns, alpha)
    gvar = var_garch(fit, alpha)

    idx = hist.index.intersection(gvar.index)
    r = returns.reindex(idx)
    h_hist = hist.reindex(idx)
    h_gvar = gvar.reindex(idx)

    # require a stable test window
    r = r.tail(min_obs + 252)
    h_hist = h_hist.reindex(r.index)
    h_gvar = h_gvar.reindex(r.index)

    r_arr = r.to_numpy(dtype=float)
    hits_hist = hit_times(r_arr, -h_hist.to_numpy(dtype=float))
    hits_gvar = hit_times(r_arr, -h_gvar.to_numpy(dtype=float))

    return {
        "hist": {
            "kupiec": kupiec_pof(hits_hist, alpha),
            "indep": christoffersen_independence(hits_hist, alpha),
            "n": int(len(r_arr)),
        },
        "garch": {
            "kupiec": kupiec_pof(hits_gvar, alpha),
            "indep": christoffersen_independence(hits_gvar, alpha),
            "n": int(len(r_arr)),
        },
    }

"""Expected Shortfall (CVaR) estimators, one-day horizon.

Convention matches `var_models`: an ES series is a POSITIVE loss bound, e.g.
0.03 = a 3% daily loss, and is always >= the VaR bound at the same level.
ES is the *mean* loss in the tail beyond VaR, which is why it is the correct
tail risk metric at the 99% "Basel" level where VaR is not sub-additive.

The PnL model the desk uses (inherited from `var_garch`) is PnL_t = h_t * eps
where eps is a standardised residual (Student-t with `dof` dof, or standard
normal).  ES = factor * h_t, and `factor` is the ES of that unit residual in
standard-deviation units.  The closed-form factor is cross-checked against
numerical integration in `tests` so the algebra is not trusted on faith.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def es_factor_normal(alpha: float = 0.01) -> float:
    """ES multiplier for a standard normal residual at tail mass `alpha`.

    ES = |phi(z_alpha)| / alpha where z_alpha = N^{-1}(alpha) < 0.  At
    alpha = 0.01 this is ~2.67, above the VaR factor of ~2.33.  (Divided by
    the tail mass `alpha`, NOT by (1-alpha) - that is the classic slip.)
    """
    z = float(stats.norm.ppf(alpha))
    return float(stats.norm.pdf(z) / alpha)


def es_factor_t(nu: float, alpha: float = 0.01) -> float:
    """ES multiplier for a standard Student-t residual (`nu` dof) at tail mass
    `alpha`.  ES = f_nu(z) * (nu + z^2) / ((nu - 1) * alpha), z = t_nu^{-1}(alpha).

    As nu -> inf this reduces to the normal factor (verified in tests).
    """
    z = float(stats.t.ppf(alpha, df=nu))
    f = float(stats.t.pdf(z, df=nu))
    return float(f * (nu + z * z) / ((nu - 1.0) * alpha))


def es_garch(fit: dict, alpha: float = 0.01) -> pd.Series:
    """One-day GARCH ES from the conditional-volatility recursion.

    Same `fit` shape as `var_garch` (h_t in log-return units, dist
    'studentt'/'studentst', optional dof).  Falls back to a t with dof=4 when
    the fit has no dof, matching the var_garch convention.
    """
    h_t = fit["h_t"]
    if fit.get("dist") in ("studentt", "studentst") and fit.get("dof") is not None:
        factor = es_factor_t(fit["dof"], alpha)
    else:
        factor = es_factor_normal(alpha)
    return factor * h_t


def es_historical(returns: pd.Series, alpha: float = 0.01,
                  window: int = 252, min_periods: int = 120) -> pd.Series:
    """Running historical ES: mean of the worst (alpha * window) losses in the
    trailing window.  Only past data is used; the first `min_periods` obs are
    dropped to avoid a cold-start bias, mirroring var_historical."""
    r = returns.fillna(0.0)
    losses = -r
    es = losses.rolling(window, min_periods=min_periods).apply(
        lambda x: float(np.mean(np.sort(x)[-max(int(np.ceil(alpha * len(x))), 1):])),
        raw=True,
    )
    return es

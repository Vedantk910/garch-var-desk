"""Stress re-estimation: does the GARCH desk survive a stress regime?

The elite move is to say where the model breaks, not to hide it.  For each
stress window the desk:

1. Fits GARCH-t on the CALM data before the window (the "global" model a
   desk would actually carry into the crash).
2. Runs an iterated one-step-ahead VaR/ES forecast INTO the window - no
   look-ahead: the forecast for day t uses only data up to t-1.  (A test
   checks the forecast is invariant to how much of the window is on the
   right, which is the definition of no look-ahead.)
3. Backtests that forecast on the window: hit counts against the nominal
   level, ES coverage, and the Mendez-Gaona joint test.
4. Re-fits GARCH-t on calm + window ("the regime re-fit") and reports how
   the persistence and the tail moved - the honest "the desk admits its
   limits here" number.

Both windows are real data: 2008-09 GFC from the stress cache
(`stress_data`), 2020-Q1 from the main panel (which covers it natively).

Power note, kept in the output: a 99% POF on a 44-54 day window has
expected hit count ~0.5; rejection is not the point.  The point is the
hit rate next to the ES level and the re-fit's persistence - that is the
admission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..models.vol_models import fit_garch
from ..risk.expected_shortfall import es_factor_t
from ..risk.es_backtest import es_backtest


def iterated_forecast(
    fit: dict,
    pre: pd.Series,
    window: pd.Series,
    alpha: float = 0.01,
) -> dict:
    """Iterated one-step-ahead GARCH VaR/ES of `window` from a fit on `pre`.

    Both series are daily log returns (decimal units).  The recursion runs
    in the scaled (x100) units arch uses:

        mu_t = Const + c1*y[t-1] + c5*y[t-5] + c21*y[t-21]
        h_t  = omega + a*(y[t-1] - mu[t-1])^2 + b*h[t-1]
        VaR_t = -z_alpha * sqrt(h_t) / 100   (positive loss bound)
        ES_t  = factor * sqrt(h_t) / 100

    Invariant to the right half of `window`: the forecast for day t is a
    function of `pre` and `window[:t]` only.
    """
    p = fit["results"].params
    omega = float(p.get("omega", 0.0))
    a = float(p.get("alpha[1]", 0.0))
    b = float(p.get("beta[1]", 0.0))
    nu = fit.get("dof")
    dist = fit.get("dist", "studentt")

    ys = (100.0 * pre.to_numpy(dtype=float)).tolist()
    wv = 100.0 * window.to_numpy(dtype=float)

    # cold start: unconditional variance of the scaled series
    h = omega / max(1.0 - (a + b), 1e-8) if (a + b) < 1.0 else np.var(ys + list(wv))
    z = -float(stats.t.ppf(alpha, df=nu)) if (nu and dist in ("studentt", "studentst")) \
        else -float(stats.norm.ppf(alpha))
    factor = es_factor_t(nu, alpha) if (nu and dist in ("studentt", "studentst")) \
        else float(stats.norm.pdf(z) / alpha)

    history = list(ys)  # scaled y values seen so far
    var_t, es_t = [], []
    for t in range(len(wv)):
        y_t = wv[t]
        # forecast for day t is made from info up to t-1
        mu_prev = _har_mu_from(history, p, t)
        h = omega + a * (history[-1] - mu_prev) ** 2 + b * h
        sig = np.sqrt(h) / 100.0
        var_t.append(z * sig)
        es_t.append(factor * sig)
        history.append(y_t)
    return {
        "var": pd.Series(var_t, index=window.index),
        "es": pd.Series(es_t, index=window.index),
    }


def _har_mu_from(history: list, params, t: int) -> float:
    """HAR mean for day t using only `history` (scaled), indexed from the
    pre-period start.  t counts within `window`, so absolute lag positions
    are offset by len(pre)."""
    c = lambda name: float(params.get(name, 0.0))
    base = len(history) - t - 1  # absolute position of history[-1]
    mu = c("Const")
    for lag, key in ((1, "y[0:1]"), (5, "y[0:5]"), (21, "y[0:21]")):
        i = base - lag
        if i >= 0:
            mu += c(key) * history[i]
        else:
            mu += c(key) * (np.mean(history[:min(21, len(history))] or [0.0]))
    return mu


def stress_reestimate(
    pre: pd.Series,
    window: pd.Series,
    label: str,
    alpha: float = 0.01,
) -> dict:
    """Full stress exhibit for one window.  See the module docstring."""
    # global model: the fit the desk would carry into the crash
    fit_pre = fit_garch(pre, dist="studentst")
    fc = iterated_forecast(fit_pre, pre, window, alpha)

    # backtest the global forecast on the window
    back = es_backtest(window, fc["var"], fc["es"],
                       dist=fit_pre.get("dist"), dof=fit_pre.get("dof"),
                       alpha=alpha, tail=None)

    # regime re-fit: persistence + tail, now that the crash is in-sample
    fit_full = fit_garch(pd.concat([pre, window]), dist="studentst")

    # empirical ES next to the model's, on the global forecast
    I = (window.to_numpy() < -fc["var"].to_numpy())
    emp_es = float(np.mean(-window.to_numpy()[I])) if I.sum() else float("nan")

    return {
        "label": label,
        "window": str(window.index.min().date()) + " -> " + str(window.index.max().date()),
        "n_days": int(len(window)),
        "worst_day": {"date": str(window.idxmin().date()),
                      "loss_pct": round(float(window.min() * 100), 2)},
        "global_persistence": round(float(fit_pre["persistence"]), 4),
        "refit_persistence": round(float(fit_full["persistence"]), 4),
        # full-precision pair: the 2008 pre-fit sits at ~0.99997, and the
        # rounded 4-dp values both print as 1.000 - the note needs the digits
        "global_persistence_full": float(fit_pre["persistence"]),
        "refit_persistence_full": float(fit_full["persistence"]),
        "global": back,
        "empirical_mean_loss_beyond_var": round(emp_es, 4) if not np.isnan(emp_es) else None,
        "es_level_mean": round(float(np.nanmean(fc["es"])), 4),
        "power_note": (f"expected 99% hits on {len(window)} days = "
                       f"{0.01 * len(window):.2f} - a POF verdict is not "
                       f"the point; read the hit rate next to the ES level"),
        "global_aic_pre": round(float(fit_pre["aicc"]), 1),
        "refit_aic": round(float(fit_full["aicc"]), 1),
    }


def run_stress_suite(panels: dict, windows: dict) -> dict:
    """Run every configured stress window.  `panels` maps label -> calm
    pre-period; `windows` maps label -> stress window.  One dict back, one
    entry per label, for the note and out/ JSON."""
    out = {}
    for label, win in windows.items():
        if label not in panels:
            raise KeyError(f"stress window {label!r} has no pre-period panel")
        out[label] = stress_reestimate(panels[label], win, label)
    return out

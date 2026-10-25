"""VaR backtesting: hits, Kupiec POF, Christoffersen independence.

All tests are one-sided at the 5% level against the nominal 99% VaR.
The desk reads a non-rejected POF + non-rejected independence as "coverage
is honest".  A rejected POF means the VaR is too tight or too loose.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def hit_times(returns: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Boolean array: True where a 1-day VaR violation occurred.

    returns: daily log returns, aligned with var.
    var:      negative number (e.g. -0.02 = 99% VaR loss bound).
    """
    var = np.asarray(var, dtype=float)
    r = np.asarray(returns, dtype=float)
    return (r < var).astype(int)


def kupiec_pof(hits: np.ndarray, alpha: float = 0.01) -> dict:
    """Kupiec proportion-of-failures test.

    H0: unconditional hit probability == alpha.
    Statistic z = (phat - alpha) / sqrt(alpha(1-alpha)/T),
    asymptotically N(0,1).  Report |z| vs N(0.975) critical value.
    """
    T = len(hits)
    phat = hits.sum() / T
    se = np.sqrt(alpha * (1 - alpha) / T)
    z = (phat - alpha) / se
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "T": int(T),
        "hits": int(hits.sum()),
        "hit_rate": float(phat),
        "alpha": alpha,
        "z": float(z),
        "p_value": pval,
        "rejected_5pct": bool(pval < 0.05),
    }


def christoffersen_independence(hits: np.ndarray, alpha: float = 0.01) -> dict:
    """Christoffersen: LR-indep test for independence of violations.

    2x2 transition matrix H_0->H_0, H_0->H_1, H_1->H_0, H_1->H_1 where
    H_0 = no hit, H_1 = hit.  Under the null (Poisson hits at rate alpha),
    the probability of a hit yesterday does not inform the probability
    today.  The statistic is a chi-squared on the 4 transition counts.
    """
    T = len(hits)
    if T < 2:
        return {"error": "need >= 2 obs"}
    h = np.asarray(hits, dtype=int)
    p_hit = h[:-1]
    q_hit = h[1:]
    # transition counts (.sum() then int() — int() on a bool array raises)
    hh = int(((p_hit == 1) & (q_hit == 1)).sum())
    h0 = int(((p_hit == 1) & (q_hit == 0)).sum())
    hh0 = int(((p_hit == 0) & (q_hit == 1)).sum())
    h00 = int(((p_hit == 0) & (q_hit == 0)).sum())
    # probabilities
    p_hit_y = p_hit.mean()
    p_hit_x_given_y1 = hh / max(hh + h0, 1)
    p_hit_x_given_y0 = hh0 / max(hh0 + h00, 1)
    # expected counts under independence
    Tm1 = T - 1
    exp11 = p_hit_y * p_hit_x_given_y1 * Tm1
    exp10 = p_hit_y * (1 - p_hit_x_given_y1) * Tm1
    exp01 = (1 - p_hit_y) * p_hit_x_given_y0 * Tm1
    exp00 = (1 - p_hit_y) * (1 - p_hit_x_given_y0) * Tm1
    obs = np.array([hh, h0, hh0, h00], dtype=float)
    exp = np.array([exp11, exp10, exp01, exp00], dtype=float)
    mask = exp > 0
    chi2 = ((obs[mask] - exp[mask]) / np.sqrt(exp[mask])) ** 2
    lr_ind = chi2.sum()
    pval = 1 - stats.chi2.cdf(lr_ind, df=1)
    return {
        "chi2": float(lr_ind),
        "p_value": pval,
        "rejected_5pct": pval < 0.05,
    }


def spillover_test(
    hits_short: np.ndarray, hits_long: np.ndarray, alpha: float = 0.01
) -> dict:
    """Spillover: does a violation in the short window predict the long?"""
    T = min(len(hits_short), len(hits_long))
    s = np.asarray(hits_short[:T], dtype=int)
    l = np.asarray(hits_long[:T], dtype=int)
    Tm1 = T - 1
    hh = int(((s[1:] == 1) & (l[1:] == 1)).sum())
    h0 = int(((s[1:] == 1) & (l[1:] == 0)).sum())
    hh0 = int(((s[1:] == 0) & (l[1:] == 1)).sum())
    h00 = int(((s[1:] == 0) & (l[1:] == 0)).sum())
    p_s = s[1:].mean()
    p_l_given_s1 = hh / max(hh + h0, 1)
    p_l_given_s0 = hh0 / max(hh0 + h00, 1)
    exp = np.array([
        p_s * p_l_given_s1 * Tm1,
        p_s * (1 - p_l_given_s1) * Tm1,
        (1 - p_s) * p_l_given_s0 * Tm1,
        (1 - p_s) * (1 - p_l_given_s0) * Tm1,
    ])
    obs = np.array([hh, h0, hh0, h00], dtype=float)
    m = exp > 0
    chi2 = ((obs[m] - exp[m]) / np.sqrt(exp[m])) ** 2
    lr = float(chi2.sum())
    pval = float(1 - stats.chi2.cdf(lr, df=1))
    return {"chi2": lr, "p_value": pval, "rejected_5pct": bool(pval < 0.05)}


def summary_table(hits: dict[str, np.ndarray], alpha: float = 0.01) -> dict:
    """Run POF + independence + spillover for a dict of hit arrays."""
    out = {}
    names = list(hits.keys())
    for name in names:
        h = hits[name]
        kup = kupiec_pof(h, alpha)
        chi = christoffersen_independence(h, alpha)
        out[name] = {"kupiec": kup, "indep": chi}
    # pairwise spillover on consecutive keys
    for i in range(len(names) - 1):
        out.setdefault("spillover", {})[f"{names[i]}_then_{names[i+1]}"] = \
            spillover_test(hits[names[i]], hits[names[i + 1]], alpha)
    return out

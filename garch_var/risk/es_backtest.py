"""ES backtests: ES coverage + the Mendez-Gaona joint VaR/ES likelihood-ratio.

ES is a *level* on the loss distribution (the mean loss in the tail), so it
has no violation rate of its own the way VaR does.  Two honest checks:

  1. ES coverage.  On a correctly calibrated model the *violation* rate of
     the ES level is beta = P(loss > ES), NOT the tail mass alpha: for a
     normal 99% model VaR = 2.326 sigma and ES = 2.665 sigma, so beta ~=
     0.0038.  The Kupiec POF run on the ES indicator against beta is the
     right calibration null (reused from `kupiec_pof`), and the
     Christoffersen independence test checks the hits do not cluster.

  2. The Mendez-Gaona joint test.  Form the nested pair
       I_t = 1{ loss > VaR_t },  J_t = 1{ loss > ES_t }.
     Because ES_t >= VaR_t, J = 1 implies I = 1, so the pair reaches three
     states: (1,1) beyond ES; (1,0) the middle band VaR < loss <= ES;
     (0,0) clean.  Under correct estimation the state probabilities are
     fully specified by the two tail masses,
       P(1,1) = beta,  P(1,0) = alpha - beta,  P(0,0) = 1 - alpha,
     and a likelihood-ratio chi-squared on the observed state counts
     (df = 2, a fully specified null) is the Mendez-Gaona backtest.  It
     fails when either level is miscalibrated OR the middle band of losses
     is mis-placed - the information a VaR-only POF misses.

The state probabilities are the *model-implied* constants (alpha, beta
known a priori, not refit on the window), so the test stays meaningful on
a stress window whose empirical tail is far from nominal - that is exactly
where the desk must say whether the model admitted its limits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import kupiec_pof, christoffersen_independence
from .expected_shortfall import es_factor_normal, es_factor_t


def es_violation_prob(dist: str, dof, alpha: float = 0.01) -> float:
    """Model-implied beta = P(loss > ES_alpha): the ES *violation* rate.

    The ES factor (ES/sigma) sits above the VaR factor, so beta < alpha:
    for normal 99% it is P(N > 2.665) ~= 0.0038.  Passing alpha through to
    a POF would test the wrong null - that is the classic ES slip.
    """
    if dist in ("studentt", "studentst") and dof is not None:
        factor = es_factor_t(dof, alpha)
        return float(1.0 - stats.t.cdf(factor, df=dof))
    factor = es_factor_normal(alpha)
    return float(1.0 - stats.norm.cdf(factor))


def mendez_gaona(
    returns: np.ndarray,
    var: np.ndarray,
    es: np.ndarray,
    alpha: float = 0.01,
    beta: float | None = None,
    alpha_level: float = 0.05,
) -> dict:
    """Mendez-Gaona likelihood-ratio chi-squared on the (VaR, ES) indicators.

    `returns`: daily log returns; `var` / `es`: aligned POSITIVE loss bounds
    (ES_t >= VaR_t, the nested configuration of this desk).  `beta` is the
    model-implied P(loss > ES); leave it None to derive it from `alpha` for
    a standard normal residual, or pass the GARCH-fit value from
    `es_violation_prob`.

    The G^2 statistic is 2 * sum O_i ln(O_i / E_i) over the three
    reachable states; it is non-negative and asymptotically chi-square
    with 2 df.  (When beta == alpha the middle band has zero width in
    probability and the test drops to the ES coverage POF, df = 1.)
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(var, dtype=float)
    e = np.asarray(es, dtype=float)
    if beta is None:
        beta = es_violation_prob("normal", None, alpha)
    if not (0.0 < beta <= alpha < 1.0):
        raise ValueError(f"need 0 < beta <= alpha < 1 (got {beta}, {alpha})")
    nested_ok = bool(np.all(e >= v - 1e-15))
    if not nested_ok:
        # not a nested configuration; state it rather than silently run the
        # three-state test on data where state (0,1) is reachable
        raise ValueError("ES bound below VaR bound somewhere; nested test undefined")

    I = (r < -v).astype(int)
    J = (r < -e).astype(int)
    c_beyond, c_band, c_clean = int(J.sum()), int((I & ~J).sum()), int(((I | J) == 0).sum())
    obs = np.array([c_beyond, c_band, c_clean], dtype=float)
    p = np.array([beta, alpha - beta, 1.0 - alpha])
    expected = p * len(r)

    keep = p > 0
    # filter all three state arrays by `keep` together: when beta == alpha
    # the middle band has zero mass and drops out, and `expected` must be
    # filtered too - indexing it unfiltered mis-aligns on the df=1 case
    p, obs, expected = p[keep], obs[keep], expected[keep]
    df = max(int(keep.sum()) - 1, 1)
    nonzero = obs > 0
    g2 = float(2.0 * np.sum(obs[nonzero] * np.log(obs[nonzero] / expected[nonzero])))
    pval = float(1.0 - stats.chi2.cdf(g2, df=df))
    return {
        "n": int(len(r)),
        "states": int(keep.sum()),
        "state_probabilities": p.tolist(),
        "counts_beyond_es": c_beyond if keep[0] else None,
        "counts_middle_band": c_band if keep[1] else None,
        "counts_clean": c_clean if keep[2] else None,
        "alpha": alpha,
        "beta": float(beta),
        "chi2": g2,
        "df": df,
        "p_value": pval,
        "rejected_at_5pct": bool(pval < alpha_level),
    }


def es_backtest(
    returns,
    var,
    es,
    dist: str = "studentt",
    dof=None,
    alpha: float = 0.01,
    tail: int = 552,
) -> dict:
    """The three ES checks over the shared out-of-sample window.

    Mirrors `var_comparison`: align on the index intersection and keep the
    last `tail` observations.  `var` / `es` are Series (GARCH) or scalars
    (flat historical levels).
    """
    r = pd.Series(returns)
    idx = r.index
    r = r.reindex(idx)
    v = var.reindex(idx) if isinstance(var, pd.Series) else float(var)
    e = es.reindex(idx) if isinstance(es, pd.Series) else float(es)
    if tail and tail < len(r):
        r, v, e = r.tail(tail), (v.tail(tail) if isinstance(v, pd.Series) else v), \
                  (e.tail(tail) if isinstance(e, pd.Series) else e)
    r = r.to_numpy(dtype=float)
    v = v.to_numpy(dtype=float) if isinstance(v, pd.Series) else np.full(len(r), float(v))
    e = e.to_numpy(dtype=float) if isinstance(e, pd.Series) else np.full(len(r), float(e))

    beta = es_violation_prob(dist, dof, alpha)
    es_hits = (r < -e).astype(int)
    return {
        "n": int(len(r)),
        "beta_model_implied": float(beta),
        "es_coverage": kupiec_pof(es_hits, beta),
        "es_indep": christoffersen_independence(es_hits, beta),
        "mendez_gaona": mendez_gaona(r, v, e, alpha=alpha, beta=beta),
        # empirical ES next to the model level, for the note
        "mean_loss_beyond_var": _mean_loss_beyond(r, v),
        "es_level_mean": float(np.nanmean(e)),
    }


def _mean_loss_beyond(r: np.ndarray, var: np.ndarray) -> float:
    """Average loss on the days VaR was violated (the empirical ES)."""
    I = r < -var
    if I.sum() == 0:
        return float("nan")
    return float(np.mean(-r[I]))

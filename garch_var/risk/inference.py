"""Inference for the portfolio arms: bootstrap Sharpe CIs + deflated Sharpe.

The elite gap this fills: every arm in the desk is a point estimate
("Sharpe 0.68").  Two upgrades, both offline and seeded:

* **Block bootstrap CI** on the arm's annualized Sharpe.  Daily PnL is
  auto-correlated, so an i.i.d. bootstrap understates the variance of the
  Sharpe; the block version resamples contiguous chunks.  The edge is
  deterministic given the seed, so the CI in the note is reproducible.

* **Deflated Sharpe (Bailey & Lopez de Prado)**: the expected Sharpe of
  the *best* of M independent strategies tried on the same data, when M >
  1.  The desk tried three risk models (GARCH, EGARCH, GARCH-t) and three
  overlay arms on one panel, so a reported Sharpe above zero is not
  automatically signal - the DSR corrects the null.  Report the DSR p
  value next to the raw Sharpe in the note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _annualised_sharpe(pnl: pd.Series) -> float:
    n = len(pnl)
    if n < 2:
        return float("nan")
    ann = pnl.mean() * 252
    sd = pnl.std() * np.sqrt(252)
    return ann / sd if sd > 0 else float("nan")


def block_bootstrap_ci(
    pnl: pd.Series,
    n_blocks: int = 5000,
    block_len: int = 21,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap CI on the annualized Sharpe of a daily PnL series.

    Resamples `n_blocks` blocks of `block_len` consecutive days (with
    replacement on the block grid) and takes the empirical quantiles of
    the resulting Sharpe distribution.  Deterministic for a fixed seed.
    """
    x = pnl.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2 * block_len:
        return {"error": f"need >= {2 * block_len} obs for block bootstrap"}
    rng = np.random.default_rng(seed)
    n_full = n // block_len
    starts = rng.integers(0, n_full, size=(n_blocks, n_full))
    resampled = np.concatenate([x[s * block_len:(s + 1) * block_len]
                                for s in starts.flat])
    # each resample has n_full * block_len days; compute Sharpe per resample
    n_sub = n_full * block_len
    sharpes = []
    for i in range(n_blocks):
        chunk = resampled[i * n_sub:(i + 1) * n_sub]
        m = chunk.mean() * 252
        sd = chunk.std(ddof=1) * np.sqrt(252)
        sharpes.append(m / sd if sd > 0 else np.nan)
    sharpes = np.array(sharpes)
    lo = float(np.nanpercentile(sharpes, 100 * (1 - ci) / 2))
    hi = float(np.nanpercentile(sharpes, 100 * (1 + ci) / 2))
    return {
        "point": _annualised_sharpe(pnl),
        "ci_low": lo,
        "ci_high": hi,
        "ci": ci,
        "n_blocks": n_blocks,
        "block_len": block_len,
        "seed": seed,
        "excludes_zero": bool(lo > 0),
    }


def deflated_sharpe(
    pnl: pd.Series,
    n_trials: int = 3,
    sr0: float = 0.0,
) -> dict:
    """Deflated Sharpe Ratio of the *best* of `n_trials` strategies.

    Implements the Bailey & Lopez de Prado null: under H0 the maximum
    observed Sharpe of M independent trials has a non-zero mean, so the
    DSR asks how far the observed Sharpe sits above that expected
    best-of-M.  `sr0` is an optional floor (0 = no floor).

    The formula, with T daily obs, gamma3/gamma4 the skew/kurtosis of the
    (mean-subtracted) PnL, and SR the PER-PERIOD Sharpe (mean/std of one
    day's PnL, NOT annualised):

        E[max] ~ sqrt( (2 ln M) / (T - 1) )      (M > 1, extreme-value, per-period)
        DSR  =  (SR - E[max]) * sqrt(T - 1)
                / sqrt( 1 - g3*SR + (g4 - 1)/4 * SR^2 )

    The null E[max] is in per-period units, so SR must be too - comparing an
    annualised Sharpe against it understates the multiple-trials haircut by
    about sqrt(252), which is the classic implementation slip.  The
    annualised Sharpe is reported next to the statistic, in per-period
    units.  p-value under N(0,1).  `n_trials` should count every strategy
    the panel was tried on - that is the honest input, not 1.
    """
    x = pnl.dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 30:
        return {"error": "need >= 30 obs for deflated Sharpe"}
    xc = x - x.mean()
    sd = xc.std(ddof=1)
    if sd <= 0:
        return {"error": "zero-variance PnL"}
    # per-period Sharpe of the RAW series - xc is mean-subtracted, so
    # xc.mean() is identically zero and would silently zero the statistic
    sr_pp = float(x.mean() / sd)
    m = max(int(n_trials), 2)
    g3 = float(np.mean((xc / sd) ** 3))
    g4 = float(np.mean((xc / sd) ** 4))
    # Bailey-LdP expected best-of-M under H0, in per-period units
    sr_zero = float(np.sqrt(2.0 * np.log(m) / (T - 1)))
    var_term = 1.0 - g3 * sr_pp + 0.25 * (g4 - 1.0) * sr_pp * sr_pp
    if var_term <= 0:
        return {"error": "negative variance term - Sharpe too high for this formula"}
    stat = (sr_pp - sr_zero - sr0) * np.sqrt(T - 1) / np.sqrt(var_term)
    pval = float(1.0 - stats.norm.cdf(stat))
    return {
        "sharpe_per_period": sr_pp,
        "sharpe": float(_annualised_sharpe(pnl)),
        "n_trials": m,
        "sr_zero": sr_zero,
        "skew": g3,
        "kurtosis": g4,
        "statistic": float(stat),
        "p_value": pval,
        "significant_at_5pct": bool(pval < 0.05),
        "note": "sr_zero = expected per-period best-of-M under H0; trials count every strategy tried on the panel",
    }


def arm_inference(
    pnl: pd.Series,
    n_trials: int = 3,
    seed: int = 0,
) -> dict:
    """Both inferences for one arm, as one dict for the note / out JSON."""
    return {
        "bootstrap_sharpe_ci": block_bootstrap_ci(pnl, seed=seed),
        "deflated_sharpe": deflated_sharpe(pnl, n_trials=n_trials),
    }

import numpy as np
import pandas as pd

from garch_var.risk.backtest import (
    hit_times,
    kupiec_pof,
    christoffersen_independence,
    spillover_test,
)
from garch_var.models.var_models import var_historical, var_garch
from garch_var.portfolio.vol_target import vol_target_portfolio, annualised_vol


def _synthetic_returns(seed: int = 1, n: int = 2000) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    # inject a vol cluster in 2020-Q2 (like the real crash)
    mask = (idx >= "2020-03-15") & (idx <= "2020-04-30")
    r[mask] = rng.normal(-0.005, 0.03, int(mask.sum()))
    return r


def test_hit_times_detects_violations():
    r = np.array([-0.01, 0.005, -0.05, 0.002, -0.001])
    var = np.array([-0.02] * 5)
    hits = hit_times(r, var)
    assert hits.tolist() == [0, 0, 1, 0, 0]


def test_kupiec_pof_on_expected_rate():
    # 1000 obs, 1% nominal, exactly 10 hits -> should NOT reject
    rng = np.random.default_rng(0)
    hits = np.zeros(1000, dtype=int)
    hits[:10] = 1
    res = kupiec_pof(hits, alpha=0.01)
    assert res["rejected_5pct"] is False
    assert abs(res["hit_rate"] - 0.01) < 1e-9


def test_kupiec_pof_rejects_too_many_hits():
    hits = np.zeros(1000, dtype=int)
    hits[:100] = 1  # 10% hit rate on a 1% VaR must fail POF
    res = kupiec_pof(hits, alpha=0.01)
    assert res["rejected_5pct"] is True


def test_christoffersen_runs_and_bounded():
    rng = np.random.default_rng(1)
    hits = rng.binomial(1, 0.02, 500)
    res = christoffersen_independence(hits)
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["chi2"] >= 0.0


def test_spillover_two_independent_series():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.02, 800)
    b = rng.binomial(1, 0.02, 800)
    res = spillover_test(a, b)
    assert 0.0 <= res["p_value"] <= 1.0


def test_var_historical_positive_and_bounded():
    r = _synthetic_returns()
    v = var_historical(r, alpha=0.01)
    valid = v.dropna()
    assert (valid > 0).all()  # positive loss bound
    assert (valid < 0.10).all()  # sanity: never above 10% daily


def test_var_garch_positive_on_synthetic_fit():
    # hand-built fit dict matching the new fit_garch output shape
    r = _synthetic_returns()
    idx = r.index
    fit = {
        "h_t": pd.Series(np.full(len(idx), 0.012), index=idx),
        "dist": "normal",
        "dof": None,
        "persistence": 0.95,
    }
    v = var_garch(fit, alpha=0.01)
    assert (v > 0).all()
    # normal dist: z = -norm.ppf(0.01) = 2.326
    assert abs(v.iloc[0] - 0.012 * 2.3263) < 1e-4


def test_vol_target_weight_never_exceeds_one():
    r = _synthetic_returns()
    rf = pd.Series(0.03 / 252.0, index=r.index)
    out = vol_target_portfolio(r, rf, target_vol=0.15)
    w = out["weights"]
    assert (w >= 0).all() and (w <= 1.0).all()
    assert out["metrics"]["max_drawdown"] > 0
    # vol-target should not blow past 15% target by much
    assert out["metrics"]["volatility"] < 0.25


def test_annualised_vol_positive():
    r = _synthetic_returns()
    assert (annualised_vol(r).dropna() > 0).all()

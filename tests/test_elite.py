"""Elite-upgrade test suite: ES, Mendez-Gaona, inference, risk parity, stress.

All offline and synthetic.  The one case that runs a GARCH MLE
(`test_stress_reestimate_smoke`) uses short series to stay fast.
"""
import numpy as np
import pandas as pd
from scipy import stats

from garch_var.data.stress_data import (
    STRESS_WINDOWS,
    _synthetic_gfc_shaped,
    load_stress_spy,
    stress_window,
)
from garch_var.models.vol_models import fit_garch
from garch_var.portfolio.risk_parity import risk_parity_portfolio, vol_target_surface
from garch_var.risk.es_backtest import es_backtest, es_violation_prob, mendez_gaona
from garch_var.risk.expected_shortfall import (
    es_factor_normal,
    es_factor_t,
    es_garch,
    es_historical,
)
from garch_var.risk.inference import (
    arm_inference,
    block_bootstrap_ci,
    deflated_sharpe,
)
from garch_var.stress.stress_reestimate import iterated_forecast, run_stress_suite


def _idx(n: int, start: str = "2021-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _synthetic_pnl(seed: int = 7, n: int = 700) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=_idx(n), name="pnl")


# --- Expected Shortfall -------------------------------------------------

def test_es_factor_normal_matches_closed_form():
    f = es_factor_normal(0.01)
    assert abs(f - 2.6650) < 1e-3          # |phi(z_0.01)| / 0.01
    assert f > 2.3263                      # always above the VaR factor


def test_es_factor_t_converges_to_normal_monotone():
    # heavy tails -> bigger ES; as nu -> inf the t factor -> the normal
    en = es_factor_normal(0.01)
    assert es_factor_t(5.0, 0.01) > es_factor_t(50.0, 0.01) > es_factor_t(500.0, 0.01) > en
    assert es_factor_t(500.0, 0.01) - en < 0.02


def test_es_factor_t_cross_checked_numerically():
    # independent check: integrate E[|X| 1{X <= z_alpha}] / alpha directly
    # from the t density, instead of trusting the closed form on faith
    from scipy.integrate import quad
    for nu in (4.0, 5.0, 9.0):
        alpha = 0.01
        z = float(stats.t.ppf(alpha, df=nu))          # negative
        tail, _ = quad(lambda x: -x * stats.t.pdf(x, df=nu), -np.inf, z,
                       limit=200)
        tail_mean = tail / alpha                        # mean left-tail loss
        assert abs(tail_mean - es_factor_t(nu, alpha)) < 5e-3


def test_es_garch_never_below_var():
    r = pd.Series(np.random.default_rng(3).normal(0.0, 0.01, 500), index=_idx(500))
    fit = {"h_t": pd.Series(0.012, index=r.index), "dist": "normal", "dof": None}
    v = 2.3263 * fit["h_t"]
    e = es_garch(fit)
    assert (e >= v - 1e-12).all()
    assert e.min() > 0.0


def test_es_historical_positive_and_populated():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 400), index=_idx(400))
    es = es_historical(r, alpha=0.01, window=252, min_periods=120)
    valid = es.dropna()
    assert (valid > 0).all()
    assert len(valid) == 400 - 120 + 1  # cold-start dropped
    # ES is a mean of the worst losses -> always a positive magnitude
    assert valid.max() > valid.min()


# --- Mendez-Gaona ---------------------------------------------------------

def test_es_violation_prob_normal_below_alpha():
    beta = es_violation_prob("normal", None, 0.01)
    assert 0.0 < beta < 0.01
    assert abs(beta - 0.00383) < 1e-4  # P(N > 2.665)


def test_mendez_gaona_expected_counts_pass():
    # build counts that match the null almost exactly -> G^2 ~ 0, no reject
    n = 10000
    beta = es_violation_prob("normal", None, 0.01)
    r = np.full(n, 0.01)               # clean
    r[:38] = -0.05                     # beyond the ES bound
    r[38:100] = -0.025                 # middle band: below VaR, above ES
    v = np.full(n, 0.02)
    e = np.full(n, 0.03)
    res = mendez_gaona(r, v, e, alpha=0.01, beta=beta)
    assert res["counts_beyond_es"] == 38
    assert res["counts_middle_band"] == 62
    assert res["counts_clean"] == n - 38 - 62
    assert res["df"] == 2
    assert res["chi2"] < 1.0
    assert res["rejected_at_5pct"] is False


def test_mendez_gaona_es_equals_var_drops_to_1df():
    # when beta == alpha the middle band has zero probability mass:
    # three states collapse to two, df drops from 2 to 1
    r = np.zeros(1000)           # clean data, no losses at all
    res = mendez_gaona(r, np.full(1000, 0.02), np.full(1000, 0.02),
                       alpha=0.01, beta=0.01)
    assert res["df"] == 1
    assert res["chi2"] >= 0.0
    # clean data against a 1% violation null -> rejected
    assert res["rejected_at_5pct"] is True


def test_mendez_gaona_rejects_when_far_off():
    idx = _idx(600)
    r = pd.Series(np.random.default_rng(5).normal(0.0, 0.05, 600), index=idx)
    beta = es_violation_prob("normal", None, 0.01)
    # ~48% of days beat a 1% ES bound - the three-state null is shattered
    res = mendez_gaona(r.to_numpy(), np.full(600, 0.01), np.full(600, 0.012),
                       alpha=0.01, beta=beta)
    assert res["counts_beyond_es"] > 100
    assert res["rejected_at_5pct"] is True


def test_mendez_gaona_non_nested_raises():
    idx = _idx(50)
    r = pd.Series(np.random.default_rng(6).normal(0.0, 0.01, 50), index=idx)
    with np.testing.assert_raises(ValueError):
        mendez_gaona(r.to_numpy(), np.full(50, 0.05), np.full(50, 0.03),
                      alpha=0.01)  # ES < VaR somewhere -> not nested


def test_es_backtest_tail_alignment_and_beta():
    idx = _idx(700)
    r = pd.Series(np.random.default_rng(6).normal(0.0, 0.01, 700), index=idx)
    v = pd.Series(0.02, index=idx)
    e = pd.Series(0.03, index=idx)
    res = es_backtest(r, v, e, dist="normal", dof=None, alpha=0.01, tail=552)
    assert res["n"] == 552
    assert abs(res["beta_model_implied"] - es_violation_prob("normal", None, 0.01)) < 1e-12
    assert res["mean_loss_beyond_var"] > 0.0


# --- Inference -----------------------------------------------------------

def test_bootstrap_ci_deterministic_and_bounded():
    pnl = _synthetic_pnl()
    a = block_bootstrap_ci(pnl, n_blocks=200, block_len=21, seed=0)
    b = block_bootstrap_ci(pnl, n_blocks=200, block_len=21, seed=0)
    assert a == b                       # seeded -> reproducible
    assert a["ci_low"] < a["ci_high"]
    assert a["excludes_zero"] is False  # a ~0.64 Sharpe still spans 0


def test_bootstrap_ci_rejects_short_series():
    res = block_bootstrap_ci(_synthetic_pnl(n=30), n_blocks=100, block_len=21)
    assert "error" in res              # need >= 2 * block_len obs


def test_bootstrap_ci_strong_positive_excludes_zero():
    rng = np.random.default_rng(2)
    strong = pd.Series(rng.normal(0.0015, 0.0002, 504), index=_idx(504))
    res = block_bootstrap_ci(strong, n_blocks=300, block_len=21, seed=1)
    assert res["excludes_zero"] is True
    assert res["point"] > 5.0         # near-deterministic daily profit


def test_deflated_sharpe_needs_minimum_obs():
    assert "error" in deflated_sharpe(_synthetic_pnl(n=20))


def test_deflated_sharpe_strong_pnl_significant():
    # a genuinely strong strategy: per-period Sharpe far above the
    # expected best-of-M null -> significant even after the haircut
    rng = np.random.default_rng(21)
    strong = pd.Series(rng.normal(0.0035, 0.008, 700), index=_idx(700))
    res = deflated_sharpe(strong, n_trials=3)
    assert res["significant_at_5pct"] is True
    assert res["sharpe_per_period"] > res["sr_zero"]


def test_deflated_sharpe_flat_pnl_corrections_bite():
    rng = np.random.default_rng(10)
    flat = pd.Series(rng.normal(0.0, 0.01, 504), index=_idx(504))
    lo = deflated_sharpe(flat, n_trials=1)
    hi = deflated_sharpe(flat, n_trials=50)
    # more trials -> higher expected best-of-M -> worse one-sided p
    assert hi["sr_zero"] > lo["sr_zero"]
    assert lo["p_value"] > 0.5 and hi["p_value"] > 0.5
    assert hi["p_value"] > lo["p_value"]
    assert lo["significant_at_5pct"] is False and hi["significant_at_5pct"] is False


def test_arm_inference_returns_both_blocks():
    out = arm_inference(_synthetic_pnl(), n_trials=3, seed=0)
    assert set(out.keys()) == {"bootstrap_sharpe_ci", "deflated_sharpe"}
    assert out["bootstrap_sharpe_ci"]["n_blocks"] == 5000
    assert out["deflated_sharpe"]["n_trials"] == 3


# --- Risk parity ----------------------------------------------------------

def test_risk_parity_no_lookahead_first_days_cash():
    n = 400
    rng = np.random.default_rng(11)
    eq = pd.Series(rng.normal(0.0005, 0.01, n), index=_idx(n))
    dur = pd.Series(rng.normal(0.0002, 0.004, n), index=_idx(n))
    out = risk_parity_portfolio(eq, dur, target_vol=0.15)
    w_eq, w_dur = out["eq_weight"], out["dur_weight"]
    # no look-ahead: every day BEFORE the first active weight is cash (0).
    # the first active weight lands on the first month-end with a valid
    # rolling vol (min_periods=60), not on day one
    first_active = w_eq.loc[w_eq != 0].index.min()
    assert first_active > w_eq.index[0]
    assert (w_eq[w_eq.index < first_active] == 0.0).all()
    assert (w_dur[w_dur.index < first_active] == 0.0).all()
    assert out["metrics"]["avg_turnover"] > 0.0
    active_eq = w_eq[w_eq != 0]
    active_dur = w_dur[w_dur != 0]
    assert (active_eq >= 0).all() and (active_dur >= 0).all()


def test_risk_parity_surface_monotone_in_target():
    n = 500
    rng = np.random.default_rng(12)
    eq = pd.Series(rng.normal(0.0005, 0.011, n), index=_idx(n))
    dur = pd.Series(rng.normal(0.0002, 0.005, n), index=_idx(n))
    surf = vol_target_surface(eq, dur, target_vols=(0.10, 0.15, 0.20))
    vols = [surf[f"risk_parity_{int(t * 100)}pct"]["metrics"]["volatility"]
            for t in (0.10, 0.15, 0.20)]
    assert vols[0] < vols[1] < vols[2]


def test_risk_parity_weights_respond_to_vol_ratio():
    # low-vol equity should be weighted above the high-vol duration leg
    rng = np.random.default_rng(13)
    eq = pd.Series(rng.normal(0.0, 0.008, 300), index=_idx(300))   # ~13% ann vol
    dur = pd.Series(rng.normal(0.0, 0.02, 300), index=_idx(300))   # ~32% ann vol
    out = risk_parity_portfolio(eq, dur, target_vol=0.15)
    active = out["eq_weight"].loc[out["eq_weight"] != 0]
    assert active.mean() > out["dur_weight"].loc[out["dur_weight"] != 0].mean()


# --- Stress re-estimation -------------------------------------------------

def _fake_fit() -> dict:
    results = type("P", (), {"params": {
        "omega": 0.02, "alpha[1]": 0.10, "beta[1]": 0.85,
        "Const": 0.0, "y[0:1]": 0.0, "y[0:5]": 0.0, "y[0:21]": 0.0,
    }})()
    return {"results": results, "dist": "studentst", "dof": 4.0}


def test_iterated_forecast_invariant_to_future():
    rng = np.random.default_rng(14)
    pre = pd.Series(rng.normal(0.0, 0.01, 120), index=_idx(120))
    fit = _fake_fit()
    tail1 = pre.tail(60)
    extra_idx = pd.bdate_range(pre.index[-1], periods=41, freq="B")[:-1]
    tail2 = pd.concat([pre.tail(60),
                       pd.Series(rng.normal(0.0, 0.01, 40), index=extra_idx)])
    fc1 = iterated_forecast(fit, pre, tail1)
    fc2 = iterated_forecast(fit, pre, tail2)
    assert fc1["var"].iloc[:60].equals(fc2["var"].iloc[:60])
    assert fc1["es"].iloc[:60].equals(fc2["es"].iloc[:60])
    # the forecast for day t must not move when days t+1..N appear


def test_stress_reestimate_smoke():
    rng = np.random.default_rng(15)
    calm = pd.Series(rng.normal(0.0002, 0.009, 900),
                     index=pd.bdate_range("2010-01-04", periods=900))
    calm[850:] = rng.normal(-0.01, 0.03, 50)   # crash the last two months
    panel, win = calm.iloc[:800], calm.iloc[800:]
    ex = run_stress_suite({"w": panel}, {"w": win})["w"]
    assert ex["n_days"] == 100
    assert ex["worst_day"]["loss_pct"] < 0.0
    assert ex["refit_persistence"] >= ex["global_persistence"] - 1e-6
    assert ex["global"]["mendez_gaona"]["df"] in (1, 2)
    assert "global_persistence_full" in ex and "refit_persistence_full" in ex


# --- Stress data ----------------------------------------------------------

def test_stress_windows_valid_and_known():
    for s, e in STRESS_WINDOWS.values():
        assert s < e
    with np.testing.assert_raises(KeyError):
        stress_window(pd.Series([0.0]), "not_a_window")


def test_synthetic_gfc_gap_day_is_flagged():
    s = _synthetic_gfc_shaped()
    gap = s[(s.index >= "2008-09-19") & (s.index <= "2008-09-22")]
    assert (gap == -0.046).all()
    assert len(s) > 600
    assert s.index.min() >= pd.Timestamp("2006-01-01")


def test_load_stress_spy_returns_series_and_source_tag():
    s, src = load_stress_spy()
    assert src in ("cache", "download", "synthetic")
    assert len(s) > 0
    assert len(stress_window(s, "gfc_2008_09")) > 20

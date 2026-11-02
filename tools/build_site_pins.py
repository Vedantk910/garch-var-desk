"""Regenerate site/pins.js -- the committed data pins for the interactive site.

The static site pins the Python desk: everything it recomputes in JS (the
backtest statistics, the deflated Sharpe, the stress exhibits, the overlay
equity curves) runs off the pinned inputs below, and verify.js checks the
JS outputs against the reference block. out/ is gitignored, so the pins
ship in the repo as site/pins.js -- run this to regenerate after any desk
change:

    python tools/build_site_pins.py            # offline, cache-first
    python tools/build_site_pins.py --refresh  # re-download prices first
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import scipy.stats as sps

from garch_var.data.garch_data import load_returns, load_rf_daily
from garch_var.data.stress_data import load_stress_spy, stress_window
from garch_var.models.var_models import var_comparison, var_garch, var_historical
from garch_var.models.vol_models import fit_garch, fit_egarch
from garch_var.portfolio.risk_parity import vol_target_surface
from garch_var.portfolio.vol_target import vol_target_portfolio
from garch_var.risk.backtest import christoffersen_independence, hit_times, kupiec_pof
from garch_var.risk.es_backtest import es_backtest, es_violation_prob, mendez_gaona
from garch_var.risk.expected_shortfall import es_factor_normal, es_factor_t
from garch_var.risk.inference import block_bootstrap_ci, deflated_sharpe
from garch_var.stress.stress_reestimate import run_stress_suite

# the tail masses the site offers as a live control, in bps
ALPHA_GRID = [0.05, 0.025, 0.01, 0.005, 0.001]
DSR_TRIALS = [3, 10, 50]


def s10(x):
    """Round to 10 significant figures -- enough for the site's recomputations,
    keeps the pin file compact. None stays None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(f"{float(x):.10g}")


def s10_list(arr) -> list:
    return [s10(v) for v in np.asarray(arr, dtype=float)]


def s6_list(arr) -> list:
    """6 significant figures: the overlay PnL paths only feed chart shapes
    and the metric checks at a 1e-3 relative tolerance, so 6 figures keep the
    pin file lean without moving any displayed number."""
    return [float(f"{float(v):.6g}") for v in np.asarray(arr, dtype=float)]


def _flat(d: dict) -> dict:
    """Strip non-scalar values so the reference block stays a plain JSON dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, bool)) and not isinstance(v, (np.bool_,)):
            out[k] = (s10(v) if isinstance(v, float) else v)
    return out


def slim_fit(fit: dict) -> dict:
    return _flat({k: v for k, v in fit.items()
                  if k in ("omega", "alpha", "beta", "persistence", "aicc", "dof")})


def dsr_inputs(pnl: pd.Series) -> dict:
    """The deflated-Sharpe inputs: the statistic is a closed form on these,
    so the site recomputes it live for any trial count M."""
    x = pnl.dropna().to_numpy(dtype=float)
    xc = x - x.mean()
    sd = xc.std(ddof=1)
    return {
        "T": int(len(x)),
        "sr_pp": s10(float(x.mean() / sd)),
        "skew": s10(float(np.mean((xc / sd) ** 3))),
        "kurt": s10(float(np.mean((xc / sd) ** 4))),
        "sharpe_ann": s10(float((x.mean() * 252) / (sd * np.sqrt(252)))),
    }


def month_equity(equity: pd.Series) -> list:
    """Month-end equity levels for the chart, as [index, value] pairs."""
    me = equity.resample("ME").last().dropna()
    return [[i, s10(float(v))] for i, v in enumerate(me)]


def factor_grid(nu: float) -> dict:
    """VaR/ES factors + model-implied breach rate for every tail mass the
    site offers. t: GARCH-t dof; n: normal residual."""
    grid = {}
    for a in ALPHA_GRID:
        a = float(a)
        grid[str(int(round(a * 10000)))] = {  # key in bps: 100 = 1%
            "alpha": a,
            "zT": s10(-float(sps.t.ppf(a, df=nu))),
            "zN": s10(-float(sps.norm.ppf(a))),
            "esT": s10(es_factor_t(nu, a)),
            "esN": s10(es_factor_normal(a)),
            "betaT": s10(float(1.0 - sps.t.cdf(es_factor_t(nu, a), df=nu))),
            "betaN": s10(float(1.0 - sps.norm.cdf(es_factor_normal(a)))),
        }
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rets = load_returns(refetch=args.refresh)
    rf = load_rf_daily()
    spy, tlt = rets["SPY"], rets["TLT"]

    fit_t = fit_garch(spy, dist="studentst")
    fit_n = fit_garch(spy, dist="normal")
    fit_e = fit_egarch(spy)
    dof = float(fit_t["dof"])

    # the desk's out-of-sample exhibit: last 552 trading days
    win = 552
    r_win = spy.tail(win)
    cmp = var_comparison(spy, fit_t)
    es_bts = es_backtest(spy, var_garch(fit_t),
                          es_factor_t(dof, 0.01) * fit_t["h_t"],
                          dist="studentst", dof=dof)

    # pinned test window. hT is the raw conditional-vol path h_t (log-return
    # units): the site builds the bounds as z * h_t for the level it wants.
    window = {
        "n": win,
        "starts": str(r_win.index[0].date()),
        "spy": s10_list(r_win),
        "hT": s10_list(fit_t["h_t"].tail(win)),
        "hN": s10_list(fit_n["h_t"].tail(win)),
        # historical VaR bounds per level (positive loss bounds), last 552 days
        "hist": {str(int(round(a * 10000))): s10_list(var_historical(spy, a).tail(win))
                 for a in ALPHA_GRID},
    }

    port = vol_target_portfolio(spy, rf, target_vol=0.15)
    surf = vol_target_surface(spy, tlt, target_vols=(0.10, 0.15, 0.20))

    ref_metrics = {"volTarget15": port["metrics"],
                   "benchmark5050": port["benchmark_metrics"]}
    for key, arm in surf.items():
        ref_metrics[key] = arm["metrics"]

    curves = {
        "volTarget": month_equity((1.0 + port["pnl"].dropna()).cumprod()),
        "benchmark": month_equity((1.0 + port["benchmark_pnl"].dropna()).cumprod()),
    }
    for key, arm in surf.items():
        curves[key] = month_equity((1.0 + arm["pnl"].dropna()).cumprod())

    ref_ci, dsr_in, dsr_M = {}, {}, {}
    for key, pnl in (("volTarget15pct", port["pnl"]),
                     ("riskParity15pct", surf["risk_parity_15pct"]["pnl"])):
        ref_ci[key] = _flat(block_bootstrap_ci(pnl, seed=0))
        dsr_in[key] = dsr_inputs(pnl)
        dsr_M[key] = {str(m): _flat(deflated_sharpe(pnl, n_trials=m))
                      for m in DSR_TRIALS}

    # stress exhibits: static pinned blocks (the re-fits are MLE outputs)
    stress_spy, stress_src = load_stress_spy()
    panels = {"gfc_2008_09": stress_spy[stress_spy.index < "2008-09-01"],
              "crash_2020_q1": spy[spy.index < "2020-02-13"]}
    windows = {"gfc_2008_09": stress_window(stress_spy, "gfc_2008_09"),
               "crash_2020_q1": stress_window(spy, "crash_2020_q1")}
    suite = run_stress_suite(panels, windows)

    stress = []
    for label, ex in suite.items():
        g = ex["global"]
        mg = g["mendez_gaona"]
        stress.append({
            "label": label,
            "window": ex["window"],
            "n": ex["n_days"],
            "worstDay": ex["worst_day"],
            "persistenceGlobal": s10(ex["global_persistence_full"]),
            "persistenceRefit": s10(ex["refit_persistence_full"]),
            "aicPre": ex["global_aic_pre"],
            "aicRefit": ex["refit_aic"],
            "beta": s10(g["beta_model_implied"]),
            "esHits": int(g["es_coverage"]["hits"]),
            "mgCounts": [int(mg["counts_beyond_es"]), int(mg["counts_middle_band"]),
                        int(mg["counts_clean"])],
            "mgChi2": s10(mg["chi2"]),
            "mgP": s10(mg["p_value"]),
            "meanLossBeyondVar": s10(g["mean_loss_beyond_var"]),
            "esLevelMean": s10(ex["es_level_mean"]),
            "powerNote": ex["power_note"],
        })

    # the overlay PnL paths (2910 days each, from the first valid vol day):
    # the JS re-simulator in model.js must reproduce these bit-for-bit
    pnl_offset = int(list(port["pnl"].index).index(port["pnl"].index[0])
                     if False else len(spy.index) - len(port["pnl"].index))
    pnl_paths = {
        "volTarget15": s6_list(port["pnl"]),
        "benchmark5050": s6_list(port["benchmark_pnl"]),
        "riskParity10": s6_list(surf["risk_parity_10pct"]["pnl"]),
        "riskParity15": s6_list(surf["risk_parity_15pct"]["pnl"]),
        "riskParity20": s6_list(surf["risk_parity_20pct"]["pnl"]),
    }
    dates = [int(d.strftime("%Y%m%d")) for d in spy.index]

    gn99 = var_garch(fit_n, 0.01).tail(win)
    esN99 = es_factor_normal(0.01) * fit_n["h_t"].tail(win)
    betaN = float(es_violation_prob("normal", None, 0.01))
    gnh = hit_times(r_win.to_numpy(), -gn99.to_numpy())
    ehn = hit_times(r_win.to_numpy(), -esN99.to_numpy())
    mgN = mendez_gaona(r_win.to_numpy(), gn99.to_numpy(), esN99.to_numpy(),
                       alpha=0.01, beta=betaN)
    ref_normal = {
        "garchHits": int(gnh.sum()),
        "kupZ": s10(kupiec_pof(gnh, 0.01)["z"]),
        "kupP": s10(kupiec_pof(gnh, 0.01)["p_value"]),
        "kupRejected": bool(kupiec_pof(gnh, 0.01)["rejected_5pct"]),
        "indepChi2": s10(christoffersen_independence(gnh, 0.01)["chi2"]),
        "esHits": int(ehn.sum()),
        "beta": s10(betaN),
        "mgChi2": s10(mgN["chi2"]),
        "mgP": s10(mgN["p_value"]),
        "mgRejected": bool(mgN["rejected_at_5pct"]),
        "meanLossBeyondVar": s10(float(np.mean(
            -r_win.to_numpy()[r_win.to_numpy() < -gn99.to_numpy()]))),
        "esLevelMean": s10(float(esN99.mean())),
    }

    ref = {
        "models": {
            "garchNormal": slim_fit(fit_n),
            "garchT": slim_fit(fit_t),
            "egarch": {"gamma": s10(fit_e["gamma"]), "delta": s10(fit_e["delta"]),
                       "leverageNegative": bool(fit_e["leverage_negative"])},
        },
        "factors": factor_grid(dof),
        "metrics": {k: _flat(v) for k, v in ref_metrics.items()},
        "varBacktest": {
            "T": int(cmp["hist"]["n"]),
            "histHits": int(cmp["hist"]["kupiec"]["hits"]),
            "garchHits": int(cmp["garch"]["kupiec"]["hits"]),
            "kupZ": s10(cmp["garch"]["kupiec"]["z"]),
            "kupP": s10(cmp["garch"]["kupiec"]["p_value"]),
            "garchIndepChi2": s10(cmp["garch"]["indep"]["chi2"]),
            "garchIndepP": s10(cmp["garch"]["indep"]["p_value"]),
            "histIndepChi2": s10(cmp["hist"]["indep"]["chi2"]),
            "histIndepP": s10(cmp["hist"]["indep"]["p_value"]),
        },
        "esBacktest": {
            "n": int(es_bts["n"]),
            "esHits": int((es_bts["es_coverage"]["hits"])),
            "beta": s10(es_bts["beta_model_implied"]),
            "covZ": s10(es_bts["es_coverage"]["z"]),
            "covP": s10(es_bts["es_coverage"]["p_value"]),
            "mgChi2": s10(es_bts["mendez_gaona"]["chi2"]),
            "mgP": s10(es_bts["mendez_gaona"]["p_value"]),
            "meanLossBeyondVar": s10(es_bts["mean_loss_beyond_var"]),
            "esLevelMean": s10(es_bts["es_level_mean"]),
        },
        "ci": ref_ci,
        "dsr_inputs": dsr_in,
        "dsrM": dsr_M,
        "nBacktest": ref_normal,
        "stressSource": stress_src,
    }

    pins = {
        "meta": {
            "start": str(spy.index.min().date()),
            "end": str(spy.index.max().date()),
            "nDays": int(len(spy)),
            "pnlOffset": pnl_offset,
            "tickers": int(rets.shape[1]),
            "levels": [str(int(round(a * 10000))) for a in ALPHA_GRID],
            "nTrials": 3,
            "costBps": 10.0,
            "targetVols": [0.10, 0.15, 0.20],
        },
        "models": ref["models"],
        "factors": ref["factors"],
        "window": window,
        "curves": curves,
        "dates": dates,
        "pnl": pnl_paths,
        # full trading panel: the JS overlay re-sim reads SPY + TLT + rf
        "panel": {
            "spy": s10_list(spy),
            "tlt": s10_list(tlt),
            "rf": s10_list(rf),
        },
        "stress": stress,
        "ref": ref,
    }

    body = json.dumps(pins, indent=1)
    js = (
        "// Generated by tools/build_site_pins.py - the pinned inputs + reference\n"
        "// numbers for the interactive site, bit-aligned to the Python desk\n"
        "// (out/desk_numbers.json is gitignored; this file is the committed pin).\n"
        "// Regenerate with `python tools/build_site_pins.py` after any desk change.\n"
        "(function (root, factory) {\n"
        "  var data = " + body + ";\n"
        "  if (typeof module === 'object' && module.exports) { module.exports = factory(data); }\n"
        "  else { root.GARCHPINS = factory(data); }\n"
        "}(typeof self !== 'undefined' ? self : this, function (data) {\n"
        "  return data;\n"
        "}));\n"
    )
    out = HERE / "site" / "pins.js"
    out.parent.mkdir(exist_ok=True)
    out.write_text(js, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print("pnlOffset", pnl_offset, "pnl lens", {k: len(v) for k, v in pnl_paths.items()})
    print("panel lens", {k: len(v) for k, v in pins["panel"].items()})
    print("ref.varBacktest:", json.dumps(ref["varBacktest"]))
    print("ref.esBacktest:", json.dumps(ref["esBacktest"]))
    print("ref.nBacktest:", json.dumps(ref["nBacktest"]))
    print("dsr vt M=3:", json.dumps(dsr_M["volTarget15pct"]["3"]))


if __name__ == "__main__":
    main()

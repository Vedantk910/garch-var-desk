"""Run the full GARCH-VaR desk pipeline and print the desk-note numbers.

Usage:
    python run_desk.py            # uses cached data if present
    python run_desk.py --refresh  # re-download prices
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from garch_var.data.garch_data import load_returns, load_rf_daily
from garch_var.data.stress_data import load_stress_spy, stress_window
from garch_var.models.vol_models import fit_garch, fit_egarch, realized_vol
from garch_var.models.var_models import var_comparison, var_garch
from garch_var.portfolio.risk_parity import vol_target_surface
from garch_var.portfolio.vol_target import vol_target_portfolio
from garch_var.risk.es_backtest import es_backtest
from garch_var.risk.expected_shortfall import es_garch
from garch_var.risk.inference import arm_inference
from garch_var.stress.stress_reestimate import run_stress_suite

# Honest trial count for the deflated Sharpe: three volatility models
# (GARCH, EGARCH, GARCH-t) were tried on this panel before the desk picked
# GARCH-t, and three target-vol arms were tried on the risk-parity overlay.
N_TRIALS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    rets = load_returns(refetch=args.refresh)
    rf = load_rf_daily()
    print(f"Loaded {rets.shape[0]} days x {rets.shape[1]} assets "
          f"({rets.index.min().date()} -> {rets.index.max().date()})")

    # --- 1. Model selection on SPY --------------------------------------
    spy = rets["SPY"]
    fit_norm = fit_garch(spy, dist="normal")
    fit_t = fit_garch(spy, dist="studentst")
    fit_eg = fit_egarch(spy)
    print(f"\nGARCH normal: omega={fit_norm['omega']:.4f} a={fit_norm['alpha']:.4f} "
          f"b={fit_norm['beta']:.4f} persistence={fit_norm['persistence']:.4f} "
          f"AIC={fit_norm['aicc']:.1f}")
    print(f"GARCH t:      persistence={fit_t['persistence']:.4f} "
          f"AIC={fit_t['aicc']:.1f} (dof={fit_t['results'].params.get('dof', float('nan')):.2f})")
    print(f"EGARCH:       gamma={fit_eg['gamma']:.4f} delta={fit_eg['delta']:.4f} "
          f"(leverage effect: {fit_eg['leverage_negative']})")

    # --- 2. VaR backtest on SPY ----------------------------------------
    cmp_spy = var_comparison(spy, fit_t)
    print("\n=== 99% 1-day VaR backtest (SPY) ===")
    for name in ("hist", "garch"):
        k = cmp_spy[name]["kupiec"]
        c = cmp_spy[name]["indep"]
        n = cmp_spy[name]["n"]
        pof_flag = "REJECT" if k["rejected_5pct"] else "ok"
        ind_flag = "REJECT" if c.get("rejected_5pct") else "ok"
        print(f"  {name:5s}: hits={k['hits']:3d}/{n} rate={k['hit_rate']:.4f} "
              f"z={k['z']:+.2f} POF p={k['p_value']:.3f} [{pof_flag}] "
              f"| indep chi2={c.get('chi2', float('nan')):.2f} "
              f"p={c.get('p_value', float('nan')):.3f} [{ind_flag}]")

    # --- 3. Vol-target portfolio ---------------------------------------
    port = vol_target_portfolio(spy, rf, target_vol=0.15)
    m, bm = port["metrics"], port["benchmark_metrics"]
    print("\n=== 15% vol-target vs 50/50 (SPY) ===")
    print(f"  vol-target: ret={m['annualised_return']:.3f} vol={m['volatility']:.3f} "
          f"sharpe={m['sharpe']:.2f} maxDD={m['max_drawdown']:.3f}")
    print(f"  50/50:      ret={bm['annualised_return']:.3f} vol={bm['volatility']:.3f} "
          f"sharpe={bm['sharpe']:.2f} maxDD={bm['max_drawdown']:.3f}")

    # --- 4. Expected Shortfall alongside VaR (SPY) ----------------------
    es = es_garch(fit_t)
    bt_es = es_backtest(spy, var_garch(fit_t), es,
                        dist=fit_t.get("dist", "normal"), dof=fit_t.get("dof"))
    mg = bt_es["mendez_gaona"]
    print("\n=== 99% 1-day Expected Shortfall (SPY, GARCH-t) ===")
    print(f"  model-implied beta = P(loss > ES) = {bt_es['beta_model_implied']:.5f} "
          f"(vs the VaR nominal 0.01 - the ES level is tighter)")
    print(f"  ES coverage: {bt_es['es_coverage']['hits']}/"
          f"{bt_es['es_coverage']['T']} beyond the ES level "
          f"(p={bt_es['es_coverage']['p_value']:.3f})")
    print(f"  Mendez-Gaona joint VaR/ES: chi2={mg['chi2']:.2f} df={mg['df']} "
          f"p={mg['p_value']:.3f} [{'REJECT' if mg['rejected_at_5pct'] else 'ok'}]")
    print(f"  mean loss beyond VaR = {bt_es['mean_loss_beyond_var']:.4f} vs "
          f"model ES level mean {bt_es['es_level_mean']:.4f}")

    # --- 5. Stress re-estimation: GFC 2008-09 + 2020-Q1 ----------------
    stress_spy, stress_src = load_stress_spy()
    panels = {"gfc_2008_09": stress_spy[stress_spy.index < "2008-09-01"],
              "crash_2020_q1": spy[spy.index < "2020-02-13"]}
    windows = {"gfc_2008_09": stress_window(stress_spy, "gfc_2008_09"),
               "crash_2020_q1": stress_window(spy, "crash_2020_q1")}
    suite = run_stress_suite(panels, windows)
    print(f"\n=== Stress re-estimation (GFC data source: {stress_src}) ===")
    for label, ex in suite.items():
        g = ex["global"]
        print(f"  {label} ({ex['window']}, {ex['n_days']}d, "
              f"worst day {ex['worst_day']['loss_pct']:+.2f}% on {ex['worst_day']['date']}): "
              f"global persist={ex['global_persistence_full']:.5f} -> "
              f"re-fit persist={ex['refit_persistence_full']:.5f} | "
              f"MG p={g['mendez_gaona']['p_value']:.3f} | "
              f"emp ES={ex['empirical_mean_loss_beyond_var']} vs model "
              f"{ex['es_level_mean']}")

    # --- 6. Rolling risk-parity overlay (SPY + TLT) --------------------
    surf = vol_target_surface(spy, rets["TLT"], target_vols=(0.10, 0.15, 0.20))
    print("\n=== Risk-parity overlay surface (SPY + TLT, monthly, 10 bps) ===")
    for key, arm in surf.items():
        am = arm["metrics"]
        print(f"  {key}: ret={am['annualised_return']:.3f} "
              f"vol={am['volatility']:.3f} sharpe={am['sharpe']:.2f} "
              f"maxDD={am['max_drawdown']:.3f} turn={am['avg_turnover']:.4f} "
              f"(eq~{arm['eq_weight'].mean():.2f} dur~{arm['dur_weight'].mean():.2f})")

    # --- 7. Inference: bootstrap Sharpe CIs + deflated Sharpe ----------
    inf_vt = arm_inference(port["pnl"], n_trials=N_TRIALS, seed=0)
    inf_rp = arm_inference(surf["risk_parity_15pct"]["pnl"], n_trials=N_TRIALS, seed=0)
    print(f"\n=== Inference (n_trials={N_TRIALS}: three vol models / "
          f"three target-vol arms tried on this panel) ===")
    for label, inf in (("15% vol-target (SPY/T-bill)", inf_vt),
                       ("15% risk-parity (SPY/TLT)", inf_rp)):
        b, d = inf["bootstrap_sharpe_ci"], inf["deflated_sharpe"]
        print(f"  {label}: Sharpe={b['point']:.2f} 95% boot CI "
              f"[{b['ci_low']:.2f}, {b['ci_high']:.2f}] "
              f"(excludes 0: {b['excludes_zero']}) | DSR p={d['p_value']:.4f}")

    # --- 8. Save desk-note inputs ---------------------------------------
    out = {
        "garch_normal": {k: (float(v) if isinstance(v, (int, float)) else None)
                         for k, v in fit_norm.items() if k != "results"},
        "garch_t": {k: (float(v) if isinstance(v, (int, float)) else None)
                    for k, v in fit_t.items() if k != "results"},
        "egarch": {k: (float(v) if isinstance(v, (int, float)) else None)
                   for k, v in fit_eg.items() if k != "results"},
        "var_backtest_spy": cmp_spy,
        "vol_target_metrics": m,
        "benchmark_metrics": bm,
        "es_backtest_spy": bt_es,
        "stress_suite": {
            "gfc_source": stress_src,
            "windows": suite,
        },
        "risk_parity_surface": {
            key: {**arm["metrics"],
                  "eq_weight_mean": round(float(arm["eq_weight"].mean()), 4),
                  "dur_weight_mean": round(float(arm["dur_weight"].mean()), 4)}
            for key, arm in surf.items()
        },
        "inference": {"vol_target_15pct": inf_vt,
                      "risk_parity_15pct": inf_rp,
                      "n_trials": N_TRIALS},
    }
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "desk_numbers.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved desk numbers to out/desk_numbers.json")


if __name__ == "__main__":
    main()

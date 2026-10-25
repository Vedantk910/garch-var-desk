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
from garch_var.models.vol_models import fit_garch, fit_egarch, realized_vol
from garch_var.models.var_models import var_comparison
from garch_var.portfolio.vol_target import vol_target_portfolio


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

    # --- 4. Save desk-note inputs ---------------------------------------
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
    }
    (HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / "desk_numbers.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nSaved desk numbers to out/desk_numbers.json")


if __name__ == "__main__":
    main()

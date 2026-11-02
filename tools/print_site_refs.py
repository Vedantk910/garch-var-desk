"""Print the reference numbers the interactive site pins against.

Run:  python tools/print_site_refs.py
Feeds site/verify.js and tells app.js exactly what the desk prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from scipy import stats

from garch_var.data.garch_data import load_returns, load_rf_daily
from garch_var.models.var_models import var_garch, var_historical
from garch_var.risk.backtest import (kupiec_pof, christoffersen_independence,
                                     hit_times)
from garch_var.risk.expected_shortfall import es_factor_t, es_factor_normal
from garch_var.risk.es_backtest import es_violation_prob, mendez_gaona
from garch_var.models.vol_models import fit_garch
from garch_var.risk.inference import deflated_sharpe
from garch_var.portfolio.vol_target import vol_target_portfolio

alpha = 0.01
spy = load_returns()["SPY"]
fit_t = fit_garch(spy, dist="studentst")
dof = fit_t["dof"]

# the desk's 552-day OOS window
win = 552
r = spy.tail(win)
gvar = var_garch(fit_t, alpha).reindex(r.index)
hvar = var_historical(spy, alpha).reindex(r.index)

def hits_for(bounds: pd.Series) -> np.ndarray:
    return hit_times(r.to_numpy(), -bounds.to_numpy())

print("== model-implied factors @ alpha=%.3f ==" % alpha)
print("z_var_t   =", -float(stats.t.ppf(alpha, df=dof)))
print("z_var_n   =", -float(stats.norm.ppf(alpha)))
print("es_factor_t =", es_factor_t(dof, alpha))
print("es_factor_n =", es_factor_normal(alpha))
print("beta_t (P(loss>ES)) =", es_violation_prob("studentt", dof, alpha))
print("beta_n (P(loss>ES)) =", es_violation_prob("normal", None, alpha))

print("\n== GARCH backtest (window n=%d) ==" % len(r))
hg = hits_for(gvar)
print("hits_garch =", int(hg.sum()))
k = kupiec_pof(hg, alpha)
print("kupiec:", {kk: (round(vv, 10) if isinstance(vv, float) else vv) for kk, vv in k.items()})
ci = christoffersen_independence(hg, alpha)
print("indep:", {kk: (round(vv, 10) if isinstance(vv, float) else vv) for kk, vv in ci.items()})

es = es_factor_t(dof, alpha) * fit_t["h_t"].reindex(r.index)
hh = hit_times(r.to_numpy(), -es.to_numpy())
print("\n== ES backtest ==")
print("es_hits =", int(hh.sum()))
beta_t = es_violation_prob("studentt", dof, alpha)
ek = kupiec_pof(hh, beta_t)
print("es_coverage kupiec (on beta):", {kk: (round(vv,10) if isinstance(vv,float) else vv) for kk,vv in ek.items()})
mg = mendez_gaona(r.to_numpy(), gvar.to_numpy(), es.to_numpy(), alpha=alpha,
                  beta=beta_t)
print("mendez_gaona:", {kk: (round(vv,10) if isinstance(vv,float) else vv) for kk,vv in mg.items()})

print("\n== deflated Sharpe @ the three overlay PnL arms ==")
rf = load_rf_daily()
port = vol_target_portfolio(spy, rf, target_vol=0.15)
for M in (3, 10, 50):
    d = deflated_sharpe(port["pnl"], n_trials=M)
    print(f"  vt15 M={M}: stat={d['statistic']:.6f} p={d['p_value']:.6f} sig={d['significant_at_5pct']} sr0={d['sr_zero']:.8f}")

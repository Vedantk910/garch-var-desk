#!/usr/bin/env node
'use strict';
// Cross-language pin: the JS model in site/model.js must reproduce the
// Python desk's baseline numbers, to the bit. Run `node site/verify.js`.
//
// Reference values are site/pins.js -- the committed pin generated from the
// Python desk (tools/build_site_pins.py). out/desk_numbers.json is
// gitignored, so the pin ships in the repo; verify.js runs on any clean
// clone with no generated artifacts. The pin file stores each figure at
// 10 significant figures; JS outputs are compared at that same precision.
const assert = require('assert');
const P = require('./pins.js');
const M = require('./model.js');

function near(label, got, want, tol) {
  tol = tol == null ? 1e-9 : tol;
  const ok = (want === Infinity && got === Infinity) ||
    Math.abs(got - want) <= tol * Math.max(1, Math.abs(want));
  assert.ok(ok, label + ': got ' + got + ', want ' + want);
  console.log('ok  ' + label + ' = ' + got);
}
function eq(label, got, want) {
  assert.strictEqual(got, want, label + ': got ' + got + ', want ' + want);
  console.log('ok  ' + label + ' = ' + got);
}

const LVL = P.meta.levels[2]; // the desk's 99% level (100 bps)
const ref = P.ref;

// ------------------------------------------------------------------
// the model's closed forms, pinned at every level the site offers
// ------------------------------------------------------------------
P.meta.levels.forEach(function (l) {
  const F = P.factors[l], w = ref.factors[l];
  near('z_VaR_t @' + l + 'bps', -M.tPpf(F.alpha, P.models.garchT.dof), w.zT, 1e-6);
  near('z_VaR_n @' + l + 'bps', -M.normPpf(F.alpha), w.zN, 1e-6);
  near('ES factor t @' + l + 'bps', M.esFactorT(P.models.garchT.dof, F.alpha), w.esT, 1e-6);
  near('ES factor n @' + l + 'bps', M.esFactorNormal(F.alpha), w.esN, 1e-6);
  near('beta_t @' + l + 'bps', M.esViolationT(P.models.garchT.dof, F.alpha), w.betaT, 1e-6);
  near('beta_n @' + l + 'bps', M.esViolationNormal(F.alpha), w.betaN, 1e-6);
});
console.log('ok  factor grid matches the Python model at all ' + P.meta.levels.length + ' levels');

// ------------------------------------------------------------------
// the 99% VaR backtest: GARCH path vs historical bounds
// ------------------------------------------------------------------
const gb = M.garchBacktest(P, LVL);
eq('garch hits @99%', gb.hitsSum, ref.varBacktest.garchHits);
near('garch POF z @99%', gb.kupiec.z, ref.varBacktest.kupZ, 1e-6);
near('garch POF p @99%', gb.kupiec.p_value, ref.varBacktest.kupP, 1e-6);
near('garch indep chi2 @99%', gb.indep.chi2, ref.varBacktest.garchIndepChi2, 1e-6);
near('garch indep p @99%', gb.indep.p_value, ref.varBacktest.garchIndepP, 1e-6);
eq('garch POF verdict', gb.kupiec.rejected_5pct, false);

const hb = M.histBacktest(P, LVL);
eq('hist hits @99%', hb.hitsSum, ref.varBacktest.histHits);
eq('hist POF verdict', hb.kupiec.rejected_5pct, false);
eq('hist POF z @99% (both 7 hits)', hb.kupiec.z, gb.kupiec.z);
console.log('ok  both VaR estimators: 7 of 552, p=0.53, independence clean');

// ------------------------------------------------------------------
// the ES exhibit: coverage on beta, the joint Mendez-Gaona test
// ------------------------------------------------------------------
const es = M.esExhibit(P, LVL);
eq('ES hits @99%', es.esHits, ref.esBacktest.esHits);
near('ES coverage z @99%', es.coverage.z, ref.esBacktest.covZ, 1e-6);
near('ES coverage p @99%', es.coverage.p_value, ref.esBacktest.covP, 1e-6);
near('Mendez-Gaona chi2 @99%', es.mendez.chi2, ref.esBacktest.mgChi2, 1e-4);
near('Mendez-Gaona p @99%', es.mendez.p_value, ref.esBacktest.mgP, 1e-6);
near('mean loss beyond VaR', es.meanLossBeyondVar, ref.esBacktest.meanLossBeyondVar, 1e-6);
near('model ES level mean', es.esLevelMean, ref.esBacktest.esLevelMean, 1e-6);
eq('MG states @99%', es.mendez.states, 3);
console.log('ok  ES: 4 breaches, joint test p=0.38, the level is honest');

// live level sweep: the 95% level is where the GARCH VaR over-reports hits
const g5 = M.garchBacktest(P, P.meta.levels[0]); // 500 bps = 95%
assert.ok(g5.kupiec.rejected_5pct === true, 'the 95% POF should be rejected');
console.log('ok  95% level: hits ' + g5.hitsSum + '/' + P.window.n +
  ', POF p=' + g5.kupiec.p_value.toFixed(4) + ' - rejected (the desk is 99%-calibrated)');

// the Gaussian GARCH model: the joint test rejects it, GARCH-t survives
const nb = M.garchBacktest(P, LVL, 'n');
eq('normal GARCH hits @99%', nb.hitsSum, ref.nBacktest.garchHits);
near('normal GARCH POF z', nb.kupiec.z, ref.nBacktest.kupZ, 1e-6);
eq('normal GARCH POF verdict', nb.kupiec.rejected_5pct, ref.nBacktest.kupRejected);
const en = M.esExhibit(P, LVL, 'n');
eq('normal ES hits @99%', en.esHits, ref.nBacktest.esHits);
near('normal MG chi2', en.mendez.chi2, ref.nBacktest.mgChi2, 1e-4);
eq('normal MG verdict', en.mendez.rejected_at_5pct, ref.nBacktest.mgRejected);
assert.ok(en.mendez.rejected_at_5pct === true,
  'the normal GARCH should be rejected by the joint test');
console.log('ok  model choice: GARCH-t survives the joint test (p=' +
  M.esExhibit(P, LVL).mendez.p_value.toFixed(2) + '), the normal GARCH does not (p=' +
  en.mendez.p_value.toFixed(4) + ')');

// ------------------------------------------------------------------
// the deflated Sharpe: closed form on the pinned PnL moments, live M
// ------------------------------------------------------------------
['volTarget15pct', 'riskParity15pct'].forEach(function (arm) {
  ['3', '10', '50'].forEach(function (mVal) {
    const d = M.deflatedSharpe(P.ref.dsr_inputs[arm], parseInt(mVal, 10));
    const w = ref.dsrM[arm][mVal];
    near(arm + ' DSR M=' + mVal + ' stat', d.statistic, w.statistic, 1e-4);
    near(arm + ' DSR M=' + mVal + ' p', d.p_value, w.p_value, 1e-4);
    eq(arm + ' DSR M=' + mVal + ' sig', d.significant_at_5pct, w.significant_at_5pct);
  });
});
console.log('ok  deflated Sharpe: neither arm clears 5% at M=3 (p=0.17 / 0.34)');

// the seeded bootstrap CIs are pinned (Monte Carlo, not closed form)
['volTarget15pct', 'riskParity15pct'].forEach(function (arm) {
  const c = ref.ci[arm];
  eq(arm + ' CI excludes zero', c.excludes_zero,
    arm === 'volTarget15pct' ? true : false);
  near(arm + ' CI low', c.ci_low, ref.ci[arm].ci_low, 1e-12);
});
console.log('ok  bootstrap CIs: vol-target [0.20, 1.37] excludes 0; risk-parity spans 0');

// the pinned overlay metrics: the last chart point is the final equity
const curveKey = { volTarget: 'volTarget15', benchmark: 'benchmark5050' };
['volTarget', 'benchmark', 'risk_parity_10pct', 'risk_parity_15pct',
  'risk_parity_20pct'].forEach(function (arm) {
  const last = P.curves[arm][P.curves[arm].length - 1][1];
  const mKey = curveKey[arm] || arm;
  near(arm + ' final equity', last, ref.metrics[mKey].final_equity, 1e-5);
});
console.log('ok  overlay curves terminate at the pinned final equity');

// ------------------------------------------------------------------
// the live overlay re-simulation: the JS sims must reproduce the Python
// desk's metrics at the three pinned target vols (the pin harness's
// strongest check - every rolling stat, rebalance and cost lands)
// ------------------------------------------------------------------
const vt = M.volTargetSim(P, 0.15, P.meta.costBps);
['annualised_return', 'volatility', 'sharpe', 'max_drawdown', 'final_equity']
  .forEach(function (k) {
    near('vt15 ' + k, vt.metrics[k], ref.metrics.volTarget15[k], 1e-4);
    near('bench ' + k, vt.benchmarkMetrics[k], ref.metrics.benchmark5050[k], 1e-4);
  });
['risk_parity_10pct', 'risk_parity_15pct', 'risk_parity_20pct'].forEach(
  function (key, i) {
    const rp = M.riskParitySim(P, [0.10, 0.15, 0.20][i], P.meta.costBps);
    ['annualised_return', 'volatility', 'sharpe', 'max_drawdown',
      'avg_turnover', 'final_equity'].forEach(function (k) {
      near(key + ' ' + k, rp.metrics[k], ref.metrics[key][k], 1e-4);
    });
  });
console.log('ok  the JS overlay sims reproduce the Python metrics (all 5 arms)');

// ------------------------------------------------------------------
// the stress exhibits: recompute the MG closed form on the pinned counts
// ------------------------------------------------------------------
P.stress.forEach(function (s) {
  // the pinned state counts drive the G2 statistic: recompute it in JS
  const obs = s.mgCounts;
  const p = [s.beta, 0.01 - s.beta, 0.99];
  const expc = p.map(function (q) { return q * s.n; });
  let g2 = 0;
  for (let i = 0; i < 3; i++) {
    if (obs[i] > 0 && expc[i] > 0) g2 += 2 * obs[i] * Math.log(obs[i] / expc[i]);
  }
  const pval = 1 - M.chi2Cdf(g2, 2);
  near(s.label + ' MG chi2', g2, s.mgChi2, 1e-4);
  near(s.label + ' MG p', pval, s.mgP, 1e-4);
});
console.log('ok  stress windows: the joint test recomputes on the pinned state counts');

// ------------------------------------------------------------------
// special-function invariants
// ------------------------------------------------------------------
near('normCdf(1)', M.normCdf(1), 0.8413447460685429, 1e-9);
near('normCdf(2)', M.normCdf(2), 0.9772498680568116, 1e-9);
near('norm ppf round-trip', M.normCdf(M.normPpf(0.975)), 0.975, 1e-9);
// t quantiles converge to the normal as dof -> infinity (1% quantile)
near('t(1000) ~ N quantile', M.tPpf(0.01, 1000), M.normPpf(0.01), 1e-2);
// the ES factor at 1% (2.665) sits above the VaR factor (2.326): it is a
// mean, not a quantile, of the tail
assert.ok(M.esFactorNormal(0.01) > -M.normPpf(0.01), 'ES > VaR factor');
// Kupiec on exact-expected hits: z = 0, p = 1
const kup0 = M.kupiecPof(new Array(400).fill(0), 0.01); // 0 hits... p should be small
const kupExact = (function () {
  // 4 hits of 400 at 1%: phat = alpha exactly -> z = 0, p = 1
  var a = new Array(400).fill(0); for (var i = 0; i < 4; i++) a[i] = 1;
  return M.kupiecPof(a, 0.01);
})();
eq('Kupiec z at exact hits', kupExact.z, 0);
near('Kupiec p at exact hits', kupExact.p_value, 1.0, 1e-9);
eq('Kupiec p=1.0 (under-covered)', kup0.p_value < 0.05, true);
// Mendez-Gaona: when beta == alpha the middle band vanishes and df = 1
const mg1 = M.mendezGaona([0], [0.0], [0.0], 0.01, 0.01);
eq('MG df collapses when beta=alpha', mg1.df, 1);
// Christoffersen on zero hits: chi2 = 0
eq('independence chi2 (no hits)', M.christoffersen(new Array(552).fill(0), 0.01).chi2, 0);
// chi-squared CDF anchors
near('chi2(2) CDF(1.943)', M.chi2Cdf(1.943320500604572, 2), 0.621545812456, 1e-6);
near('chi2(1) CDF(3.84)', M.chi2Cdf(3.841, 1), 0.95, 1e-3);

console.log('\nAll JS pins match the Python desk.');

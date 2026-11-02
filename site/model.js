(function (root, factory) {
  if (typeof module === 'object' && module.exports) { module.exports = factory(); }
  else { root.GDModel = factory(); }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ============================================================
  // special functions: gamma, normal, Student-t (mirrors scipy)
  // ============================================================

  // Numerical Recipes 6-term gamma (matches scipy gammaln to ~1e-11)
  var GCOF = [76.18009172947146, -86.50532032841077, 24.01409824023081,
    -1.231739572450155, 0.120865097383015e-2, -0.5395239304263e-5];
  function logGamma(x) {
    if (x < 0.5) {
      // reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x)
      return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
    }
    var y = x, tmp = y + 5.5, ser = 1.0, i;
    for (i = 0; i < 6; i++) ser += GCOF[i] / (y + 1 + i);
    tmp -= (y + 0.5) * Math.log(tmp);
    return -tmp + Math.log(2.50662827465 * ser / y);
  }
  function gammaFn(x) { return Math.exp(logGamma(x)); }

  function erf(x) {
    // erf(x) = 2/sqrt(pi) * integral_0^x exp(-t^2) dt, by composite
    // Simpson (step 0.005) - accurate to ~1e-14 over the range we use.
    if (x === 0) return 0;
    var sign = x < 0 ? -1 : 1; x = Math.abs(x);
    if (x > 8) return sign; // tail beyond 8 sigma is ~1e-15
    var upper = x, N = Math.ceil(upper / 0.005);
    if (N % 2) N++;
    var h = upper / N;
    var s = Math.exp(-upper * upper), i; // f(upper)
    for (i = 1; i < N; i++) s += (i % 2 ? 4 : 2) * Math.exp(-(upper - i * h) * (upper - i * h));
    s += 1; // f(0) = 1
    s *= h / 3;
    return sign * (2 / Math.sqrt(Math.PI)) * s;
  }
  function normPdf(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
  function normCdf(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
  function normPpf(p) {
    // bisection on the CDF (p in (0,1)); robust and adequate for the grid
    var lo = -50, hi = 50;
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    for (var i = 0; i < 200; i++) {
      var mid = 0.5 * (lo + hi);
      if (normCdf(mid) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // regularized incomplete beta I_x(a,b), via continued fraction
  function betacf(a, b, x) {
    var MAXIT = 200, EPS = 3e-14, FPMIN = 1e-300;
    var qab = a + b, qap = a + 1, qam = a - 1;
    var c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d; var h = d;
    for (var m = 1; m <= MAXIT; m++) {
      var m2 = 2 * m;
      var aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; var del = d * c; h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function ibeta(a, b, x) {
    if (x <= 0) return 0; if (x >= 1) return 1;
    // front = x^a (1-x)^b / B(a,b); ln B(a,b) = lnG(a) + lnG(b) - lnG(a+b)
    var lnFront = logGamma(a + b) - logGamma(a) - logGamma(b) +
                  a * Math.log(x) + b * Math.log(1 - x);
    var front = Math.exp(lnFront);
    if (x < (a + 1) / (a + b + 2)) return front * betacf(a, b, x) / a;
    return 1 - front * betacf(b, a, 1 - x) / b;
  }

  function tPdf(z, nu) {
    return gammaFn((nu + 1) / 2) / (Math.sqrt(nu * Math.PI) * gammaFn(nu / 2)) *
           Math.pow(1 + z * z / nu, -(nu + 1) / 2);
  }
  function tCdf(z, nu) {
    var x = nu / (nu + z * z);
    var p = 0.5 * ibeta(nu / 2, 0.5, x);
    return z > 0 ? 1 - p : p;
  }
  function tPpf(p, nu) {
    if (p <= 0) return -Infinity; if (p >= 1) return Infinity;
    var lo = -1000, hi = 1000;
    for (var i = 0; i < 200; i++) {
      var mid = 0.5 * (lo + hi);
      if (tCdf(mid, nu) < p) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
  }

  // ============================================================
  // the model's closed forms (mirror garch_var/risk)
  // ============================================================

  function esFactorNormal(alpha) {
    var z = normPpf(alpha); // < 0
    return normPdf(z) / alpha;
  }
  function esFactorT(nu, alpha) {
    var z = tPpf(alpha, nu); // < 0
    return tPdf(z, nu) * (nu + z * z) / ((nu - 1) * alpha);
  }
  // model-implied beta = P(loss > ES)
  function esViolationNormal(alpha) { var f = esFactorNormal(alpha); return 1 - normCdf(f); }
  function esViolationT(nu, alpha) { var f = esFactorT(nu, alpha); return 1 - tCdf(f, nu); }

  function kupiecPof(hitsArr, alpha) {
    var T = hitsArr.length;
    var s = 0; for (var i = 0; i < T; i++) s += hitsArr[i];
    var phat = s / T;
    var se = Math.sqrt(alpha * (1 - alpha) / T);
    var z = (phat - alpha) / se;
    var pval = 2 * (1 - normCdf(Math.abs(z)));
    return { T: T, hits: s, hit_rate: phat, alpha: alpha, z: z,
             p_value: pval, rejected_5pct: pval < 0.05 };
  }

  function christoffersen(hitsArr, alpha) {
    var T = hitsArr.length;
    if (T < 2) return { chi2: NaN, p_value: NaN, rejected_5pct: false };
    var hh = 0, h0 = 0, hh0 = 0, h00 = 0;
    for (var i = 0; i < T - 1; i++) {
      var p = hitsArr[i], q = hitsArr[i + 1];
      if (p === 1 && q === 1) hh++;
      else if (p === 1) h0++;
      else if (q === 1) hh0++;
      else h00++;
    }
    var pHitY = 0; for (var j = 0; j < T - 1; j++) pHitY += hitsArr[j];
    pHitY /= (T - 1);
    var pXgY1 = hh / Math.max(hh + h0, 1);
    var pXgY0 = hh0 / Math.max(hh0 + h00, 1);
    var Tm1 = T - 1;
    var exp = [
      pHitY * pXgY1 * Tm1,
      pHitY * (1 - pXgY1) * Tm1,
      (1 - pHitY) * pXgY0 * Tm1,
      (1 - pHitY) * (1 - pXgY0) * Tm1
    ];
    var obs = [hh, h0, hh0, h00];
    var chi2 = 0;
    for (var k = 0; k < 4; k++) {
      if (exp[k] > 0) { var d = (obs[k] - exp[k]) / Math.sqrt(exp[k]); chi2 += d * d; }
    }
    var pval = 1 - chi2Cdf(chi2, 1);
    return { chi2: chi2, p_value: pval, rejected_5pct: pval < 0.05 };
  }

  // chi2 survival for the dfs the desk uses (df=1 closed form, df=2 exp form;
  // general case via the regularized lower incomplete gamma)
  function chi2Cdf(x, df) {
    if (x <= 0) return 0;
    if (df === 1) {
      // P(|Z| <= sqrt(x)) = 2*normCdf(sqrt(x)) - 1
      return 2 * normCdf(Math.sqrt(x)) - 1;
    }
    if (df === 2) return 1 - Math.exp(-x / 2);
    return gammaLower(df / 2, x / 2) / gammaFn(df / 2);
  }
  // regularized lower incomplete gamma P(a, x), by series for x < a+1 and
  // the complementary relation otherwise (x never gets large in the desk)
  function gammaLower(a, x) {
    if (x < a + 1) {
      var sum = 1 / a, term = 1 / a, n = 1;
      for (; n < 500; n++) {
        term *= x / (a + n); sum += term;
        if (Math.abs(term) < 1e-15 * Math.abs(sum)) break;
      }
      return sum * Math.exp(-x + a * Math.log(x) - logGamma(a));
    }
    // continued fraction for Q(a, x), then P = 1 - Q
    var b = x + 1 - a, c = 1e-30, d = 1 / b, h = d, i, q, an;
    for (i = 2; i < 300; i++) {
      an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-30) d = 1e-30;
      c = b + an / c; if (Math.abs(c) < 1e-30) c = 1e-30;
      d = 1 / d; q = d * c; h *= q;
      if (Math.abs(q - 1) < 1e-14) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - logGamma(a)) * h;
  }

  function hitTimes(returns, bounds) {
    // returns aligned with bounds; hit where return < -bound (bound = positive loss)
    var out = [], n = returns.length;
    for (var i = 0; i < n; i++) out.push(returns[i] < -bounds[i] ? 1 : 0);
    return out;
  }

  // rolling alpha-quantile, linear interpolation (mirrors pandas rolling quantile)
  function rollingQuantile(arr, window, minPeriods, alpha) {
    var n = arr.length, out = new Array(n);
    for (var i = 0; i < n; i++) {
      var start = Math.max(0, i - window + 1);
      var len = i - start + 1;
      if (len < minPeriods) { out[i] = NaN; continue; }
      var seg = arr.slice(start, i + 1).slice().sort(function (a, b) { return a - b; });
      out[i] = linearQuantile(seg, alpha);
    }
    return out;
  }
  function linearQuantile(sorted, q) {
    var n = sorted.length;
    if (n === 1) return sorted[0];
    var pos = (n - 1) * q;
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
  }

  function mendezGaona(returns, varB, esB, alpha, beta) {
    var I = 0, J = 0, n = returns.length;
    for (var i = 0; i < n; i++) {
      var vi = returns[i] < -varB[i] ? 1 : 0;
      var ei = returns[i] < -esB[i] ? 1 : 0;
      if (ei) J++; else if (vi) I++;
    }
    var cBeyond = J, cBand = I, cClean = n - cBeyond - cBand;
    var obs = [cBeyond, cBand, cClean];
    var p = [beta, alpha - beta, 1 - alpha];
    var keep = [true, p[1] > 0, p[2] > 0];
    var df = Math.max(keep.filter(Boolean).length - 1, 1);
    var g2 = 0;
    for (var k = 0; k < 3; k++) {
      if (!keep[k]) continue;
      var expK = p[k] * n;
      if (obs[k] > 0 && expK > 0) g2 += 2 * obs[k] * Math.log(obs[k] / expK);
    }
    var pval = 1 - chi2Cdf(g2, df);
    return { n: n, states: keep.filter(Boolean).length,
             counts: [cBeyond, cBand, cClean], alpha: alpha, beta: beta,
             chi2: g2, df: df, p_value: pval, rejected_at_5pct: pval < 0.05 };
  }

  // deflated Sharpe (Bailey & Lopez de Prado) on pinned PnL moments
  function deflatedSharpe(inp, nTrials) {
    var T = inp.T, sr = inp.sr_pp, g3 = inp.skew, g4 = inp.kurt;
    var m = Math.max(nTrials, 2);
    var sr_zero = Math.sqrt(2 * Math.log(m) / (T - 1));
    var varTerm = 1 - g3 * sr + 0.25 * (g4 - 1) * sr * sr;
    if (varTerm <= 0) return { error: 'negative variance term' };
    var stat = (sr - sr_zero) * Math.sqrt(T - 1) / Math.sqrt(varTerm);
    var pval = 1 - normCdf(stat);
    return { sharpe_per_period: sr, n_trials: m, sr_zero: sr_zero,
             skew: g3, kurtosis: g4, statistic: stat, p_value: pval,
             significant_at_5pct: pval < 0.05 };
  }

  // ============================================================
  // exhibit builders: compose the primitives over pinned data
  // ============================================================

  // GARCH VaR / ES series at one level, from the pinned conditional-vol path
  // dist: 't' (GARCH-t, the desk default, on the t-fit vol path) or
  // 'n' (Gaussian GARCH, on the normal-fit vol path)
  function garchSeries(P, level, dist) {
    var F = P.factors[level];
    var hT = (dist === 'n' && P.window.hN) ? P.window.hN : P.window.hT;
    var z = (dist === 'n') ? F.zN : F.zT;
    var out = [];
    for (var i = 0; i < hT.length; i++) { out.push(z * hT[i]); }
    return out;
  }

  function garchBacktest(P, level, dist) {
    var F = P.factors[level];
    var varB = garchSeries(P, level, dist);
    var hits = hitTimes(P.window.spy, varB);
    return {
      varBounds: varB,
      hits: hits,
      hitsSum: hits.reduce(function (a, b) { return a + b; }, 0),
      kupiec: kupiecPof(hits, F.alpha),
      indep: christoffersen(hits, F.alpha)
    };
  }

  function histBacktest(P, level) {
    var F = P.factors[level];
    var bounds = P.window.hist[level];
    var hits = [], r = P.window.spy, s = 0;
    for (var i = 0; i < r.length; i++) {
      var h = (r[i] < -bounds[i]) ? 1 : 0;
      hits.push(h); s += h;
    }
    return {
      hits: hits,
      hitsSum: s,
      kupiec: kupiecPof(hits, F.alpha),
      indep: christoffersen(hits, F.alpha)
    };
  }

  function hitsFromLevels(returns, esB) {
    var out = [], n = returns.length;
    for (var i = 0; i < n; i++) out.push(returns[i] < -esB[i] ? 1 : 0);
    return out;
  }

  function esExhibit(P, level, dist) {
    var F = P.factors[level];
    var zVar = (dist === 'n') ? F.zN : F.zT;
    var esF = (dist === 'n') ? F.esN : F.esT;
    var beta = (dist === 'n') ? F.betaN : F.betaT;
    var varB = garchSeries(P, level, dist);
    var esB = varB.map(function (v) {
      return v * esF / zVar; // same h_t: ES = esF*h_t = (esF/zVar) * (zVar*h_t)
    });
    var r = P.window.spy, n = r.length, I = 0, J = 0, lossSum = 0;
    var esHitsArr = hitsFromLevels(r, esB);
    for (var i = 0; i < n; i++) {
      var vi = r[i] < -varB[i] ? 1 : 0;
      if (esHitsArr[i]) { J++; } else if (vi) { I++; }
      if (vi) lossSum += -r[i];
    }
    return {
      esHits: J,
      varHits: I + J,
      // Kupiec POF of the ES indicator against beta (the model-implied ES breach rate)
      coverage: kupiecPof(esHitsArr, beta),
      mendez: mendezGaona(r, varB, esB, F.alpha, beta),
      meanLossBeyondVar: I + J > 0 ? lossSum / (I + J) : NaN,
      esLevelMean: esB.reduce(function (a, b) { return a + b; }, 0) / n
    };
  }

  // ============================================================
  // overlay re-simulation: mirrors garch_var/portfolio/{vol_target,
  // risk_parity}.py on the pinned panel, so the target-vol control
  // recomputes the arms live instead of reading a static pin.
  // ============================================================

  // trailing 126-day std, ddof=1, min_periods=60 (pandas rolling defaults)
  function rollingStd(arr, window, minPeriods) {
    var n = arr.length, out = new Array(n), i, j, s, cnt, mean, v, d;
    s = 0;
    for (i = 0; i < n; i++) {
      s += arr[i];
      if (i >= window) s -= arr[i - window];
      cnt = Math.min(i + 1, window);
      if (cnt >= minPeriods) {
        mean = s / cnt; v = 0;
        for (j = i - cnt + 1; j <= i; j++) { d = arr[j] - mean; v += d * d; }
        out[i] = Math.sqrt(v / (cnt - 1)) * Math.sqrt(252); // annualised
      } else out[i] = NaN;
    }
    return out;
  }

  // rolling mean that skips NaN (pandas rolling.mean behaviour)
  function rollingMean(arr, window, minPeriods) {
    var n = arr.length, out = new Array(n), i, j, from, s, cnt;
    for (i = 0; i < n; i++) {
      from = Math.max(0, i - window + 1); s = 0; cnt = 0;
      for (j = from; j <= i; j++) {
        if (!isNaN(arr[j])) { s += arr[j]; cnt++; }
      }
      out[i] = cnt >= minPeriods ? s / cnt : NaN;
    }
    return out;
  }

  // last-observed-day-per-calendar-month flags (pandas groupby Period("M").max())
  function monthEndFlags(dates) {
    var out = new Array(dates.length), i;
    for (i = 0; i < dates.length; i++) {
      out[i] = i === dates.length - 1 ||
        Math.floor(dates[i] / 100) !== Math.floor(dates[i + 1] / 100);
    }
    return out;
  }

  // the repo's _metrics: cumprod equity, annualised return / vol / sharpe / DD
  function pnlMetrics(pnl) {
    var n = pnl.length, eq = 1.0, peak = 1.0, dd = 0.0, sum = 0.0, i, draw, dx;
    for (i = 0; i < n; i++) {
      eq *= 1.0 + pnl[i];
      if (eq > peak) peak = eq;
      draw = eq / peak - 1.0;
      if (draw < dd) dd = draw;
      sum += pnl[i];
    }
    var mean = sum / n, varr = 0.0;
    for (i = 0; i < n; i++) { dx = pnl[i] - mean; varr += dx * dx; }
    var std = Math.sqrt(varr / (n - 1));
    var annRet = Math.pow(eq, 252.0 / n) - 1.0;
    var vol = std * Math.sqrt(252);
    return {
      annualised_return: annRet,
      volatility: vol,
      sharpe: vol > 0 ? annRet / vol : NaN,
      max_drawdown: -dd,
      final_equity: eq
    };
  }

  var _cache = {};
  function panelStats(P) {
    // the rolling stats are target-vol independent, so compute them once
    if (!_cache.stats) {
      var eq = P.panel.spy, dur = P.panel.tlt;
      var eqVol = rollingStd(eq, 126, 60), durVol = rollingStd(dur, 126, 60);
      var eqM = rollingMean(eq, 126, 60), durM = rollingMean(dur, 126, 60);
      var prod = eq.map(function (e, i) {
        return isNaN(eqM[i]) || isNaN(durM[i]) ? NaN :
               (e - eqM[i]) * (dur[i] - durM[i]);
      });
      var cov = rollingMean(prod, 126, 60).map(function (v) {
        return isNaN(v) ? NaN : v * 252;
      });
      var start = 0;
      while (start < eqVol.length && !isFinite(eqVol[start])) start++;
      _cache.stats = {
        eqVol: eqVol, durVol: durVol, cov: cov, start: start,
        rebal: monthEndFlags(P.dates)
      };
    }
    return _cache.stats;
  }

  // single-asset vol-target vs the static 50/50 benchmark
  // (mirror vol_target_portfolio): the basket in force on day t was set at the
  // close of the last rebalance strictly before t; cost hits the rebalance day
  function volTargetSim(P, targetVol, costBps) {
    var s = panelStats(P);
    var eq = P.panel.spy, rf = P.panel.rf, n = eq.length;
    var wCur = null, benchBuilt = false;
    var pnl = [], pnlB = [];
    for (var i = s.start; i < n; i++) {
      var wEff = wCur === null ? 0.0 : wCur;
      var bEff = benchBuilt ? 0.5 : 0.0;
      var r = wEff * eq[i] + (1.0 - wEff) * rf[i];
      var rb = bEff * eq[i] + (1.0 - bEff) * rf[i];
      var cost = 0.0, costB = 0.0;
      if (s.rebal[i]) {
        var tv = s.eqVol[i];
        var newW = (isFinite(tv) && tv > 0) ? Math.min(1.0, targetVol / tv) : 1.0;
        cost = Math.abs(newW - wEff) * costBps / 10000.0;
        wCur = newW;
        if (!benchBuilt) { costB = 0.5 * costBps / 10000.0; benchBuilt = true; }
      }
      pnl.push(r - cost);
      pnlB.push(rb - costB);
    }
    return {
      pnl: pnl,
      benchmarkPnl: pnlB,
      metrics: pnlMetrics(pnl),
      benchmarkMetrics: pnlMetrics(pnlB)
    };
  }

  // two-asset inverse-vol risk parity, scaled to a target annual vol
  // (mirror risk_parity_portfolio): the basket is levered up to at most 2x
  function riskParitySim(P, targetVol, costBps) {
    var s = panelStats(P);
    var eq = P.panel.spy, dur = P.panel.tlt, n = eq.length;
    var wEq = 0.0, wDur = 0.0, active = false, totalTurn = 0.0;
    var pnl = [];
    for (var i = s.start; i < n; i++) {
      var wEqEff = active ? wEq : 0.0;
      var wDurEff = active ? wDur : 0.0;
      var gross = wEqEff * eq[i] + wDurEff * dur[i];
      var cost = 0.0;
      if (s.rebal[i] && isFinite(s.eqVol[i]) && isFinite(s.durVol[i])) {
        var ve = s.eqVol[i], vd = s.durVol[i];
        var cv = isFinite(s.cov[i]) ? s.cov[i] : 0.0;
        if (ve > 0 && vd > 0) {
          var invE = 1.0 / ve, invD = 1.0 / vd;
          var rawE = invE / (invE + invD), rawD = 1.0 - rawE;
          var v = rawE * rawE * ve * ve + rawD * rawD * vd * vd +
                  2 * rawE * rawD * cv;
          var combo = Math.sqrt(Math.max(v, 1e-12));
          var scale = Math.min(2.0, targetVol / combo);
          var nEq = rawE * scale, nDur = rawD * scale;
          totalTurn += Math.abs(nEq - wEq) + Math.abs(nDur - wDur);
          wEq = nEq; wDur = nDur; active = true;
          cost = (Math.abs(nEq - wEqEff) + Math.abs(nDur - wDurEff)) *
                 costBps / 10000.0;
        }
      }
      pnl.push(gross - cost);
    }
    var m = pnlMetrics(pnl);
    m.avg_turnover = totalTurn / Math.max(n - s.start, 1);
    return { pnl: pnl, metrics: m };
  }

  // ------------------------------------------------------------

  return {
    logGamma: logGamma, normPdf: normPdf, normCdf: normCdf, normPpf: normPpf,
    tPdf: tPdf, tCdf: tCdf, tPpf: tPpf, ibeta: ibeta,
    esFactorNormal: esFactorNormal, esFactorT: esFactorT,
    esViolationNormal: esViolationNormal, esViolationT: esViolationT,
    kupiecPof: kupiecPof, christoffersen: christoffersen, mendezGaona: mendezGaona,
    hitTimes: hitTimes, rollingQuantile: rollingQuantile,
    deflatedSharpe: deflatedSharpe, chi2Cdf: chi2Cdf,
    garchBacktest: garchBacktest, histBacktest: histBacktest,
    esExhibit: esExhibit,
    volTargetSim: volTargetSim, riskParitySim: riskParitySim, pnlMetrics: pnlMetrics
  };
}));

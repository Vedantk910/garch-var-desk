// site/app.js - the desk's glue: wires GDModel (site/model.js) and the
// pinned inputs (site/GARCHPINS, from site/pins.js) into the page.
// Every figure recomputes live off the pinned GARCH-t vol path; nothing
// here re-fits MLEs - the fits are the model's own and they ship pinned.
(function () {
  'use strict';

  var P = window.GARCHPINS;
  var M = window.GDModel;
  if (!P || !M) {
    var s0 = document.getElementById('status');
    if (s0) s0.textContent = 'Failed to load the desk (missing pins or model script).';
    return;
  }

  var ref = P.ref;
  var meta = P.meta;

  function $(id) { return document.getElementById(id); }
  function set(id, html) { $(id).innerHTML = html; }

  // bps level -> nominal tail percent, e.g. "100" -> "99"
  function pctOf(lv) { return (100 - Number(lv) / 100).toString(); }

  function fmt(x, d) {
    if (x === null || x === undefined || !isFinite(x)) return '—';
    return Number(x).toFixed(d == null ? 2 : d);
  }
  function pct(x, d) { return isFinite(x) ? (100 * x).toFixed(d == null ? 1 : d) + '%' : '—'; }
  function pctPlain(x, d) { return isFinite(x) ? (100 * x).toFixed(d == null ? 1 : d) : '—'; }
  function pfmt(x) { return isFinite(x) ? (x < 0.001 ? x.toExponential(1) : x.toFixed(3)) : '—'; }

  // one canonical state; render() redraws everything from it.
  // The desk's defaults are the hero's numbers: GARCH-t, the 99% level
  // ("100" bps), 3 trials, the 15% overlay target.
  var S = { dist: 't', level: '100', m: meta.nTrials, tv: meta.targetVols[1] };

  var statusTimer;
  function status(msg) {
    var s = $('status');
    s.textContent = msg;
    s.style.opacity = '1';
    s.style.display = '';
    clearTimeout(statusTimer);
    statusTimer = setTimeout(function () { s.style.display = 'none'; }, 1800);
  }

  // ----------------------------------------------------------------------
  // controls
  // ----------------------------------------------------------------------

  function buildControls() {
    var lv = $('inLevel');
    meta.levels.forEach(function (l) {
      var o = document.createElement('option');
      o.value = l; o.textContent = pctOf(l) + '% tail';
      lv.appendChild(o);
    });
    lv.value = S.level;
    lv.addEventListener('change', function () { S.level = lv.value; render(); });

    var tv = $('inTv');
    meta.targetVols.forEach(function (t) {
      var o = document.createElement('option');
      o.value = t; o.textContent = (100 * t).toFixed(0) + '% target';
      tv.appendChild(o);
    });
    tv.value = String(S.tv);
    tv.addEventListener('change', function () { S.tv = parseFloat(tv.value); render(); });

    document.querySelectorAll('#modelBtns button').forEach(function (b) {
      b.addEventListener('click', function () {
        S.dist = b.getAttribute('data-dist');
        markOn('modelBtns', b);
        render();
        status(b.getAttribute('data-dist') === 't'
          ? 'GARCH-t - the honest conditional on its pinned vol path.'
          : 'Normal GARCH - the tail collapses to what the Gaussian says: watch the ES gap and the verdict.');
      });
    });

    document.querySelectorAll('#mBtns button').forEach(function (b) {
      b.addEventListener('click', function () {
        S.m = parseInt(b.getAttribute('data-m'), 10);
        markOn('mBtns', b);
        render();
        status('M = ' + S.m + ' trials - the deflated Sharpe re-prices both 15% arms.');
      });
    });

    markOn('modelBtns', document.querySelector('#modelBtns [data-dist="' + S.dist + '"]'));
    markOn('mBtns', document.querySelector('#mBtns [data-m="' + S.m + '"]'));
  }

  function markOn(containerId, btn) {
    document.querySelectorAll('#' + containerId + ' button').forEach(function (b) {
      b.classList.toggle('on', b === btn);
    });
  }

  // ----------------------------------------------------------------------
  // renderers - each reads P + M and writes its slice of the DOM
  // ----------------------------------------------------------------------

  function renderHero() {
    // the hero number is the t-fit ES level mean at the selected tail; the
    // copy under it re-prices the normal-GARCH counterfactual at the same
    // tail so the two never disagree
    var esT = M.esExhibit(P, S.level, 't').esLevelMean;
    var esN = M.esExhibit(P, S.level, 'n').esLevelMean;
    set('heroEs', pctPlain(esT, 1));
    set('heroTail', 'the ' + pctOf(S.level) + '% one-day Expected Shortfall. Flip the model to a normal GARCH and it collapses to ' +
      pctPlain(esN, 1) + '% - that mispricing is why the desk runs GARCH-t.');
  }

  function renderVerdict() {
    var bt = M.garchBacktest(P, S.level, S.dist);
    var es = M.esExhibit(P, S.level, S.dist);
    var ok = !bt.kupiec.rejected_5pct && !es.mendez.rejected_at_5pct;
    var modelLabel = S.dist === 't' ? 'GARCH-t' : 'Normal GARCH';
    var rec = $('rec');
    rec.textContent = modelLabel + ' · ' + pctOf(S.level) + '% - ' + (ok ? 'level holds' : 'coverage fails');
    rec.classList.toggle('rec-t', ok);
    rec.classList.toggle('rec-n', !ok);
    set('recLine', ok
      ? 'Neither the Kupiec POF nor the joint Mendez-Gaona VaR/ES test rejects at 5% - the level is honest for this model, at this tail.'
      : 'A 5% backtest rejects: at this model and tail the loss is under-priced, and the desk says so out loud.');
    set('dHits', bt.hitsSum + ' of ' + P.window.n);
    set('dEsHits', es.esHits + ' of ' + P.window.n);
    set('dMgP', pfmt(es.mendez.p_value));
    var esTmean = M.esExhibit(P, S.level, 't').esLevelMean;
    var esNmean = M.esExhibit(P, S.level, 'n').esLevelMean;
    set('dEsT', pct(esTmean));
    set('dEsN', pct(esNmean));
  }

  function renderDeriv() {
    var bt = M.garchBacktest(P, S.level, S.dist);
    var es = M.esExhibit(P, S.level, S.dist);
    var mt = P.models.garchT, mn = P.models.garchNormal, F = P.factors[S.level];
    set('dPersist', fmt(mt.persistence, 3));
    set('dAic', fmt(mt.aicc, 0));
    set('dAicN', fmt(mn.aicc, 0));
    set('dLevel', pctOf(S.level));
    set('dHits2', String(bt.hitsSum));
    set('dN', String(P.window.n));
    set('dKupP', pfmt(bt.kupiec.p_value));
    set('dKupVerdict', bt.kupiec.rejected_5pct ? 'POF rejects' : 'POF holds');
    set('dBeta', pct(F[S.dist === 'n' ? 'betaN' : 'betaT'], 4));
    set('dMg', fmt(es.mendez.chi2, 2));
    set('dMgDf', String(es.mendez.df));
    set('dMgP2', pfmt(es.mendez.p_value));
    set('dMgVerdict', es.mendez.rejected_at_5pct ? 'joint test rejects' : 'joint test holds');
    set('dModel', S.dist === 't' ? 'GARCH-t' : 'Normal GARCH');
    set('dLevel2', pctOf(S.level) + '% level');
    set('dRec', (!bt.kupiec.rejected_5pct && !es.mendez.rejected_at_5pct)
      ? 'held by both tests' : 'rejected by a backtest');
  }

  function renderVarTable() {
    var tbody = $('toolTable').querySelector('tbody');
    tbody.innerHTML = '';
    var rows = [
      { name: 'GARCH-t (pinned)', dist: 't' },
      { name: 'Normal GARCH (pinned)', dist: 'n' },
      { name: 'Historical (rolling 252)', dist: null }
    ];
    rows.forEach(function (r) {
      var bt = r.dist ? M.garchBacktest(P, S.level, r.dist) : M.histBacktest(P, S.level);
      var tr = document.createElement('tr');
      if (r.dist) {
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', function () {
          S.dist = r.dist;
          markOn('modelBtns', document.querySelector('#modelBtns [data-dist="' + r.dist + '"]'));
          render();
        });
      }
      tr.innerHTML =
        '<td><b>' + r.name + '</b>' +
          (r.dist === S.dist ? ' <span class="note">· in force</span>' : '') + '</td>' +
        '<td class="n">' + bt.hitsSum + '</td>' +
        '<td class="n">' + pfmt(bt.kupiec.p_value) + '</td>' +
        '<td class="n">' + pfmt(bt.indep.p_value) + '</td>' +
        '<td class="n">' + (bt.kupiec.rejected_5pct ? 'No' : 'Yes') + '</td>';
      tbody.appendChild(tr);
    });
  }

  function renderEsTable() {
    var es = M.esExhibit(P, S.level, S.dist);
    var F = P.factors[S.level];
    set('esModelName', S.dist === 't' ? 'GARCH-t' : 'Normal GARCH');
    var rows = [
      ['ES level (mean over the window)', pct(es.esLevelMean)],
      ['ES breaches', es.esHits + ' of ' + P.window.n],
      ['VaR breaches', es.varHits + ' of ' + P.window.n],
      ['Implied breach rate β', pct(F[S.dist === 'n' ? 'betaN' : 'betaT'], 4)],
      ['ES-coverage POF p', pfmt(es.coverage.p_value) + (es.coverage.rejected_5pct ? ' · rejects' : ' · holds')],
      ['Joint MG χ²', fmt(es.mendez.chi2, 2) + ' (df ' + es.mendez.df + ')'],
      ['Joint p', pfmt(es.mendez.p_value) + (es.mendez.rejected_at_5pct ? ' · rejects' : ' · holds')],
      ['Mean loss beyond VaR', pct(es.meanLossBeyondVar)]
    ];
    $('esTable').querySelector('tbody').innerHTML = rows.map(function (r) {
      return '<tr><td>' + r[0] + '</td><td class="n">' + r[1] + '</td></tr>';
    }).join('');
  }

  function renderOverlay() {
    var tv = S.tv;
    var vt = M.volTargetSim(P, tv, meta.costBps);
    var bench = M.pnlMetrics(vt.benchmarkPnl);
    var rps = meta.targetVols.map(function (t) { return M.riskParitySim(P, t, meta.costBps); });

    var arms = [
      { name: (100 * tv).toFixed(0) + '% vol-target (SPY/T-bill)', m: vt.metrics, turn: null, sel: true },
      { name: '50/50 benchmark', m: bench, turn: null, sel: false },
      { name: 'RP @ ' + (100 * meta.targetVols[0]) + '%', m: rps[0].metrics, turn: rps[0].metrics.avg_turnover, sel: tv === meta.targetVols[0] },
      { name: 'RP @ ' + (100 * meta.targetVols[1]) + '%', m: rps[1].metrics, turn: rps[1].metrics.avg_turnover, sel: tv === meta.targetVols[1] },
      { name: 'RP @ ' + (100 * meta.targetVols[2]) + '%', m: rps[2].metrics, turn: rps[2].metrics.avg_turnover, sel: tv === meta.targetVols[2] }
    ];
    $('ovTable').querySelector('tbody').innerHTML = arms.map(function (a) {
      var m = a.m;
      return '<tr class="' + (a.sel ? 'sel' : '') + '">' +
        '<td><b>' + a.name + '</b>' + (a.sel ? ' <span class="note">· selected</span>' : '') + '</td>' +
        '<td class="n">' + pct(m.annualised_return) + '</td>' +
        '<td class="n">' + pct(m.volatility) + '</td>' +
        '<td class="n">' + fmt(m.sharpe, 2) + '</td>' +
        '<td class="n">' + pct(m.max_drawdown) + '</td>' +
        '<td class="n">' + (a.turn != null ? fmt(a.turn * 100, 2) + '%' : '—') + '</td></tr>';
    }).join('');

    drawCurves();
    set('dTv', (100 * tv).toFixed(0));
    set('dLever', fmt(M.riskParitySim(P, tv, meta.costBps).metrics.final_gross, 2));
  }

  // pinned monthly equity curves; legend + axes in the house style
  function drawCurves() {
    var host = $('overlayChart');
    var W = 660, H = 250, pad = { l: 40, r: 10, t: 10, b: 22 };
    var series = [
      { key: 'volTarget', color: '#0a0a0a', w: 2.5, label: 'vol-target ' + (100 * S.tv).toFixed(0) + '%' },
      { key: 'benchmark', color: '#8e8e93', w: 1.5, label: '50/50' },
      { key: 'risk_parity_10pct', color: '#0071e3', w: 1.5, label: 'RP 10%' },
      { key: 'risk_parity_15pct', color: '#5e5ce6', w: 1.5, label: 'RP 15%' },
      { key: 'risk_parity_20pct', color: '#0aa67c', w: 1.5, label: 'RP 20%' }
    ];
    var all = [];
    series.forEach(function (s) { P.curves[s.key].forEach(function (pt) { all.push(pt[1]); }); });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    if (hi - lo < 1e-9) hi = lo + 1e-9;
    var n = P.curves.volTarget.length;
    function X(i) { return pad.l + (i / (n - 1)) * (W - pad.l - pad.r); }
    function Y(v) { return pad.t + (1 - (v - lo) / (hi - lo)) * (H - pad.t - pad.b); }

    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Overlay equity curves, monthly pinned">';
    for (var g = 0; g <= 4; g++) {
      var gv = lo + g / 4 * (hi - lo), gy = Y(gv);
      svg += '<line x1="' + pad.l + '" y1="' + gy + '" x2="' + (W - pad.r) + '" y2="' + gy + '" stroke="#d2d2d7" stroke-width="0.6"/>' +
        '<text x="' + (pad.l - 6) + '" y="' + (gy + 3.5) + '" text-anchor="end" font-size="10" fill="#86868b">' + fmt(gv, 1) + '</text>';
    }
    [0, Math.round(n / 4), Math.round(n / 2), Math.round(3 * n / 4), n - 1].forEach(function (i) {
      var d = P.dates[P.meta.pnlOffset + i] || P.dates[i];
      var yy = Math.floor(d / 10000);
      var mm = ('0' + Math.floor((d % 10000) / 100)).slice(-2);
      svg += '<text x="' + X(i) + '" y="' + (H - 4) + '" text-anchor="middle" font-size="10" fill="#86868b">' +
        yy + '-' + mm + '</text>';
    });
    series.forEach(function (s) {
      var pts = P.curves[s.key].map(function (pt, i) { return X(i) + ',' + Y(pt[1]); }).join(' ');
      svg += '<polyline fill="none" stroke="' + s.color + '" stroke-width="' + s.w + '" points="' + pts + '"/>';
    });
    svg += '</svg>';
    var leg = '<div class="leg">' + series.map(function (s) {
      return '<span><i style="background:' + s.color + '"></i>' + s.label + '</span>';
    }).join('') + '</div>';
    host.innerHTML = svg + leg;
  }

  function renderStress() {
    var prefix = { gfc_2008_09: 'sg', crash_2020_q1: 'sc' };
    P.stress.forEach(function (s) {
      var p = prefix[s.label];
      if (!p) return;
      set(p + 'Pers', fmt(s.persistenceRefit, 4));
      set(p + 'Global', fmt(s.persistenceGlobal, 4));
      set(p + 'Refit', fmt(s.persistenceRefit, 4));
      set(p + 'Worst', s.worstDay.date + ' · ' + s.worstDay.loss_pct.toFixed(1) + '%');
      set(p + 'Es', pct(s.esLevelMean));
      set(p + 'Real', pct(s.meanLossBeyondVar));
      set(p + 'P', pfmt(s.mgP));
      set(p + 'Note', s.powerNote || '');
    });
  }

  function renderInference() {
    // pinned 95% block-bootstrap CIs on the 15% arms; the deflated Sharpe
    // recomputes live at the chosen M from the pinned PnL moments
    var ciV = ref.ci.volTarget15pct, ciR = ref.ci.riskParity15pct;

    function ciBar(hostId, ci) {
      var lo = ci.ci_low, hi = ci.ci_high, pt = ci.point;
      var sLo = Math.min(0, lo) - 0.15, sHi = Math.max(0, hi) + 0.15;
      function pos(v) { return ((v - sLo) / (sHi - sLo)) * 100; }
      var host = $(hostId);
      host.querySelector('i').style.cssText =
        'position:absolute;top:0;bottom:0;border-radius:6px;left:' + pos(lo) +
        '%;width:' + Math.max(pos(hi) - pos(lo), 0.6) + '%;background:rgba(10,10,10,.16);';
      // reuse the point marker: one per bar, never accumulate on re-render
      var dot = host.querySelector('i.point');
      if (!dot) {
        dot = document.createElement('i');
        dot.className = 'point';
        host.appendChild(dot);
      }
      dot.style.cssText =
        'position:absolute;top:0;bottom:0;width:3px;background:#0a0a0a;z-index:1;left:' + pos(pt) + '%;';
      host.querySelector('.zero').style.left = pos(0) + '%';
      $(hostId + 'Lo').textContent = fmt(lo, 2);
      $(hostId + 'Hi').textContent = fmt(hi, 2);
    }
    ciBar('ciVt', ciV);
    ciBar('ciRp', ciR);

    set('shVt', fmt(ciV.point, 2));
    set('ciVtRange', '[' + fmt(ciV.ci_low, 2) + ', ' + fmt(ciV.ci_high, 2) + ']');
    set('shRp', fmt(ciR.point, 2));
    set('ciRpRange', '[' + fmt(ciR.ci_low, 2) + ', ' + fmt(ciR.ci_high, 2) + ']');

    var dV = M.deflatedSharpe(ref.dsr_inputs.volTarget15pct, S.m);
    var dR = M.deflatedSharpe(ref.dsr_inputs.riskParity15pct, S.m);
    function chip(id, d) {
      var el = $(id);
      el.textContent = 'p = ' + pfmt(d.p_value) + (d.significant_at_5pct ? ' · clears 5%' : ' · not at 5%');
      el.title = 'M=' + S.m + ' · statistic ' + fmt(d.statistic, 2) + ' · sr_zero ' + fmt(d.sr_zero, 4);
    }
    chip('dsrVt', dV);
    chip('dsrRp', dR);
  }

  function render() {
    renderHero();
    renderVerdict();
    renderDeriv();
    renderVarTable();
    renderEsTable();
    renderOverlay();
    renderStress();
    renderInference();
  }

  // ----------------------------------------------------------------------
  // page chrome: mobile menu + scroll reveals
  // ----------------------------------------------------------------------

  function chrome() {
    var menuBtn = $('menu'), links = $('links');
    if (menuBtn) {
      menuBtn.addEventListener('click', function () {
        var open = links.classList.toggle('open');
        menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    }
    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
        });
      }, { threshold: 0.12 });
      document.querySelectorAll('.reveal').forEach(function (el) { io.observe(el); });
    } else {
      document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('in'); });
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    buildControls();
    render();
    chrome();
    status('Desk loaded - ' + P.window.n + '-day backtest, every figure pinned to the Python model.');
  });
})();

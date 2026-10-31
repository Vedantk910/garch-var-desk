# garch-var-desk

A one-day 99% **VaR + Expected Shortfall** risk desk that names where it breaks. GARCH(1,1)-t / EGARCH forecasts on daily SPY, coverage backtests (Kupiec POF, Christoffersen independence, and the joint **Méndez–Gaona** VaR/ES test) against a decade of returns, stress re-estimation into the GFC 2008-09 and 2020-Q1 windows, a rolling SPY/TLT inverse-vol **risk-parity** overlay, and **bootstrap + deflated-Sharpe** CIs on every headline number.

## The honest numbers

- GARCH-t persistence 0.994 on 2015–2026 SPY; AIC 7334 vs 7540 Gaussian → heavy tails drive the model choice.
- 99% VaR: 7 of 552 hits (1.27%), Kupiec POF p = 0.53 — coverage not rejected.
- ES: mean 3.87%, Méndez–Gaona joint test χ² = 1.94 (df 2, p = 0.38) — level and intermediate band both in.
- Stress admission: GFC/2020 calm fits sit at the persistence boundary (1.0000 / 0.9876 → re-fit 1.0000 / 0.9995), so the model is **level-conservative but persistence-wrong** in a crash; expected 99% hits over 44–54 days ≈ 0.5, so non-rejection is *not* validation.
- Overlay: risk-parity @ 10/15/20% target vol is a monotone surface (higher vol → lower Sharpe, higher drawdown); the duration leg costs Sharpe over the equity-bull panel.
- Inference: 15% vol-target Sharpe 0.72 (boot CI [0.20, 1.37]); RP @ 15% Sharpe 0.56 (boot CI [-0.02, 1.14]). After the 3-trial deflation, **neither clears 5%** (p = 0.17 / 0.34) — the edge is real, not proven.

## What it does

1. **Data** — daily SPY + 8 mega-cap log returns, FRED risk-free (cache-first, offline).
2. **Model** — GARCH-t / EGARCH MLE via `arch`, HAR mean, one-step-ahead vol.
3. **Risk** — 99% VaR *and* ES; Kupiec POF, Christoffersen CC, and the joint Méndez–Gaona VaR/ES likelihood-ratio test.
4. **Stress** — re-fit persistence into the GFC 2008-09 and 2020-Q1 windows (cache-first, deterministic fallback), with an explicit "where the desk admits its limits" exhibit.
5. **Overlay** — rolling SPY/TLT inverse-vol risk-parity at three target vols, full Sharpe/drawdown/turnover surface.
6. **Inference** — block-bootstrap Sharpe CI + deflated Sharpe (multiple-trials corrected) on each arm.

Every number is out-of-sample relative to its window, reproducible by one command, and the desk states its limits instead of hiding them.

## Project layout

```
garch_var/
  data/       # series + cache-first loaders (incl. stress S&P cache)
  models/     # GARCH / EGARCH MLE wrappers
  risk/       # VaR + ES, Kupiec POF, Christoffersen CC, Méndez–Gaona, bootstrap/DSR
  portfolio/  # vol-target + rolling risk-parity overlay
  stress/     # 2008/2020 stress re-estimation (no look-ahead, iterated forecast)
run_desk.py   # end-to-end CLI, writes out/desk_numbers.json
report/       # risk_desk_note.md (one-pager)
tests/        # pytest suite (35 offline tests)
```

## Quick start

```bash
pip install -r requirements.txt
python run_desk.py          # writes out/desk_numbers.json
python -m pytest tests/ -q  # 35 offline tests
```

All figures in `report/risk_desk_note.md` are out-of-sample relative to their window and reproducible by the one command above.

## 7-gate trail (P2 elite upgrade)

- **G0 scope** — one decision: which risk overlay does the client actually get? VaR alone can't answer it.
- **G1 build** — ES (Méndez–Gaona), 2008/2020 stress re-estimation, rolling SPY/TLT risk-parity, bootstrap + deflated Sharpe; all seeded, cache-first.
- **G2 test** — 26 new offline tests (35 total); each new module has a no-look-ahead / convergence / determinism check.
- **G3 verify** — `python run_desk.py` regenerates `out/desk_numbers.json` with every quoted number; GFC data served from cache, no network needed.
- **G4 stress/ablation** — dropping the t-distribution returns VaR to the too-tight normal GARCH; the duration leg alone *costs* Sharpe, which is why the overlay is reported as a surface, not a point.
- **G5 document** — this README + the one-pager `report/risk_desk_note.md` with the honest limits.
- **G6 hostile pass** — the three attacks and the answers:

| Attack | Answer |
|---|---|
| "VaR is the 1990s answer — what's the Basel answer?" | ES (CVaR) at 99%, sub-additive, and backtested with the Méndez–Gaona joint VaR/ES LR test (χ² = 1.94, p = 0.38). |
| "GARCH vol over-reacts — prove the desk survives a crash." | Stress re-estimation into GFC 2008-09 and 2020-Q1; the desk *admits* it is level-conservative but persistence-wrong in a crash, and states the power limit (~0.5 expected 99% hits over the window). |
| "Your Sharpe is a point estimate — how many arms did you try?" | Deflated Sharpe over the 3 arms: p = 0.17 / 0.34, so after the multiple-trials correction the edge is real but **not** proven; CIs reported on every headline metric. |

What I would change next: a square-root cost model (current is flat 10 bps), a longer pre-GFC window for the 2008 exhibit, and an unscented-Kalman volatility recursion so the overlay re-prices risk daily instead of monthly.

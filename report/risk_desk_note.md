# Risk Desk Note — SPY GARCH VaR/ES & Overlay
*One-pager. 2015-01 → 2026-10, 2,969 daily obs. Stress windows: GFC 2008-09 (cache) + 2020-Q1.
Every number is reproducible via `python run_desk.py` (offline, cache-first).*

## 1. GARCH-t is the honest volatility model
GARCH(1,1) with a Student-t conditional, fit to SPY: persistence α+β = **0.994**
(near-integrated, the expected equity signature) and AIC 7334 vs 7540 for Gaussian
GARCH. The t's lower AIC is the clearest heavy-tail signal: a normal GARCH
systematically mis-prices the tails, so any 99% VaR/ES it emits is too tight.

## 2. 99% 1-day VaR and Expected Shortfall are defensible
On the last 552 obs both the historical and GARCH-t VaR logged **7 of 552 hits =
1.27%** against the 1% nominal. Kupiec POF z = +0.63, p = 0.53 — coverage is not
rejected, over- or under-conservative. ES sits at a mean **3.87%** (the t mean of
the tail) with model-implied breach probability β = 0.0034. The joint VaR/ES
likelihood-ratio test (**Méndez–Gaona**) gives χ² = 1.94, df = 2, p = 0.38: neither
the level nor the intermediate loss band is miscalibrated. ES is the "Basel"
tail metric VaR can't answer — sub-additivity at the 99% level — and it backtests
clean.

## 3. The desk admits its limits in a stress regime (the elite move)
Re-fit the *global* GARCH-t on the calm pre-period, iterate the one-step-ahead
VaR/ES forecast **into** each named window (no look-ahead — a test pins the
forecast invariant to future data), and report where the model concedes:

| Window | n | persistence global → re-fit | model ES (mean) | realized mean tail loss | MG p |
|---|---|---|---|---|---|
| GFC 2008-09 | 44 | 1.0000 → 1.0000 | 13.9% | 8.2% | 0.51 |
| 2020-Q1 | 54 | 0.9876 → 0.9995 | 14.2% | 3.4% | 0.57 |

Read: (i) the calm fit is already at the persistence boundary, so a crash has
**no persistence to raise** — the 2020 "regime" shows up only as 0.988→1.000.
(ii) GARCH vol **over-shoots** the shock: the ES level stays ~14% for weeks after
the worst day while realized tail loss is 3–8%. The desk is *level-conservative
but persistence-wrong* in a crash — exactly where the 1990s VaR answer fails,
and why the fix is ES + an explicit stress window, not the recursion. (iii) Power:
a 99% POF over 44–54 days expects ~0.5 hits, so a non-rejection **is not
validation** — read the hit rate against the ES level, not the p-value.

## 4. Overlay: report the surface, not the point
Single-asset 15% vol-target vs a rolling SPY+TLT inverse-vol **risk-parity**
basket (monthly, 10 bps), at three target vols:

| Arm | Ann ret | Ann vol | Sharpe | Max DD | Turnover/day |
|---|---|---|---|---|---|
| 15% vol-target (SPY/T-bill) | 10.1% | 14.8% | 0.68 | 28.3% | — |
| RP @ 10% (SPY/TLT) | 6.1% | 11.2% | 0.54 | 28.8% | 0.45% |
| RP @ 15% (SPY/TLT) | 7.8% | 15.8% | 0.50 | 40.4% | 0.55% |
| RP @ 20% (SPY/TLT) | 8.2% | 18.3% | 0.45 | 48.4% | 0.52% |

The surface is monotone: higher target vol → lower Sharpe, higher DD. On
2015–2026 the duration leg *costs* Sharpe (TLT drags in a long equity bull); the
case for risk parity is diversification **during** high-vol equity stress, not
across the whole panel. Both legs are levered (mean gross ~1.6× at the 15% target
— disclosed), so the basket is a levered risk-parity, not a long-only one.

## 5. Distributions, not point estimates
Block-bootstrap 95% CI on the annualized Sharpe + **deflated Sharpe** (corrected
for the 3 arms tried on this panel):
- **15% vol-target:** Sharpe 0.72, CI [0.20, 1.37] excludes 0, DSR p = 0.17 (not sig.)
- **RP @ 15%:** Sharpe 0.56, CI [-0.02, 1.14] spans 0, DSR p = 0.34 (not sig.)

The elite read: the raw Sharpe looks respectable, but after the multiple-trials
correction **neither arm clears 5%** — the edge is real, not proven. That is the
honest line, and the one a hostile reviewer wants to see.

## What we would ask the client
1. Is the objective a "don't lose 20%" (drawdown cap) or "maximize Sharpe"
   (risk-adjusted)? These goals point at opposite arms above.
2. Can you hold a 100%-equity peak and a ~1.6× levered risk-parity basket
   through a 2020-03 style drawdown?
3. Is the 10 bps cost realistic at your venue, and does the levered basket clear
   your collateral/margin limits?

## CV bullet
> GARCH/EGARCH risk desk with 99% VaR **and Expected Shortfall** (Méndez–Gaona
> tested), 2008/2020 stress re-estimation that names where the model breaks, a
> rolling SPY/TLT risk-parity overlay, and bootstrap / deflated-Sharpe CIs on all
> headline metrics. 35 offline tests, one-command reproducible.

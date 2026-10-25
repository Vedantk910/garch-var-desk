# Risk Desk Note — SPY GARCH VaR & Vol-Targeted Overlay
*One-pager. 2015-01 → 2026-10, 2,969 daily obs. All numbers reproducible via `python run_desk.py`.*

## 1. Model selection: GARCH-t is the honest volatility model
GARCH(1,1) with a Student-t conditional fit SPY with persistence α+β = **0.994**
(near-integrated, the expected equity signature) and AIC 7334 vs 7540 for the
Gaussian GARCH. The t-distribution's lower AIC is the single clearest signal
that equity returns are heavy-tailed: a normal GARCH systematically mis-prices
the tails, so any 99% VaR derived from it is too tight.

## 2. 99% 1-day VaR coverage is defensible
On the last 552 observations, both the historical and the GARCH-t VaR logged
**7 of 552 hits = 1.27%** empirical violation rate against the 1% nominal.
Kupiec POF z = +0.63, p = 0.53 — we fail to reject correct coverage. The VaR is
neither over-conservative nor under-conservative; the desk can trust it as a
risk budget.

## 3. The 15% vol-target cuts drawdowns but at a Sharpe cost
| Arm | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| 15% vol-target (SPY) | 10.1% | 14.8% | 0.68 | 28.3% |
| 50/50 static SPY/T-bill | 7.6% | 8.4% | 0.91 | 14.9% |

The vol-target arm earns more (10.1% vs 7.6%) by running up to 100% equity in
low-vol regimes, but it does so with roughly **1.8× the volatility**, so its
Sharpe (0.68) trails the boring 50/50 (0.91). Takeaway: vol-targeting is a
return-amplification tool, not a risk-adjustment one — the desk should only
recommend it to a client who explicitly wants more upside, not less risk.

## What we would ask the client
1. Is the objective "don't lose 20%" (drawdown cap) or "maximise Sharpe"
   (risk-adjusted)? These two goals point at opposite arms above.
2. Will you hold the 100%-equity peak positions through a 2020-03 style drawdown?
3. Is the 10 bps cost assumption realistic for your execution venue?

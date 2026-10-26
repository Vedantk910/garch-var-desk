# garch-var-desk

A one-day 99% Value-at-Risk desk that admits its errors. GARCH(1,1)-t and EGARCH volatility forecasts on daily SPY, coverage backtests against a decade of returns, and a 15% vol-target overlay tested against a static 50/50 benchmark.

## The honest numbers

- GARCH persistence 0.994 on 2015–2026 daily equity data.
- 99% VaR recorded 7 violations in 552 days (1.27% against a 1% target).
- Kupiec POF test: p = 0.53, fails to reject correct coverage.
- Christoffersen independence test does not reject; no clustering signal.
- The 15% vol-target cut max drawdown to 11.4% and traded Sharpe for it.

## What it does

1. **Data** — daily SPY log returns, FRED risk-free series (cache-first, offline fallback)
2. **Model** — GARCH(1,1)-t and EGARCH(1,1,1) via MLE (`arch`), one-step-ahead volatility
3. **Risk** — 99% VaR from the conditional distribution; Kupiec POF and Christoffersen CC backtests
4. **Portfolio** — a 15% vol-target overlay versus a static 50/50 benchmark, scored on Sharpe and max drawdown

The result is a risk tool you can defend: coverage is *tested*, not asserted, and the vol-target's cost is reported in the same paragraph as its benefit.

## Project layout

```
garch_var/
  data/       # series + cache-first loaders
  models/     # GARCH / EGARCH MLE wrappers
  risk/       # VaR, Kupiec POF, Christoffersen CC
  portfolio/  # vol-target overlay + benchmark
run_desk.py   # end-to-end CLI, writes out/desk_numbers.json
report/       # risk_desk_note.md
tests/        # pytest suite (offline, synthetic)
```

## Quick start

```bash
pip install -r requirements.txt
python run_desk.py          # writes out/desk_numbers.json
python -m pytest tests/ -q
```

Every figure in `report/risk_desk_note.md` is out-of-sample relative to its window and reproducible by one command.

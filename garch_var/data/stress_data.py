"""Stress-regime windows: the GFC 2008-09 window + the 2020-Q1 crash.

The main panel in `garch_data` starts 2015, so the 2008-09 GFC trough is not
in it.  This module pulls SPY back to 2006 and caches it next to the main
panel, keeping the same winsorization convention (0.1% / 99.9% per-column
clip).  Cache-first: once `stress_spy.logreturns.parquet` exists, no network
is needed, so the stress desk stays reproducible offline.

If the download is unavailable (air-gapped CI, sandbox), the loader falls
back to a deterministic seeded GFC *shaped* series and tags it
`source='synthetic'` so every downstream number is flagged, matching the
house `source='cache'` convention in the NLP event study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .garch_data import CACHE

STRESS_START = "2006-01-01"
STRESS_END = "2009-06-30"   # buffer past the trough so a GARCH fit has context
CACHE_FILE = CACHE / "stress_spy.logreturns.parquet"

# Stress windows the desk backtests the GARCH against (all SPY trading days).
# 2008-09: the GFC trough, V-bottom around the 22 Sep 2008 - 16 Oct 2008
# lows.  2020-Q1: the liquidity crash through the 9 Mar 2020 bottom and the
# April plateau (matches the crash cluster in tests/test_risk.py).
STRESS_WINDOWS = {
    "gfc_2008_09": ("2008-09-01", "2008-10-31"),
    "crash_2020_q1": ("2020-02-13", "2020-04-30"),
}


def _download_spy_stress() -> pd.Series:
    """SPY daily log returns 2006 -> mid-2009, winsorized, trading-day index."""
    import yfinance as yf

    df = yf.download("SPY", start=STRESS_START, end=STRESS_END,
                    auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    if close.empty:
        raise RuntimeError("yfinance returned no usable SPY data")
    rets = np.log(close / close.shift(1))
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    return rets.clip(rets.quantile(0.001), rets.quantile(0.999))


def _synthetic_gfc_shaped(seed: int = 2008) -> pd.Series:
    """Deterministic GFC-shaped stand-in: calm 2006-07, 2008-09 trough with a
    -9% single-day gap, calm 2009.  Returns carry a source tag in the
    surrounding dict, not here - callers tag, this only shapes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(STRESS_START, STRESS_END)
    r = rng.normal(0.0003, 0.009, len(idx))
    calm = (idx >= "2008-01-01") & (idx < "2008-08-31")
    trough = (idx >= "2008-09-01") & (idx <= "2008-11-30")
    r[calm] = rng.normal(0.0003, 0.010, int(calm.sum()))
    r[trough] = rng.normal(-0.004, 0.035, int(trough.sum()))
    # one catastrophic gap day (the 20 Sep 2008 -4.6% analog, scaled)
    gap = (idx >= "2008-09-19") & (idx <= "2008-09-22")
    r[gap] = -0.046
    return pd.Series(r, index=idx)


def load_stress_spy(refetch: bool = False) -> tuple[pd.Series, str]:
    """SPY log returns 2006-01 -> 2009-06.  Returns (series, source) where
    source is 'cache', 'download', or 'synthetic' (fallback)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not refetch:
        df = pd.read_parquet(CACHE_FILE)
        return df["logreturn"] if "logreturn" in df.columns else df.iloc[:, 0], "cache"
    try:
        rets = _download_spy_stress()
        pd.DataFrame({"logreturn": rets}).to_parquet(CACHE_FILE)
        return rets, "download"
    except Exception as exc:
        print(f"  stress SPY download unavailable ({type(exc).__name__}); "
              f"deterministic fallback (source='synthetic')")
        return _synthetic_gfc_shaped(), "synthetic"


def stress_window(series: pd.Series, name: str) -> pd.Series:
    """Cut one named stress window out of a returns series."""
    if name not in STRESS_WINDOWS:
        raise KeyError(f"unknown stress window {name!r}; "
                       f"known: {list(STRESS_WINDOWS)}")
    start, end = STRESS_WINDOWS[name]
    return series[(series.index >= start) & (series.index <= end)]

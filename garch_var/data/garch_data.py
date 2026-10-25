"""Daily equity returns + risk-free proxy, cached to parquet.

Deterministic: once data/cache is populated, no network is needed.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
CACHE = HERE / "data" / "cache"

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "CAT", "XLE", "TLT"]
START = "2015-01-01"


def _download_daily() -> pd.DataFrame:
    """Per-ticker yfinance downloads (matches the P1 loader pattern).

    The multi-ticker yf.download API has changed column layout between
    releases, so fetching one ticker at a time is the robust path.
    """
    import yfinance as yf

    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].dropna()
            if not close.empty:
                data[t] = close
        except Exception:
            print(f"  yfinance failed for {t}; skipped")
    close = pd.DataFrame(data)
    # Per-ticker trading calendars differ (holidays, IPO date); align everything
    # to the SPY calendar so all assets share one time axis.
    master = close["SPY"].index if "SPY" in close.columns else close.index
    close = close.reindex(master)
    close = close.ffill()
    close = close.dropna()  # only the leading warm-up rows are NaN
    if close.empty:
        raise RuntimeError("yfinance returned no usable data for any ticker")
    return close


def load_returns(refetch: bool = False) -> pd.DataFrame:
    """Daily log returns, winsorized at the 0.1% / 99.9% quantiles."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "returns.parquet"
    if cache_file.exists() and not refetch:
        rets = pd.read_parquet(cache_file)
    else:
        close = _download_daily()
        rets = np.log(close / close.shift(1))
        rets = rets.replace([np.inf, -np.inf], np.nan)
        # winsorize each column at its own 0.1% / 99.9% quantiles; the
        # per-column lambda avoids the pandas clip() Series-alignment pitfall
        rets = rets.apply(lambda col: col.clip(col.quantile(0.001),
                                                col.quantile(0.999)))
        rets = rets.dropna()
        rets.to_parquet(cache_file)
    return rets


def load_rf_daily() -> pd.Series:
    """Daily risk-free proxy: FEDFUNDS / 252. Falls back to a fixed 3%."""
    import urllib.request

    idx = load_returns().index
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
        raw = pd.read_csv(io.StringIO(text))
        raw.columns = [c.strip().lower() for c in raw.columns]
        monthly = pd.to_numeric(raw[raw.columns[1]], errors="coerce")
        monthly.index = pd.to_datetime(raw[raw.columns[0]], errors="coerce")
        monthly = monthly.dropna().sort_index()
        # FEDFUNDS is a monthly ANNUAL rate: /100 to decimal, /252 to daily
        daily = (monthly / 100.0 / 252.0).reindex(idx, method="ffill")
        return daily.rename("FEDFUNDS_daily_rf")
    except Exception:
        return pd.Series(0.03 / 252.0, index=idx, name="fallback_rf")

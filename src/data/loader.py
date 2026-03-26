"""
Data loader: fetches and caches all price/yield data needed for the carry strategy.

Data sources (both FREE, no API key required)
--------------------------------------------
1. **yfinance**  — continuous front-month futures for daily RETURNS
                   (CL=F, GC=F, GLD, etc.)
2. **FRED**      — spot prices and yield curve data for CARRY SIGNALS
                   https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>

Carry computation per market type
----------------------------------
* basis      : carry = log(futures_price / spot_price) × 12   [annualised]
* yield_spread: carry = yield_long − yield_short               [% per year]

Caching
-------
All downloads are saved to Parquet files under data/raw/ so subsequent runs
are instant.  Delete data/raw/ to force a fresh download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ─── Config ──────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/futures_universe.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Low-level fetchers ───────────────────────────────────────────────────────

def _fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    """
    Download a FRED time series to a pandas Series (no API key needed).

    Handles both daily and monthly series.  Monthly series are forward-filled
    to produce a daily index when used as spot price proxies.
    """
    cache = RAW_DIR / f"fred_{series_id}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s[(s.index >= start) & (s.index <= end)].dropna()

    url = f"{FRED_BASE_URL}?id={series_id}"
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values=["."])
        df.columns = [series_id]
        df.to_parquet(cache)
        s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s[(s.index >= start) & (s.index <= end)].dropna()
    except Exception as e:
        logger.error(f"FRED fetch failed for {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)


def _fetch_yfinance_single(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download closing prices for ONE yfinance ticker.

    Returns a Series with date index.  Returns empty Series on failure.
    """
    cache = RAW_DIR / f"yf_{ticker.replace('=', '_').replace('^', '_')}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s[(s.index >= start) & (s.index <= end)].dropna()

    try:
        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning(f"yfinance: no data for {ticker}")
            return pd.Series(dtype=float)

        # Flatten MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw[("Close", ticker)]
        else:
            close = raw["Close"]

        close = close.dropna()
        close.index = pd.to_datetime(close.index)
        close.to_frame("Close").to_parquet(cache)
        return close[(close.index >= start) & (close.index <= end)]
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float)


def _align_daily(spot: pd.Series, rebal_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Reindex spot series to rebalance dates using forward-fill + backward-fill
    (handles weekends, holidays, and monthly data gaps).
    """
    all_dates = pd.date_range(
        start=rebal_dates.min() - pd.Timedelta(days=10),
        end=rebal_dates.max() + pd.Timedelta(days=5),
        freq="B",
    )
    aligned = spot.reindex(all_dates).ffill().bfill()
    return aligned.reindex(rebal_dates, method="ffill")


# ─── Carry table builder ─────────────────────────────────────────────────────

def build_carry_table(
    config: dict,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build a monthly carry signal table.

    For each rebalance date (first business day of month) and each market,
    computes the annualised carry signal using the method specified in config.

    Returns
    -------
    DataFrame (rebalance_dates × markets) of annualised carry (float).
    NaN where data is unavailable.
    """
    cache = RAW_DIR / "carry_table.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Carry table loaded from cache {df.shape}")
        return df

    fut_cfg = config["futures"]
    rebal_dates = _get_rebalance_dates(start, end)

    # ── Pre-fetch all needed series ──────────────────────────────────────────
    futures_prices: Dict[str, pd.Series] = {}     # market → daily closing price
    spot_series: Dict[str, pd.Series] = {}        # market → spot price
    fred_series: Dict[str, pd.Series] = {}        # FRED series cache

    for mkt, spec in fut_cfg.items():
        # Continuous futures (for carry ratio)
        logger.info(f"Fetching continuous {mkt} ({spec['continuous_ticker']}) ...")
        futures_prices[mkt] = _fetch_yfinance_single(
            spec["continuous_ticker"], start, end
        )

        method = spec.get("carry_method", "basis")

        if method == "basis":
            src = spec.get("spot_source", "fred")
            if src == "fred":
                sid = spec["spot_series"]
                if sid not in fred_series:
                    logger.info(f"  Fetching FRED {sid} for {mkt} ...")
                    fred_series[sid] = _fetch_fred(sid, start, end)
                spot_series[mkt] = fred_series[sid]
            elif src == "yfinance":
                ytk = spec["spot_series"]
                logger.info(f"  Fetching yfinance {ytk} as spot for {mkt} ...")
                spot_series[mkt] = _fetch_yfinance_single(ytk, start, end)

        elif method == "yield_spread":
            for key in ("yield_long_series", "yield_short_series"):
                sid = spec[key]
                if sid not in fred_series:
                    logger.info(f"  Fetching FRED {sid} ...")
                    fred_series[sid] = _fetch_fred(sid, start, end)

        elif method == "cost_of_carry":
            # Gold model: futures ≈ spot × e^(r+storage). carry ≈ −(r + 0.15%)
            # We use −DTB3 as a proxy: gold is always in slight contango.
            sid = spec.get("yield_short_series", "DTB3")
            if sid not in fred_series:
                logger.info(f"  Fetching FRED {sid} for {mkt} (cost-of-carry) ...")
                fred_series[sid] = _fetch_fred(sid, start, end)

    # Ensure DTB3 is always loaded (used by cost_of_carry and yield_spread)
    if "DTB3" not in fred_series:
        fred_series["DTB3"] = _fetch_fred("DTB3", start, end)

    # ── Compute carry per rebalance date ─────────────────────────────────────
    records = []
    for date in rebal_dates:
        row: dict = {"date": date}

        for mkt, spec in fut_cfg.items():
            method = spec.get("carry_method", "basis")

            try:
                if method == "basis":
                    f_price = _asof(futures_prices.get(mkt), date)
                    s_raw = _asof(spot_series.get(mkt), date)

                    multiplier = spec.get("spot_multiplier", 1.0)
                    s_price = s_raw * multiplier if not np.isnan(s_raw) else np.nan

                    if np.isnan(f_price) or np.isnan(s_price) or s_price <= 0 or f_price <= 0:
                        row[mkt] = np.nan
                    else:
                        # annualised log-return carry
                        row[mkt] = np.log(f_price / s_price) * 12.0

                elif method == "yield_spread":
                    y_long = _asof(fred_series.get(spec["yield_long_series"]), date)
                    y_short = _asof(fred_series.get(spec["yield_short_series"]), date)

                    if np.isnan(y_long) or np.isnan(y_short):
                        row[mkt] = np.nan
                    else:
                        # carry in % per year; convert to decimal for comparability
                        row[mkt] = (y_long - y_short) / 100.0

                elif method == "cost_of_carry":
                    # Gold: carry ≈ −(short_rate + storage_cost)
                    # Futures are always slightly above spot (contango).
                    # carry = −DTB3/100 − 0.002  (storage ~0.2%/yr for gold vaults)
                    short_rate = _asof(
                        fred_series.get(spec.get("yield_short_series", "DTB3")), date
                    )
                    if np.isnan(short_rate):
                        row[mkt] = np.nan
                    else:
                        row[mkt] = -(short_rate / 100.0 + 0.002)

                else:
                    row[mkt] = np.nan

            except Exception as e:
                logger.debug(f"Carry compute error {mkt} {date.date()}: {e}")
                row[mkt] = np.nan

        records.append(row)

    df = pd.DataFrame(records).set_index("date")
    df.to_parquet(cache)
    logger.info(f"Carry table built: {df.shape}, "
                f"NaN rate: {df.isna().mean().mean():.1%}")
    return df


# ─── Continuous front-month returns ──────────────────────────────────────────

def build_return_series(
    config: dict,
    start: str,
    end: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build a daily return matrix for all markets.

    Uses continuous front-month contracts (e.g. CL=F) from yfinance.
    These are downloaded one at a time to avoid yfinance batch-download
    MultiIndex parsing issues.

    Returns
    -------
    DataFrame (trading_days × markets) of daily % returns.
    """
    cache = RAW_DIR / "continuous_returns.parquet"
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        df.index = pd.to_datetime(df.index)
        logger.info(f"Continuous returns loaded from cache {df.shape}")
        return df

    fut_cfg = config["futures"]
    prices: Dict[str, pd.Series] = {}

    for mkt, spec in fut_cfg.items():
        ticker = spec["continuous_ticker"]
        logger.info(f"Fetching continuous {mkt} returns ({ticker}) ...")
        series = _fetch_yfinance_single(ticker, start, end)
        if not series.empty:
            prices[mkt] = series

    price_df = pd.DataFrame(prices)
    price_df.index = pd.to_datetime(price_df.index)
    returns_df = price_df.pct_change(fill_method=None).fillna(0.0)

    returns_df.to_parquet(cache)
    logger.info(f"Continuous returns built: {returns_df.shape}")
    return returns_df


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_rebalance_dates(start: str, end: str) -> pd.DatetimeIndex:
    """First business day of each month in [start, end]."""
    month_starts = pd.date_range(start=start, end=end, freq="MS")
    biz_dates = []
    for d in month_starts:
        bd = d
        while bd.weekday() >= 5:
            bd += pd.Timedelta(days=1)
        biz_dates.append(bd)
    return pd.DatetimeIndex(biz_dates)


def _asof(series: Optional[pd.Series], date: pd.Timestamp) -> float:
    """Return the last observation on or before *date*, or NaN."""
    if series is None or series.empty:
        return np.nan
    val = series.asof(date)
    return float(val) if not pd.isna(val) else np.nan

"""
Contract calendar utilities for CME futures.

Provides functions to:
  - Construct Yahoo Finance ticker strings for specific contract months
  - Determine front/second nearby contracts for any date
  - Generate monthly rebalance dates
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict

# CME standard month codes
MONTH_CODES: List[str] = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]

# Bidirectional maps between month number (1-12) and CME code
CODE_TO_MONTH: Dict[str, int] = {c: i + 1 for i, c in enumerate(MONTH_CODES)}
MONTH_TO_CODE: Dict[int, str] = {i + 1: c for i, c in enumerate(MONTH_CODES)}


def build_ticker(root: str, month_code: str, year: int, exchange: str) -> str:
    """
    Build a Yahoo Finance futures ticker.

    Examples
    --------
    >>> build_ticker("CL", "K", 2025, "NYM")
    'CLK25.NYM'
    >>> build_ticker("GC", "Z", 2024, "CMX")
    'GCZ24.CMX'
    """
    return f"{root}{month_code}{str(year)[2:]}.{exchange}"


def get_nearby_contracts(
    root: str,
    traded_months: List[str],
    exchange: str,
    ref_date: pd.Timestamp,
    n: int = 3,
    offset_months: int = 1,
) -> List[Dict]:
    """
    Return the next *n* tradeable contract specs starting from *ref_date*.

    Parameters
    ----------
    root          : Futures root symbol, e.g. "CL"
    traded_months : List of CME month codes that trade for this product
    exchange      : Yahoo Finance exchange suffix, e.g. "NYM"
    ref_date      : Reference date (usually the rebalance date)
    n             : Number of contracts to return
    offset_months : Skip this many months before the first candidate.
                    offset_months=1 → start looking from next month (avoids
                    holding a contract in its expiry month)

    Returns
    -------
    List of dicts with keys: ticker, month_code, month, year, delivery_date
    """
    contracts: List[Dict] = []

    for i in range(offset_months, offset_months + 24):
        future = ref_date + pd.DateOffset(months=i)
        mc = MONTH_TO_CODE[future.month]

        if mc in traded_months:
            delivery = pd.Timestamp(future.year, future.month, 1)
            contracts.append(
                {
                    "ticker": build_ticker(root, mc, future.year, exchange),
                    "month_code": mc,
                    "month": future.month,
                    "year": future.year,
                    "delivery_date": delivery,
                }
            )

        if len(contracts) >= n:
            break

    return contracts


def get_all_tickers_for_market(
    root: str,
    traded_months: List[str],
    exchange: str,
    start: str,
    end: str,
) -> List[str]:
    """
    Collect every unique ticker that could be F1 or F2 across the full
    backtest window [start, end].  Used to plan bulk downloads.
    """
    rebal_dates = get_rebalance_dates(start, end)
    seen: set = set()
    tickers: List[str] = []

    for date in rebal_dates:
        for c in get_nearby_contracts(root, traded_months, exchange, date, n=3):
            if c["ticker"] not in seen:
                seen.add(c["ticker"])
                tickers.append(c["ticker"])

    return tickers


def get_rebalance_dates(start: str, end: str) -> pd.DatetimeIndex:
    """
    Monthly rebalance dates: first business day of each calendar month
    within [start, end].
    """
    month_starts = pd.date_range(start=start, end=end, freq="MS")
    biz_dates = []
    for d in month_starts:
        bd = d
        while bd.weekday() >= 5:  # Saturday = 5, Sunday = 6
            bd += pd.Timedelta(days=1)
        biz_dates.append(bd)
    return pd.DatetimeIndex(biz_dates)


def months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    """Approximate number of months between two timestamps."""
    return max((d2 - d1).days / 30.4375, 0.5)

"""
Event-driven backtest engine for the futures carry strategy.

The engine is intentionally simple: it takes a pre-computed weight matrix and
a return series, then simulates the daily P&L of holding those positions,
deducting transaction costs proportional to daily turnover.

Key assumptions
---------------
* Futures positions are fully-funded (cash collateral earns risk-free rate,
  which we ignore here for simplicity — focus is on carry alpha).
* Transaction cost = (round-trip bps) × |Δweight| per unit notional.
* No margin calls or portfolio-level stops.
* Returns are computed after costs.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


# ─── Core engine ─────────────────────────────────────────────────────────────

def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Daily one-way turnover: sum of absolute weight changes per day.

    Turnover on rebalance day reflects both closing old positions and
    opening new ones.  Non-rebalance days have near-zero turnover (only
    drift from price changes, which we approximate as zero here since
    futures weights are set and held until next rebalance).
    """
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def run_backtest(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_cost_bps: float = 1.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate the carry portfolio and return daily P&L series.

    Parameters
    ----------
    weights             : Daily weight matrix (trading_days × markets)
    returns             : Daily market returns (trading_days × markets)
    transaction_cost_bps: Round-trip transaction cost per unit notional turnover
                          (1 bps = 0.01%)

    Returns
    -------
    net_returns   : Daily net portfolio returns (after costs)
    gross_returns : Daily gross portfolio returns (before costs)
    turnover      : Daily one-way turnover
    """
    # Align on common dates and markets
    common_dates = weights.index.intersection(returns.index)
    common_mkts = [m for m in weights.columns if m in returns.columns]

    w = weights.loc[common_dates, common_mkts]
    r = returns.loc[common_dates, common_mkts]

    # Gross P&L: w_i × r_i summed across markets
    gross_returns = (w * r).sum(axis=1)

    # Transaction costs: cost_rate × turnover
    cost_rate = transaction_cost_bps / 10_000.0
    turnover = compute_turnover(w)
    tc = turnover * cost_rate

    net_returns = gross_returns - tc

    return net_returns, gross_returns, turnover


# ─── Sector attribution ───────────────────────────────────────────────────────

def sector_attribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: dict,
) -> pd.DataFrame:
    """
    Decompose net gross P&L by sector.

    Returns DataFrame (trading_days × sectors) with daily return contribution.
    """
    common_dates = weights.index.intersection(returns.index)
    common_mkts = [m for m in weights.columns if m in returns.columns]

    w = weights.loc[common_dates, common_mkts]
    r = returns.loc[common_dates, common_mkts]
    pnl_per_mkt = w * r

    sectors = sorted(set(sector_map.values()))
    sector_pnl = pd.DataFrame(index=common_dates, columns=sectors, dtype=float)

    for sector in sectors:
        mkts = [m for m, s in sector_map.items() if s == sector and m in pnl_per_mkt.columns]
        sector_pnl[sector] = pnl_per_mkt[mkts].sum(axis=1) if mkts else 0.0

    return sector_pnl


# ─── Rolling performance ──────────────────────────────────────────────────────

def rolling_sharpe(returns: pd.Series, window: int = 252, risk_free: float = 0.0) -> pd.Series:
    """
    Rolling annualised Sharpe ratio.

    Parameters
    ----------
    returns    : Daily net returns
    window     : Rolling window in trading days (default 252 = 1 year)
    risk_free  : Annualised risk-free rate (default 0 for simplicity)
    """
    excess = returns - risk_free / 252
    roll_mean = excess.rolling(window).mean() * 252
    roll_std = excess.rolling(window).std() * np.sqrt(252)
    return roll_mean / roll_std.replace(0, np.nan)


def underwater_equity(returns: pd.Series, starting_value: float = 1.0) -> pd.Series:
    """
    Return the drawdown series (0 at all-time high, negative in drawdown).
    """
    cum = starting_value * (1 + returns).cumprod()
    running_max = cum.expanding().max()
    return (cum - running_max) / running_max

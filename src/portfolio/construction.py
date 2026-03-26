"""
Portfolio construction for the cross-sectional carry strategy.

Approach
--------
1. Rank markets by carry signal each month.
2. Long the top N (highest backwardation) and short the bottom N (deepest contango).
3. Within each leg, weight inversely proportional to 60-day realised volatility
   so each market contributes equally to risk (vol parity).
4. Scale the gross book to target a given annual vol (default 15%).
5. Enforce dollar-neutrality: long book notional = short book notional.

The resulting weights represent fractional portfolio allocation.  A weight of
+0.20 means 20% of capital is long that market.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Realised volatility ─────────────────────────────────────────────────────

def realised_vol(
    returns: pd.DataFrame,
    lookback: int = 60,
    min_periods: int = 20,
    annualise: bool = True,
) -> pd.DataFrame:
    """
    Rolling realised volatility (annualised by default).

    Parameters
    ----------
    returns   : Daily returns (trading days × markets)
    lookback  : Rolling window in trading days
    min_periods: Minimum observations before computing vol
    annualise : Multiply by sqrt(252) if True

    Returns
    -------
    DataFrame of same shape as *returns*, NaN before min_periods.
    """
    vol = returns.rolling(lookback, min_periods=min_periods).std()
    if annualise:
        vol *= np.sqrt(252)
    return vol


# ─── Core portfolio builder ───────────────────────────────────────────────────

def build_weights(
    carry_signals: pd.DataFrame,
    returns: pd.DataFrame,
    n_long: int = 4,
    n_short: int = 4,
    target_vol: float = 0.15,
    vol_lookback: int = 60,
    max_position: float = 0.30,
) -> pd.DataFrame:
    """
    Build a daily weight matrix from monthly carry signals.

    Parameters
    ----------
    carry_signals : Monthly carry table (rebalance_dates × markets)
    returns       : Daily returns (trading_days × markets)
    n_long        : Number of long positions per rebalance
    n_short       : Number of short positions per rebalance
    target_vol    : Target annual portfolio volatility (gross)
    vol_lookback  : Days for realised vol estimate
    max_position  : Maximum single-market weight (each side)

    Returns
    -------
    weights : DataFrame (trading_days × markets) with daily positions.
    """
    # Compute rolling vol on full returns history
    vol_df = realised_vol(returns, lookback=vol_lookback)

    # Align markets: only trade markets present in BOTH carry and returns
    common_mkts = [m for m in carry_signals.columns if m in returns.columns]
    carry = carry_signals[common_mkts]
    vol_df = vol_df[common_mkts]
    returns_sub = returns[common_mkts]

    # Initialise weights to zero
    weights = pd.DataFrame(0.0, index=returns_sub.index, columns=common_mkts)

    rebal_dates = carry.index.tolist()

    for i, rebal_date in enumerate(rebal_dates):
        # Period covered by these weights: [rebal_date, next_rebal_date)
        if i + 1 < len(rebal_dates):
            period_end = rebal_dates[i + 1]
        else:
            period_end = returns_sub.index[-1] + pd.Timedelta(days=1)

        period_mask = (returns_sub.index >= rebal_date) & (returns_sub.index < period_end)

        carry_row = carry.loc[rebal_date].dropna()
        if len(carry_row) < n_long + n_short:
            logger.debug(f"{rebal_date.date()}: only {len(carry_row)} markets — skipping")
            continue

        # Vol estimate at rebalance date (or most recent available)
        vol_at_date = vol_df.loc[:rebal_date].iloc[-1] if rebal_date in vol_df.index else vol_df.loc[:rebal_date].iloc[-1]
        vol_at_date = vol_at_date[carry_row.index].fillna(0.20)  # default 20% vol
        vol_at_date = vol_at_date.clip(lower=0.02)  # floor at 2% to prevent huge weights

        # Rank: 1 = lowest carry, N = highest carry
        ranked = carry_row.rank(ascending=True)
        n = len(ranked)

        long_mkts = ranked[ranked >= (n - n_long + 1)].index
        short_mkts = ranked[ranked <= n_short].index

        # Inverse-vol weights within each leg
        long_inv_vol = (1.0 / vol_at_date[long_mkts])
        short_inv_vol = (1.0 / vol_at_date[short_mkts])

        long_w = long_inv_vol / long_inv_vol.sum()   # sum to 1 within long leg
        short_w = short_inv_vol / short_inv_vol.sum() # sum to 1 within short leg

        # Enforce max single-position cap
        long_w = long_w.clip(upper=max_position)
        long_w /= long_w.sum()
        short_w = short_w.clip(upper=max_position)
        short_w /= short_w.sum()

        # Vol-scale the gross book to hit target_vol
        # Approximate: portfolio vol ≈ avg_vol_of_positions × gross_leverage
        avg_vol = vol_at_date[list(long_mkts) + list(short_mkts)].mean()
        gross_scale = target_vol / avg_vol if avg_vol > 0 else 1.0
        gross_scale = np.clip(gross_scale, 0.5, 3.0)  # allow 0.5x–3x leverage

        # Apply: long leg = +0.5 × gross_scale, short leg = −0.5 × gross_scale
        # (dollar-neutral: long notional = short notional = 0.5 of portfolio)
        final_long = long_w * 0.5 * gross_scale
        final_short = short_w * 0.5 * gross_scale

        period_dates = returns_sub.index[period_mask]
        for mkt in final_long.index:
            weights.loc[period_dates, mkt] = final_long[mkt]
        for mkt in final_short.index:
            weights.loc[period_dates, mkt] -= final_short[mkt]

    return weights


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def portfolio_summary(weights: pd.DataFrame) -> pd.Series:
    """
    Quick diagnostics on the weight matrix.
    """
    nonzero = (weights != 0).any(axis=1)
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    long_notl = weights.clip(lower=0).sum(axis=1)
    short_notl = weights.clip(upper=0).abs().sum(axis=1)

    return pd.Series(
        {
            "days_invested_%": nonzero.mean() * 100,
            "avg_gross_leverage": gross[nonzero].mean(),
            "avg_net_exposure": net[nonzero].mean(),
            "avg_long_notional": long_notl[nonzero].mean(),
            "avg_short_notional": short_notl[nonzero].mean(),
            "avg_n_positions": (weights != 0).sum(axis=1)[nonzero].mean(),
        }
    )

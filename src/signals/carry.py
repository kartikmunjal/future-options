"""
Carry signal computation for cross-sectional futures carry strategy.

Theory
------
In futures markets, the *roll yield* (carry) is the return earned by
mechanically rolling a long position from the expiring front contract to
the next nearby.  For commodity markets:

    carry ≈ convenience_yield − storage_cost   (backwardation → +carry)

For financial futures (rates, equity):

    carry ≈ coupon / dividend − financing_cost

We proxy carry using the observable term-structure spread:

    carry_i(t) = log(F1_i / F2_i) × (12 / Δmonths)   [annualised]

A long/short portfolio that buys high-carry (backwardated) and sells
low-carry (contangoed) markets captures roll-yield alpha with near-zero
beta to any single commodity index.

References
----------
Erb & Harvey (2006). "The Strategic and Tactical Value of Commodity Futures."
Koijen, Moskowitz, Pedersen & Vrugt (2018). "Carry." Journal of Financial Economics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


# ─── Raw carry computation ────────────────────────────────────────────────────

def annualised_carry(
    f1: float | pd.Series,
    f2: float | pd.Series,
    months_between: float = 1.0,
) -> float | pd.Series:
    """
    Annualised log-return carry between front (F1) and second (F2) contracts.

    Parameters
    ----------
    f1, f2        : Prices of front and second nearby contracts
    months_between: Calendar months separating the two delivery dates

    Returns
    -------
    carry (float or Series) in annualised units.
    Positive → backwardation (F1 > F2) → roll long earns yield.
    Negative → contango (F1 < F2) → roll long bleeds yield.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        carry = np.log(f1 / f2) * (12.0 / max(months_between, 0.5))
    return carry


# ─── Signal normalisation ─────────────────────────────────────────────────────

def cross_sectional_zscore(carry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score carry signals cross-sectionally (across markets, per date).

    Missing values are excluded from the mean/std calculation.
    """
    return carry_df.apply(
        lambda row: (row - row.mean()) / (row.std() if row.std() > 0 else 1.0),
        axis=1,
    )


def sector_neutral_zscore(
    carry_df: pd.DataFrame, sector_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Demean carry within each sector before cross-sectional z-scoring.

    This removes the systematic sector-level carry bias (e.g. grains are
    typically in stronger backwardation than rates).

    Parameters
    ----------
    carry_df   : Monthly carry table (dates × markets)
    sector_map : {market_symbol: sector_name}

    Returns
    -------
    Z-scored carry DataFrame, same shape as input.
    """
    demeaned = carry_df.copy()

    sectors = set(sector_map.values())
    for sector in sectors:
        mkts = [m for m, s in sector_map.items() if s == sector and m in carry_df.columns]
        if len(mkts) < 2:
            continue
        sector_data = carry_df[mkts]
        sector_mean = sector_data.mean(axis=1)
        demeaned[mkts] = sector_data.sub(sector_mean, axis=0)

    return cross_sectional_zscore(demeaned)


# ─── Signal diagnostics ───────────────────────────────────────────────────────

def carry_autocorrelation(carry_df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """
    Compute lag-k autocorrelation of carry signals for each market.

    High autocorrelation → carry is persistent → signal is worth holding
    for multiple months.
    """
    records = {}
    for mkt in carry_df.columns:
        s = carry_df[mkt].dropna()
        records[mkt] = {f"ac_{lag}": s.autocorr(lag=lag) for lag in range(1, lags + 1)}
    return pd.DataFrame(records).T


def carry_summary(carry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for carry signals across the full sample.

    Returns a DataFrame (markets × stats) with mean, std, min, max,
    % in backwardation, and 1-month autocorrelation.
    """
    stats: Dict[str, dict] = {}
    for mkt in carry_df.columns:
        s = carry_df[mkt].dropna()
        if s.empty:
            continue
        stats[mkt] = {
            "mean_carry_%": s.mean() * 100,
            "std_carry_%": s.std() * 100,
            "min_carry_%": s.min() * 100,
            "max_carry_%": s.max() * 100,
            "pct_backwardation": (s > 0).mean() * 100,
            "autocorr_1m": s.autocorr(lag=1),
            "n_obs": len(s),
        }
    return pd.DataFrame(stats).T.sort_values("mean_carry_%", ascending=False)


# ─── Carry-return predictability ─────────────────────────────────────────────

def carry_return_correlation(
    carry_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    forward_months: int = 1,
) -> pd.DataFrame:
    """
    Compute the correlation between carry signal and forward returns for
    each market.  Tests whether carry predicts returns (the core hypothesis).

    Parameters
    ----------
    carry_df      : Monthly carry signals (dates × markets)
    returns_df    : Daily returns (dates × markets)
    forward_months: Months ahead over which to compute forward return

    Returns
    -------
    DataFrame (markets × stats): correlation, t-stat, hit_rate
    """
    # Monthly returns from daily
    monthly_ret = (1 + returns_df).resample("ME").prod() - 1

    results: Dict[str, dict] = {}
    for mkt in carry_df.columns:
        if mkt not in monthly_ret.columns:
            continue
        carry = carry_df[mkt].dropna()
        fwd_ret = monthly_ret[mkt].shift(-forward_months)

        common = carry.index.intersection(fwd_ret.dropna().index)
        if len(common) < 12:
            continue

        c = carry.loc[common]
        r = fwd_ret.loc[common]
        corr = c.corr(r)
        n = len(common)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(max(1 - corr**2, 1e-9))
        hit_rate = (np.sign(c) == np.sign(r)).mean()

        results[mkt] = {
            "carry_ret_corr": corr,
            "t_stat": t_stat,
            "hit_rate_%": hit_rate * 100,
            "n_obs": n,
        }

    if not results:
        return pd.DataFrame(columns=["carry_ret_corr", "t_stat", "hit_rate_%", "n_obs"])
    df = pd.DataFrame(results).T
    if "carry_ret_corr" in df.columns:
        df = df.sort_values("carry_ret_corr", ascending=False)
    return df

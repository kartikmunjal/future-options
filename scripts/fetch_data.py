#!/usr/bin/env python
"""
fetch_data.py — Download and cache all CME futures data for the carry strategy.

Usage
-----
    python scripts/fetch_data.py [--force] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

Options
-------
--force   : Re-download even if cache files exist
--start   : Start date (default from config)
--end     : End date   (default from config)

What it does
------------
1. Reads the futures universe from configs/futures_universe.yaml
2. Determines every unique contract ticker (F1 + F2) for every rebalance date
3. Batch-downloads all contract price series via yfinance
4. Saves each series as a Parquet file under data/raw/contracts/
5. Builds the monthly carry signal table (data/raw/carry/carry_table.parquet)
6. Downloads continuous front-month series for all markets (data/raw/continuous_returns.parquet)
7. Prints a summary of data availability

CME contract tickers follow the Yahoo Finance format:
    {ROOT}{MONTH_CODE}{2-digit year}.{EXCHANGE}
    e.g.  CLK25.NYM  (Crude Oil May 2025, NYMEX)
         GCZ24.CMX  (Gold Dec 2024, COMEX)
         ZCH25.CBT  (Corn Mar 2025, CBOT)
"""

import argparse
import logging
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import build_carry_table, build_return_series, load_config
from src.signals.carry import carry_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download CME futures data for carry strategy")
    parser.add_argument("--force", action="store_true", help="Force re-download (ignore cache)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    config = load_config("configs/futures_universe.yaml")
    strat_cfg = config.get("strategy", {})

    start = args.start or strat_cfg.get("start_date", "2022-01-01")
    end = args.end or strat_cfg.get("end_date", "2026-03-31")

    logger.info(f"Backtest window: {start} → {end}")
    logger.info(f"Universe: {list(config['futures'].keys())}")

    # ── Step 1: Build carry table (downloads all specific contract months) ───
    logger.info("\n=== Step 1: Building carry signal table ===")
    carry_table = build_carry_table(config, start=start, end=end, force=args.force)
    logger.info(f"Carry table shape: {carry_table.shape}")

    # ── Step 2: Download continuous front-month returns ──────────────────────
    logger.info("\n=== Step 2: Downloading continuous front-month series ===")
    returns = build_return_series(config, start=start, end=end, force=args.force)
    logger.info(f"Returns shape: {returns.shape}")

    # ── Step 3: Data quality report ──────────────────────────────────────────
    logger.info("\n=== Step 3: Data quality summary ===")

    print("\n── Carry Table Coverage ─────────────────────────────────────────")
    print(f"  Rebalance dates: {len(carry_table)}")
    print(f"  Markets: {list(carry_table.columns)}")
    print(f"\n  NaN rate per market:")
    for mkt in carry_table.columns:
        nan_pct = carry_table[mkt].isna().mean() * 100
        status = "✓" if nan_pct < 30 else "⚠" if nan_pct < 70 else "✗"
        print(f"    {status} {mkt}: {nan_pct:.0f}% missing")

    print("\n── Carry Signal Summary ─────────────────────────────────────────")
    summary = carry_summary(carry_table)
    print(summary[["mean_carry_%", "std_carry_%", "pct_backwardation", "n_obs"]].to_string())

    print("\n── Return Series Coverage ───────────────────────────────────────")
    for mkt in returns.columns:
        n = returns[mkt].replace(0, pd.NA).dropna().__len__()
        print(f"  {mkt}: {n} trading days")

    print("\n✅  Data fetch complete. Run `python scripts/run_backtest.py` to start the backtest.")


if __name__ == "__main__":
    main()

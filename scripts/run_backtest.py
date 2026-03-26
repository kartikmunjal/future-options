#!/usr/bin/env python
"""
run_backtest.py — Run the CME futures carry strategy backtest.

Usage
-----
    python scripts/run_backtest.py [options]

Options
-------
--start        : Start date (YYYY-MM-DD, default from config)
--end          : End date (YYYY-MM-DD, default from config)
--n_long       : Long positions per rebalance (default 4)
--n_short      : Short positions per rebalance (default 4)
--target_vol   : Annual vol target, e.g. 0.15 (default from config)
--cost_bps     : Round-trip transaction cost in bps (default 1.0)
--sector_neutral : Use sector-demeaned carry signal
--no_plot      : Skip generating charts

Strategy summary
----------------
Each month on the first business day:
  1. Compute annualised carry = log(F1/F2) × 12 for each market.
  2. Rank markets by carry (cross-sectionally or sector-neutral).
  3. Go long the top N markets (backwardated) and short the bottom N (contangoed).
  4. Size positions inversely proportional to 60-day realised vol (vol parity).
  5. Scale gross book to target annual vol.
  6. Hold until next rebalance, deducting 1bps round-trip cost on turnover.
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.data.loader import build_carry_table, build_return_series, load_config
from src.signals.carry import (
    carry_return_correlation,
    carry_summary,
    cross_sectional_zscore,
    sector_neutral_zscore,
)
from src.portfolio.construction import build_weights, portfolio_summary
from src.backtest.engine import run_backtest, sector_attribution, rolling_sharpe
from src.reporting.metrics import (
    compute_metrics,
    print_metrics,
    plot_performance,
    plot_sector_attribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="CME futures carry strategy backtest")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--n_long", type=int, default=None)
    parser.add_argument("--n_short", type=int, default=None)
    parser.add_argument("--target_vol", type=float, default=None)
    parser.add_argument("--cost_bps", type=float, default=None)
    parser.add_argument("--sector_neutral", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config("configs/futures_universe.yaml")
    strat_cfg = config.get("strategy", {})
    fut_cfg = config["futures"]

    start = args.start or strat_cfg.get("start_date", "2022-01-01")
    end = args.end or strat_cfg.get("end_date", "2026-03-31")
    n_long = args.n_long or strat_cfg.get("n_long", 4)
    n_short = args.n_short or strat_cfg.get("n_short", 4)
    target_vol = args.target_vol or strat_cfg.get("target_annual_vol", 0.15)
    cost_bps = args.cost_bps if args.cost_bps is not None else strat_cfg.get("transaction_cost_bps", 1.0)
    vol_lookback = strat_cfg.get("vol_lookback_days", 60)

    sector_map = {mkt: spec["sector"] for mkt, spec in fut_cfg.items()}

    logger.info("=" * 60)
    logger.info("  CME FUTURES CARRY STRATEGY BACKTEST")
    logger.info("=" * 60)
    logger.info(f"  Period     : {start} → {end}")
    logger.info(f"  Universe   : {list(fut_cfg.keys())}")
    logger.info(f"  Long/Short : {n_long}L / {n_short}S")
    logger.info(f"  Target vol : {target_vol:.0%}")
    logger.info(f"  Cost       : {cost_bps} bps")
    logger.info(f"  Signal     : {'sector-neutral' if args.sector_neutral else 'cross-sectional'}")
    logger.info("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    logger.info("\n[1/5] Loading carry signals ...")
    carry_table = build_carry_table(config, start=start, end=end)
    logger.info(f"      {carry_table.shape[0]} rebalance dates, "
                f"{carry_table.notna().sum(axis=1).mean():.1f} avg markets/date")

    logger.info("\n[2/5] Loading continuous returns ...")
    returns = build_return_series(config, start=start, end=end)
    # Only keep markets present in both carry and returns
    common_mkts = [m for m in carry_table.columns if m in returns.columns]
    carry_table = carry_table[common_mkts]
    returns = returns[common_mkts]
    logger.info(f"      {returns.shape[0]} trading days, {len(common_mkts)} markets")

    # ── Signal ───────────────────────────────────────────────────────────────
    if args.sector_neutral:
        carry_z = sector_neutral_zscore(carry_table, sector_map)
    # For portfolio construction we pass raw carry (not z-scored) since
    # ranking is done inside build_weights. Z-scores are for research.

    # ── Portfolio ────────────────────────────────────────────────────────────
    logger.info("\n[3/5] Constructing portfolio weights ...")
    weights = build_weights(
        carry_signals=carry_table,
        returns=returns,
        n_long=n_long,
        n_short=n_short,
        target_vol=target_vol,
        vol_lookback=vol_lookback,
    )
    port_stats = portfolio_summary(weights)
    logger.info(f"      Avg gross leverage : {port_stats['avg_gross_leverage']:.2f}x")
    logger.info(f"      Days invested      : {port_stats['days_invested_%']:.1f}%")
    logger.info(f"      Avg positions      : {port_stats['avg_n_positions']:.1f}")

    # ── Backtest ──────────────────────────────────────────────────────────────
    logger.info("\n[4/5] Running backtest ...")
    net_ret, gross_ret, turnover = run_backtest(
        weights=weights,
        returns=returns,
        transaction_cost_bps=cost_bps,
    )

    # Sector attribution
    sec_pnl = sector_attribution(weights, returns, sector_map)
    sec_pnl_annual = sec_pnl.sum() * 252 / len(sec_pnl) * 100

    # ── Reporting ─────────────────────────────────────────────────────────────
    logger.info("\n[5/5] Computing performance metrics ...")
    # Futures are fully-collateralized; Sharpe vs 0% is standard for carry strategies
    metrics = compute_metrics(net_ret, risk_free_annual=0.00)
    print_metrics(metrics, title="CME Futures Carry Strategy — Net Performance")

    # Carry signal diagnostics
    print("\n── Carry Signal Summary ─────────────────────────────────────────")
    cs = carry_summary(carry_table)
    print(cs[["mean_carry_%", "pct_backwardation", "autocorr_1m", "n_obs"]].to_string())

    print("\n── Carry → Return Predictability ────────────────────────────────")
    crc = carry_return_correlation(carry_table, returns, forward_months=1)
    if not crc.empty and "carry_ret_corr" in crc.columns:
        print(crc[["carry_ret_corr", "t_stat", "hit_rate_%"]].to_string())
    else:
        print("  (insufficient overlapping observations)")

    print("\n── Sector P&L Attribution (annualised %) ────────────────────────")
    for sector, pnl in sec_pnl_annual.sort_values(ascending=False).items():
        bar = "█" * max(0, int(abs(pnl) / 0.2)) + ("" if pnl >= 0 else "")
        sign = "+" if pnl >= 0 else ""
        print(f"  {sector:<10} {sign}{pnl:5.1f}%  {bar}")

    print("\n── Annual Returns ────────────────────────────────────────────────")
    annual_rets = (1 + net_ret).resample("YE").prod() - 1
    for yr, ret in annual_rets.items():
        sign = "+" if ret >= 0 else ""
        bar = "█" * int(abs(ret) * 200)
        print(f"  {yr.year}:  {sign}{ret:.1%}  {bar}")

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    net_ret.to_csv("results/net_returns.csv", header=["net_return"])
    weights.to_csv("results/weights.csv")
    carry_table.to_csv("results/carry_signals.csv")
    logger.info("Results saved to results/")

    # ── Charts ────────────────────────────────────────────────────────────────
    if not args.no_plot:
        logger.info("Generating performance charts ...")
        plot_performance(
            net_returns=net_ret,
            gross_returns=gross_ret,
            turnover=turnover,
            carry_signals=carry_table,
            sector_pnl=sec_pnl,
            save_path="results/carry_strategy_report.png",
        )
        plot_sector_attribution(
            sector_pnl=sec_pnl,
            save_path="results/sector_attribution.png",
        )

    logger.info("\n✅  Backtest complete.")


if __name__ == "__main__":
    main()

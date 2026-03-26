"""
Performance metrics and visualisation for the futures carry strategy.

Outputs
-------
* Tabular summary printed to console
* Multi-panel performance chart saved to results/carry_strategy_report.png
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    returns: pd.Series,
    risk_free_annual: float = 0.05,
    periods_per_year: int = 252,
) -> Dict:
    """
    Compute standard quant performance metrics.

    Parameters
    ----------
    returns          : Daily net return series
    risk_free_annual : Annualised risk-free rate for Sharpe/Sortino
    periods_per_year : Trading days per year (default 252)

    Returns
    -------
    dict of metrics
    """
    rfr_daily = risk_free_annual / periods_per_year
    excess = returns - rfr_daily

    annual_ret = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = excess.mean() * periods_per_year / (excess.std() * np.sqrt(periods_per_year))

    # Sortino (downside deviation only)
    downside = returns[returns < rfr_daily]
    sortino = (
        annual_ret / (downside.std() * np.sqrt(periods_per_year))
        if len(downside) > 1 else np.nan
    )

    # Drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.expanding().max()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    calmar = annual_ret / abs(max_dd) if max_dd != 0 else np.nan

    # Win stats
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Monthly returns
    monthly = (1 + returns).resample("ME").prod() - 1
    best_month = monthly.max()
    worst_month = monthly.min()
    pos_months = (monthly > 0).mean()

    return {
        "Annual Return %": annual_ret * 100,
        "Annual Vol %": annual_vol * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown %": max_dd * 100,
        "Calmar Ratio": calmar,
        "Win Rate % (daily)": win_rate * 100,
        "Profit Factor": profit_factor,
        "Best Month %": best_month * 100,
        "Worst Month %": worst_month * 100,
        "Positive Months %": pos_months * 100,
        "Skewness": returns.skew(),
        "Excess Kurtosis": returns.kurtosis(),
        "Total Return %": (cum.iloc[-1] - 1) * 100,
        "N Trading Days": len(returns),
    }


def print_metrics(metrics: Dict, title: str = "Carry Strategy Performance") -> None:
    """Print formatted metrics table."""
    divider = "─" * 45
    print(f"\n{divider}")
    print(f"  {title}")
    print(divider)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<30} {val:>10.2f}")
        else:
            print(f"  {key:<30} {val:>10}")
    print(divider)


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_performance(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    turnover: pd.Series,
    carry_signals: pd.DataFrame,
    sector_pnl: Optional[pd.DataFrame] = None,
    save_path: str = "results/carry_strategy_report.png",
) -> None:
    """
    Generate a six-panel performance report.

    Panels
    ------
    1. Equity curve (net vs gross)
    2. Drawdown
    3. Carry signal heatmap (markets × time)
    4. Average carry by market (sorted bar chart)
    5. Rolling 1-year Sharpe
    6. Monthly returns calendar heatmap
    """
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("CME Futures Carry Strategy — Performance Report", fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # ── Panel 1: Equity curve ────────────────────────────────────────────────
    cum_net = (1 + net_returns).cumprod()
    cum_gross = (1 + gross_returns).cumprod()
    ax1.plot(cum_net.index, cum_net.values, "steelblue", lw=1.5, label="Net")
    ax1.plot(cum_gross.index, cum_gross.values, "lightsteelblue", lw=1.0, ls="--", label="Gross")
    ax1.axhline(1.0, color="gray", lw=0.5)
    ax1.set_title("Equity Curve (starting $1)")
    ax1.set_ylabel("Portfolio Value")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.legend(fontsize=8)

    # ── Panel 2: Drawdown ────────────────────────────────────────────────────
    running_max = cum_net.expanding().max()
    dd = (cum_net - running_max) / running_max
    ax2.fill_between(dd.index, dd.values * 100, 0, alpha=0.7, color="crimson")
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    # ── Panel 3: Carry signal heatmap ────────────────────────────────────────
    # Z-score each market's carry time-series (for comparable colour scale)
    carry_z = carry_signals.apply(
        lambda col: (col - col.mean()) / col.std() if col.std() > 0 else col
    )
    carry_z = carry_z.T  # markets as rows, dates as columns
    # Downsample to at most 60 columns for readability
    if carry_z.shape[1] > 60:
        step = carry_z.shape[1] // 60
        carry_z = carry_z.iloc[:, ::step]
    im = ax3.imshow(
        carry_z.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-2.5,
        vmax=2.5,
        interpolation="nearest",
    )
    ax3.set_yticks(range(len(carry_z.index)))
    ax3.set_yticklabels(carry_z.index, fontsize=7)
    n_xticks = min(8, carry_z.shape[1])
    step = max(1, carry_z.shape[1] // n_xticks)
    ax3.set_xticks(range(0, carry_z.shape[1], step))
    ax3.set_xticklabels(
        [carry_z.columns[i].strftime("%b'%y") for i in range(0, carry_z.shape[1], step)],
        rotation=45,
        fontsize=7,
    )
    plt.colorbar(im, ax=ax3, label="Carry Z-score")
    ax3.set_title("Carry Signal Heatmap (backwardation=green, contango=red)")

    # ── Panel 4: Average carry by market ─────────────────────────────────────
    avg_carry = carry_signals.mean().sort_values() * 100
    colors = ["crimson" if v < 0 else "seagreen" for v in avg_carry.values]
    ax4.barh(avg_carry.index, avg_carry.values, color=colors, edgecolor="white", height=0.6)
    ax4.axvline(0, color="black", lw=0.8)
    ax4.set_title("Average Annualised Carry by Market")
    ax4.set_xlabel("Carry (%/year)")
    ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

    # ── Panel 5: Rolling Sharpe ───────────────────────────────────────────────
    roll_sharpe = (
        net_returns.rolling(252).mean() * 252 /
        (net_returns.rolling(252).std() * np.sqrt(252))
    )
    ax5.plot(roll_sharpe.index, roll_sharpe.values, "darkgreen", lw=1.2)
    ax5.axhline(0, color="gray", lw=0.5)
    ax5.axhline(1, color="steelblue", lw=0.8, ls="--", label="Sharpe = 1")
    ax5.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=roll_sharpe.values > 0, alpha=0.2, color="green")
    ax5.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                     where=roll_sharpe.values < 0, alpha=0.2, color="red")
    ax5.set_title("Rolling 1-Year Sharpe Ratio")
    ax5.legend(fontsize=8)

    # ── Panel 6: Monthly returns heatmap ─────────────────────────────────────
    monthly = (1 + net_returns).resample("ME").prod() - 1
    mdf = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    ).pivot(index="year", columns="month", values="ret")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    # mdf.columns are integer month numbers (1-12)
    mdf.columns = [month_names[int(c) - 1] for c in mdf.columns]

    sns.heatmap(
        mdf * 100,
        ax=ax6,
        cmap="RdYlGn",
        center=0,
        vmin=-8,
        vmax=8,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Return (%)"},
    )
    ax6.set_title("Monthly Net Returns (%)")
    ax6.set_xlabel("")
    ax6.set_ylabel("")

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nReport saved to {save_path}")
    plt.close(fig)


def plot_sector_attribution(
    sector_pnl: pd.DataFrame,
    save_path: str = "results/sector_attribution.png",
) -> None:
    """
    Stacked bar chart of cumulative P&L contribution by sector.
    """
    cum_sector = sector_pnl.resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    cum_sector.cumsum().plot(ax=ax, kind="area", stacked=False, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Cumulative P&L Attribution by Sector")
    ax.set_ylabel("Cumulative Return")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.legend(title="Sector")
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Sector attribution saved to {save_path}")
    plt.close(fig)

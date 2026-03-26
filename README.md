# CME Futures Carry (Roll Yield) Strategy

A systematic cross-sectional carry trade on 11 CME futures markets using publicly
available data from Yahoo Finance (CME/NYMEX/COMEX/CBOT contracts).

---

## Strategy Overview

**Carry** (also called *roll yield*) is the expected return earned by mechanically
rolling a long futures position down the term structure.

```
carry_i(t) = log(F1_i / F2_i) × (12 / ΔMonths)   [annualised]
```

| Term structure shape | Carry | Rolling long… |
|----------------------|-------|---------------|
| Backwardation F1 > F2 | **+** | earns roll yield |
| Contango  F1 < F2    | **−** | bleeds roll yield |

**Portfolio logic (monthly rebalance):**
1. Compute annualised carry for every market.
2. Long the **top 4** markets (highest backwardation → positive roll yield).
3. Short the **bottom 4** markets (deepest contango → negative roll yield).
4. Weight each leg inversely proportional to 60-day realised volatility (vol parity).
5. Scale gross book so total portfolio targets **15% annual vol**.
6. Deduct **1 bps round-trip** transaction cost on turnover.

---

## Universe — 11 CME Markets

| Sector | Name | Root | Exchange |
|--------|------|------|----------|
| Energy | Crude Oil WTI | CL | NYMEX |
| Energy | Natural Gas | NG | NYMEX |
| Energy | Heating Oil (ULSD) | HO | NYMEX |
| Metals | Gold | GC | COMEX |
| Metals | Silver | SI | COMEX |
| Metals | Copper | HG | COMEX |
| Grains | Corn | ZC | CBOT |
| Grains | Wheat (SRW) | ZW | CBOT |
| Grains | Soybeans | ZS | CBOT |
| Rates  | 10-Year T-Note | ZN | CBOT |
| Rates  | 30-Year T-Bond | ZB | CBOT |

---

## Data Sources

All data is sourced from **Yahoo Finance** via `yfinance` (free, no API key required).

Two types of contracts are downloaded:

| Type | Ticker format | Example | Purpose |
|------|--------------|---------|---------|
| Specific month | `{ROOT}{MC}{YY}.{EX}` | `CLK25.NYM` | Carry signal (F1/F2 spread) |
| Continuous front-month | `{ROOT}=F` | `CL=F` | Daily returns for P&L |

CME month codes: **F**=Jan **G**=Feb **H**=Mar **J**=Apr **K**=May **M**=Jun
**N**=Jul **Q**=Aug **U**=Sep **V**=Oct **X**=Nov **Z**=Dec

---

## Installation

```bash
git clone https://github.com/kartikmunjal/future-options.git
cd future-options
pip install -r requirements.txt
```

---

## Usage

### 1. Fetch & cache all data
```bash
python scripts/fetch_data.py
```
Downloads ~400–600 individual contract price series and builds the monthly carry
table. Data is cached under `data/raw/` as Parquet files. Re-runs are instant.

```bash
python scripts/fetch_data.py --force   # force re-download
```

### 2. Run the backtest
```bash
python scripts/run_backtest.py
```

Key CLI flags:
```
--start 2022-01-01   # backtest start (default from config)
--end   2026-03-31   # backtest end
--n_long  4          # number of long positions
--n_short 4          # number of short positions
--target_vol 0.15    # annual vol target
--cost_bps 1.0       # round-trip transaction cost
--sector_neutral     # use sector-demeaned carry signal
--no_plot            # skip chart generation
```

### Example: sector-neutral, 2 bps costs
```bash
python scripts/run_backtest.py --sector_neutral --cost_bps 2.0
```

---

## Project Structure

```
future-options/
├── configs/
│   └── futures_universe.yaml      # Product specs, strategy params
├── data/
│   └── raw/                       # Cached Parquet files (gitignored)
├── results/                       # Backtest output (charts, CSVs)
├── src/
│   ├── data/
│   │   ├── contracts.py           # Contract calendar utilities
│   │   └── loader.py              # yfinance downloader + cache
│   ├── signals/
│   │   └── carry.py               # Roll-yield computation & diagnostics
│   ├── portfolio/
│   │   └── construction.py        # Vol-parity long/short builder
│   ├── backtest/
│   │   └── engine.py              # P&L engine + sector attribution
│   └── reporting/
│       └── metrics.py             # Metrics & 6-panel chart
└── scripts/
    ├── fetch_data.py              # Data download script
    └── run_backtest.py            # Main backtest runner
```

---

## Key Concepts

### Why carry works in commodities
Commodity producers need to hedge forward production → they are natural
sellers of futures → futures trade at a discount to spot (backwardation)
→ long speculators earn the *insurance premium* (roll yield).

### Why carry works in rates
A 10-year bond futures contract has positive carry when the yield curve slopes
upward: the holder earns carry by rolling from the (higher-yielding) spot bond
into the (cheaper, lower-yielding) front contract.

### Risk factors
| Risk | Mitigation |
|------|------------|
| Supply shocks (oil, gas) | Diversification across sectors |
| Trend reversals (contango → backwardation) | Monthly rebalance |
| Crowded positioning | Vol targeting limits gross leverage |
| Correlation spikes | Dollar-neutral construction |

---

## References

- Erb & Harvey (2006). *The Strategic and Tactical Value of Commodity Futures.* FAJ.
- Gorton & Rouwenhorst (2006). *Facts and Fantasies about Commodity Futures.* FAJ.
- Asness, Moskowitz & Pedersen (2013). *Value and Momentum Everywhere.* JF.
- Koijen, Moskowitz, Pedersen & Vrugt (2018). *Carry.* JFE.

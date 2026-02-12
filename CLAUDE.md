# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase studying **price pressure in equity markets** by decomposing information-driven from liquidity-driven trading pressure using market microstructure analysis. Empirical focus: earnings announcements in S&P 500 equities (2023-2024, ~4,200 events). Data sourced from WRDS TAQ (intraday trades/quotes), CRSP (daily returns), and Compustat (earnings calendar).

## Setup

- Python 3.12 (see `.python-version`)
- Virtual environment: `source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- WRDS credentials required via `.pgpass` or `.env` file

## Running the Pipeline

The pipeline runs sequentially — each step depends on the previous:

```bash
python src/pipeline/production/build_daily_features.py   # TAQ → daily microstructure features
python src/pipeline/production/add_abnormal_volume.py    # Enrich with CRSP volume baselines
python src/pipeline/production/build_event_features.py   # Aggregate daily → event-level features
```

Alternative methods can be run independently:
```bash
python Methods_Sung_Cho/A_build_event_features.py   # Methods A through D
python Methods_June_Lee/method_1.py                  # Methods 1 through 4
```

Notebooks in `notebooks/` (01-04) are for exploration and validation — run via `jupyter lab`.

## Architecture

### Data Flow

```
WRDS TAQ + CRSP + Compustat
  → src/data_loaders/ (taq.py, crsp.py, earnings.py)
    → src/pipeline/production/build_daily_features.py
      → data/processed/daily_features.parquet (one row per stock-day)
        → src/pipeline/production/build_event_features.py
          → data/processed/event_features.parquet (one row per event)
            → src/backtest/event_backtester.py (quintile portfolios, post-event returns)
```

### Core Modules (`src/`)

- **`data_loaders/`** — WRDS data access: `taq.py` (intraday trades/quotes), `crsp.py` (daily returns), `earnings.py` (announcement calendar from Compustat)
- **`features/`** — Microstructure feature computation:
  - `order_flow.py` — Lee-Ready trade classification, Order Flow Imbalance (OFI normalized to [-1,1])
  - `liquidity.py` — Quoted spread, spread stability (CV), tail spread (p95); uses 1-second resampling
  - `volume.py` — Dollar/share volume, abnormal volume ratio, morning concentration
- **`backtest/event_backtester.py`** — Earnings-specific backtester: quintile assignment by pressure_score, post-event cumulative returns over [0,+5] and [0,+20] day horizons
- **`backtester.py`** — Generic event-driven PnL engine with execution lag and transaction costs
- **`pipeline/production/`** — Main pipeline scripts (see "Running the Pipeline")
- **`pipeline/experiments/`** — Batch processing and test variants
- **`pipeline/utils/`** — Merge utilities and distribution diagnostics

### Research Methods (Parallel Branches)

- **`Methods_Sung_Cho/`** — Four pressure aggregation variants (A-D), each producing separate `{letter}_event_features.parquet`
- **`Methods_June_Lee/`** — Four analytical methods (1-4) exploring persistence/reversal decomposition (PI/RI components), plus visualization scripts

### Key Concepts

- **Pressure Score**: `tanh(InfoScore - LiqScore)` ∈ [-1, +1]. Information score increases with OFI persistence + spread stability; liquidity score increases with volume bursts + spread widening.
- **Pre-event window**: [-10, -1] trading days before earnings announcement
- **Feature contracts**: Feature functions return frozen dicts for pipeline integration
- **Event table**: `data/processed/event_table.parquet` maps event_id → permno, ticker, announcement date

## Key Constraints

- Raw data is never committed (stored in `data/raw/`, gitignored)
- Processed parquet files live in `data/processed/`
- No test suite, linter, or CI/CD currently configured
- No Makefile or task runner — scripts are run directly

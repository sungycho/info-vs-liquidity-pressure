# Microstructure Pressure Decomposition

## Project Overview
This project studies **market microstructure–driven price pressure** and its decomposition into economically interpretable components, with a concrete empirical application to **earnings-related price dynamics in U.S. equities**.

The core objective is to build a **clean, modular research and backtesting framework** that cleanly separates:
- data acquisition,
- feature construction,
- signal generation,
- portfolio construction, and
- evaluation,

so that **dummy signals can be replaced with real signals without refactoring the system**.

This repository is intentionally designed as a **research-grade quant project**, not a one-off notebook or toy backtest.

---

## Motivation
Short-horizon price movements—especially around discrete events such as earnings announcements—are heavily influenced by market microstructure effects, including:
- order flow imbalance,
- temporary price impact,
- liquidity provision and withdrawal,
- volatility and liquidity regime shifts.

Naively treating these movements as alpha often leads to overfitting and fragile results.  
Instead, this project aims to:

1. **Decompose observed price pressure** into components attributable to microstructure frictions.
2. **Condition signals on events and regimes**, rather than relying on unconditional predictors.
3. **Evaluate robustness** using a transparent, extensible backtesting engine.

---

## Research Goals
The project is structured around three concrete research goals:

- **Economic interpretability**  
  Ensure that features and signals correspond to well-defined microstructure mechanisms rather than black-box predictors.

- **Modular experimentation**  
  Enable rapid iteration on signals, features, and assumptions without rewriting downstream code.

- **Reproducible evaluation**  
  Make all assumptions explicit and results easy to audit, replicate, and extend.

---

## System Design Philosophy
The system is built around the following principles:

- **Strict separation of concerns**  
  Data loading, feature engineering, signal logic, and backtesting are independent layers.

- **Interface-first design**  
  Each component interacts through well-defined interfaces, not concrete implementations.

- **Swapability**  
  Dummy signals are used to validate the engine early and can later be replaced by real signals without touching the backtest logic.

- **Scalability**  
  The same framework should work for small research samples and larger datasets.

- **Research clarity**  
  Code structure reflects economic intent, not just computational convenience.

---

## Backtesting Framework (Conceptual)
The backtesting engine is designed as an **event-driven simulation**, where:

- events define *when* the strategy acts,
- signals define *what positions to take*,
- the portfolio module handles *position management and PnL accounting*,
- transaction costs and constraints are modeled explicitly.

Performance evaluation focuses on both:
- **portfolio-level metrics** (Sharpe, drawdowns, turnover), and
- **diagnostic analysis** (stability across regimes, sensitivity to assumptions).

---

## Data Sources
Planned and supported data sources include:

- **TAQ**: intraday trades and quotes for microstructure feature construction  
- **CRSP**: daily returns, volumes, and identifiers  
- **Earnings data**: announcement timing and surprises  
- **Compustat** (optional): firm fundamentals for extensions  

Raw data is never committed to the repository.

---

## Current Status
- [x] Research scope and system architecture defined  
- [x] Backtesting engine specification completed  
- [ ] Data loaders implementation  
- [ ] Feature engineering modules  
- [ ] Dummy signal integration  
- [ ] Earnings-based empirical application  

---

## Intended Extensions
Potential extensions include:
- applying the framework to other event types (macro announcements, index rebalances),
- alternative microstructure measures,
- execution-aware backtesting,
- regime-dependent transaction cost models.

---

## Disclaimer
This project is for **research and educational purposes only**.  
It does not constitute investment advice and should not be used for live trading.

---

## Author
Sung Cho  
June Lee

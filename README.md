# Microstructure-Based Price Pressure Decomposition

[![DeepWiki](https://img.shields.io/badge/wiki-DeepWiki-blue)](https://deepwiki.com/sungycho/quant-lab)

## Overview

This repository contains a research project studying **price pressure in equity markets through the lens of market microstructure**, with a particular focus on distinguishing **information-driven** from **liquidity-driven** pressure. The primary empirical application examines trading behavior and return dynamics around **earnings announcements** in U.S. equities.

The project is motivated by the observation that short-horizon price movements—especially around discrete information events—are often treated as reduced-form predictors of returns, despite arising from fundamentally different economic mechanisms. This work aims to develop a disciplined framework for decomposing price pressure into economically interpretable components, rather than treating all pressure as homogeneous.

The repository is designed as a **research-grade codebase**, emphasizing clarity, modularity, and economic interpretability. It is not a one-off notebook or a production trading system, but a structured environment for empirical investigation.

---

## Research Motivation

Trading activity around earnings announcements reflects a mixture of mechanisms:

- informed trading and gradual information accumulation,
- liquidity provision and withdrawal,
- temporary price impact from forced or imbalanced trading,
- regime-dependent changes in spreads, volume, and volatility.

Conflating these channels can obscure both interpretation and inference. In particular, a price move driven by informed, patient trading may have very different implications from a move driven by transient liquidity shocks.

This project studies whether **microstructure-based measures**—constructed from order flow, spreads, and volume patterns—can meaningfully separate these mechanisms, and whether the resulting pressure types exhibit different post-event return dynamics.

Earnings announcements serve as a clean empirical laboratory: they are well-defined information events, allow pressure to build in a pre-announcement window, and provide natural post-event horizons for evaluation.

---

## Empirical Setting

- **Universe**: S&P 500 equities  
- **Period**: 2023–2024  
- **Events**: ~4,200 quarterly earnings announcements  
- **Data**: WRDS TAQ (trades & quotes), CRSP daily data, earnings announcement timing

Intraday TAQ data are aggregated to daily microstructure measures and then to event-level summaries over a fixed pre-announcement window. Raw data are never committed to the repository.

---

## Methodological Focus

A central theme of the project is that constructing a composite *pressure score* is not a mechanical exercise. Microstructure features are highly non-Gaussian, heterogeneous across events, and measured in fundamentally different units. As a result:

- scaling choices implicitly determine which events dominate a signal,
- aggregation choices impose an implicit model of how information and liquidity interact,
- normalization decisions affect both economic interpretation and empirical conclusions.

Rather than optimizing for predictive performance, the project treats pressure construction as an **identification problem**, and explicitly explores how different aggregation philosophies correspond to different economic views of price pressure.

---

## System Design Philosophy

The codebase is organized around the following principles:

- **Separation of concerns**  
  Data loading, feature engineering, signal construction, portfolio logic, and evaluation are independent layers.

- **Economic intent first**  
  Code structure reflects economic meaning and research questions, not just computational convenience.

- **Modularity and swapability**  
  Signals, features, and assumptions can be modified without refactoring downstream components.

- **Transparency and reproducibility**  
  All assumptions are explicit, and intermediate outputs are designed to be inspectable and auditable.

This structure supports disciplined experimentation while avoiding over-engineering.

---

## Backtesting Framework (Conceptual)

Evaluation is conducted through an **event-driven backtesting framework**:

- events determine when positions are initiated,
- signals determine directional exposure,
- the portfolio module manages positions and PnL accounting,
- transaction costs and constraints are modeled explicitly.

Performance analysis emphasizes not only portfolio-level metrics (e.g., returns, turnover), but also diagnostic checks such as sensitivity to construction choices and stability across subsamples.

---

## Current Status

- Research question and empirical scope defined
- Modular system architecture implemented
- Data loaders for TAQ, CRSP, and earnings events
- Core microstructure feature construction
- Multiple pressure aggregation approaches implemented
- Earnings-based event study framework

The project is actively under development and should be viewed as an evolving research codebase rather than a finalized empirical paper.

---

## Intended Extensions

Planned or potential extensions include:

- alternative corporate or market events (e.g., index rebalances, macro announcements),
- additional microstructure measures,
- richer transaction cost and liquidity modeling,
- extensions beyond event-based settings.

---

## Disclaimer

This repository is for research and educational purposes only. It does not constitute investment advice and is not intended for live trading.

---

## Authors

Sung Cho  
June Lee


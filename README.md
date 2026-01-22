# Trading Lab — Quantitative Research & Systematic Trading

**Trading Lab** is a research-oriented Python environment for **quantitative time-series analysis**, **volatility modeling**, **factor crafting**, and **systematic trading strategy prototyping**.

The project emphasizes:
- Reproducibility (containerized workflows)
- Clean research architecture (separation of notebooks, models, and data tooling)
- Robust financial time-series engineering
- Statistical rigor (stationarity, heteroskedasticity, diagnostics)
- Scalable data ingestion with caching and provider abstraction

This repository serves both as a **quant research notebook lab** and a **foundation for production-grade systematic trading experiments**.

---

## Project Goals

- Study financial time series (prices, returns, volatility)
- Model conditional heteroskedasticity (ARCH / GARCH family)
- Craft factor-adjusted residual series
- Build systematic overlays (volatility targeting, risk parity, etc.)
- Develop a reusable quantitative research framework
- Maintain DevOps discipline (containers, Makefile, automated tests)

---

## Repository Structure

```

trading-lab/
├── Containerfile              # Podman/Docker build definition
├── Makefile                   # DevOps automation (build, run, test)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── notebooks/                 # Research notebooks
│   ├── 00_foundations/
│   ├── 01_crafting/
│   └── 02_strategies/
│
├── trading_lab/               # Core reusable research library
│   ├── data/                  # Data providers, cache, and DataStack builder
│   ├── features/              # Returns, transforms, feature engineering
│   ├── models/                # GARCH & volatility models
│   └── utils/                 # Plotting & helpers
│
├── data/cache/                # Local Parquet cache (gitignored)
├── reports/                   # Figures & exported research outputs
└── tests/                     # Automated test suite (pytest)

````

---

## Key Features

### Data Engineering
- Provider-based architecture (Yahoo Finance by default)
- Builder pattern (`DataStack`) for flexible ingestion pipelines
- Incremental Parquet caching
- One cache file per `(ticker, timeframe)`
- Automatic missing-range detection and extension

### Time Series Research
- Log-return generation
- Stationarity testing (ADF)
- Autocorrelation diagnostics (ACF / PACF)
- ARCH LM testing
- Residual whiteness validation
- Conditional volatility modeling (GARCH family)

### Volatility Modeling
- In-sample GARCH estimation
- Rolling out-of-sample volatility forecasting
- Model diagnostics & statistical validation
- Forecast accuracy metrics (MSE, QLIKE)

### Strategy-Oriented Research
- Volatility forecasting overlays
- Factor residualization
- Regime detection
- Risk-aware systematic trading logic

---

## Running the Project (Podman)

### Build the container

```bash
make build
````

### Launch JupyterLab

```bash
make notebook
```

Then open:

```
http://localhost:8888
```

### Open an interactive shell

```bash
make shell
```

---

## Running Tests

```bash
make test
```

Tests validate:

* DataStack caching behavior
* Provider fetching logic
* Multi-ticker ingestion
* Timeframe-specific cache separation
* Incremental dataset extension

---

## Example: Loading Market Data via DataStack

```python
from trading_lab.data import DataStackBuilder

ds = DataStackBuilder().with_parquet_cache("data/cache").build()

(cac40_df,) = ds.get_ohlcv(
    "^FCHI",
    start="2000-01-01",
    end="2025-12-31",
    timeframe="1d",
)
```

---

## Example: Compute Log Returns

```python
from trading_lab.features.returns import log_returns

returns = log_returns(cac40_df["Adj Close"])
```

---

## Research Focus (Current)

* CAC 40 price dynamics
* Return stationarity analysis
* Volatility clustering detection
* GARCH model calibration and validation
* Rolling volatility forecasting
* Factor-based residual modeling
* Strategy construction based on volatility regimes

---

## Philosophy

This project follows a **research-first** mindset:

* Start with statistical truth before strategy ideas
* Avoid overfitting and spurious signals
* Prefer parsimonious models over complex ones
* Separate alpha discovery from execution logic
* Treat trading strategies as engineered systems, not guesses

---

## Roadmap

Planned extensions include:

* Multi-provider support (FRED, ECB, Stooq)
* Factor regression & residualization engine
* Portfolio backtesting framework
* Risk overlays (vol targeting, drawdown control)
* Execution cost simulation
* Strategy performance reporting dashboards

---

## Disclaimer

This repository is for **research and educational purposes only**.
It does **not** constitute financial advice or a recommendation to trade live capital.

Trading financial instruments carries significant risk.

---

## Author

Quantitative research & system design by the project maintainer.

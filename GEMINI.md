# GEMINI.MD: AI Collaboration Guide

This document provides essential context for AI models interacting with the **Nifty 50 Iron Condor Strategy Engine**. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose

*   **Primary Goal:** An ML-powered quantitative trading system for conservative strike placement in weekly/short-duration Iron Condor options trading on the Nifty 50 (Indian stock index).
*   **Business Domain:** Quantitative Finance / Options Trading.
*   **Key Features:**
    *   **VIX-Aware Scaling:** Dynamic strike buffers based on India VIX levels.
    *   **Quantile Modeling:** Uses LightGBM P10/P90 quantile regressors to predict price ranges.
    *   **Volatility Modeling:** GARCH(1,1) conditional volatility for breach probability estimation.
    *   **Conformal Prediction:** MAPIE-based calibration ensuring ≥85-90% coverage on prediction intervals.
    *   **Walk-Forward Backtest:** Historical P&L simulation with risk metrics (Sharpe, Drawdown, POP).

## 2. Core Technologies & Stack

*   **Language:** Python 3.10+
*   **Data Handling:** `pandas`, `numpy`, `pyarrow` (Parquet storage), `yfinance` (Market data).
*   **ML & Modeling:** `lightgbm` (Quantile regression), `scikit-learn`, `shap` (Explainability).
*   **Statistics & Volatility:** `arch` (GARCH), `scipy`, `statsmodels`.
*   **Calibration:** `MAPIE` (Conformal prediction).
*   **Infrastructure:** `python-dotenv` (Configuration), `loguru` (Logging), `joblib` (Serialization).

## 3. Architectural Patterns

*   **Overall Architecture:** Modular Pipeline (M1 through M8).
    *   `M1: Data Pipeline` — Fetch and aggregate OHLCV/VIX data.
    *   `M2: Feature Engineering` — Compute ATR, Volatility, Bollinger Bands, etc.
    *   `M3: GARCH Volatility` — Fit conditional volatility models.
    *   `M4: Model Training` — Train LightGBM P10/P90 quantile models.
    *   `M5: Calibration` — Apply MAPIE for empirical coverage guarantees.
    *   `M6: Strike Generation` — Core logic for converting predictions to option strikes.
    *   `M7: Backtest` — Walk-forward P&L simulation and metrics.
    *   `M8: Live Pipeline` — Sunday-night orchestration for execution.
*   **Directory Structure Philosophy:**
    *   `/data`: Parquet files for raw and processed data.
    *   `/models`: Serialized model files (`.pkl`).
    *   `/outputs`: Backtest results, charts (PNG), and live strike JSONs.
    *   Root: Modular source files (`module1_...` to `module8_...`) and entrypoint (`run_pipeline.py`).

## 4. Coding Conventions & Style Guide

*   **Formatting:** PEP 8 compliant. Uses `loguru` for structured logging.
*   **Naming Conventions:**
    *   Modules: `moduleX_description.py`
    *   Functions/Variables: `snake_case`
    *   Classes: `PascalCase`
*   **Data Flow Contract:** Modules communicate via Parquet files in `data/` or serialized objects in `models/`.
*   **Error Handling:** Try-except blocks with `loguru.logger.error` and `traceback` printing in the main orchestrator.

## 5. Key Files & Entrypoints

*   **Main Entrypoint:** `run_pipeline.py` (Supports modes: `setup`, `backtest`, `live`, `retrain`).
*   **Core Logic:** `module6_strikes.py` (Strike placement and VIX scaling).
*   **Configuration:** `.env` (via `.env.example`).
*   **Documentation:** `ARCHITECTURE.md` (System design), `CLAUDE.md` (Dev guidance), `CHECKLIST.md` (Deployment).

## 6. Development & Testing Workflow

*   **Local Setup:**
    1.  `python -m venv venv`
    2.  `pip install -r requirements.txt`
    3.  `python run_pipeline.py --mode setup` (Runs M1-M5)
*   **Testing:**
    *   Validated via `python run_pipeline.py --mode backtest`.
    *   Check `outputs/backtest_summary.json` for win rates, Sharpe ratio, and coverage.
*   **CI/CD:** None explicitly detected, but the pipeline is designed for scheduled execution (e.g., Sunday night via cron or Task Scheduler).

## 7. Specific Instructions for AI Collaboration

*   **No Lookahead Bias:** Ensure any changes to `module7_backtest.py` or training logic (`module4_model.py`) strictly maintain the 80/20 chronological split.
*   **Strike Logic:** Any modifications to strike placement should be made in `module6_strikes.py`. Nifty strikes must be rounded to the nearest 50 points using `round_to_50`.
*   **Configuration:** New parameters should be added to `.env.example` and loaded via the `_load_config()` helper in `module6_strikes.py`.
*   **Data Integrity:** Always check for nulls or regime shifts when updating features in `module2_features.py`.
*   **Explainability:** Maintain SHAP importance generation in `module4_model.py` to ensure model decisions remain economically grounded.

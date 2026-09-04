# GEMINI.MD: AI Collaboration Guide

This document provides essential context for AI models interacting with the **Nifty 50 Iron Condor Strategy Engine**. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose

*   **Primary Goal:** An ML-powered quantitative trading system for **conservative strike placement** in weekly/short-duration Iron Condor options trading on the Nifty 50 (Indian stock index).
*   **Business Domain:** Quantitative Finance / Options Trading.
*   **Key Features:**
    *   **VIX-Aware Scaling:** Dynamic strike buffers based on India VIX levels (High VIX → Wider Buffers).
    *   **Quantile Modeling:** Uses per-regime LightGBM P10/P90 quantile regressors to predict price ranges.
    *   **Volatility Modeling:** GJR-GARCH(1,1,1) conditional volatility for breach probability estimation.
    *   **Conformal Prediction:** MAPIE-based calibration ensuring ≥85-90% empirical coverage.
    *   **Walk-Forward Backtest:** Realistic expanding-window P&L simulation with periodic retraining.
    *   **Health Monitoring:** Automated drift detection, coverage decay alerts, and feature distribution tests.

## 2. Core Technologies & Stack

*   **Language:** Python 3.10+
*   **Data Handling:** `pandas`, `numpy`, `pyarrow` (Parquet storage), `yfinance` (Market data).
*   **ML & Modeling:** `lightgbm` (Quantile regression), `scikit-learn`, `shap` (Explainability).
*   **Statistics & Volatility:** `arch` (GARCH), `scipy`, `statsmodels`.
*   **Calibration:** `MAPIE` (Conformal prediction).
*   **Infrastructure:** `pydantic` (Config validation), `python-dotenv`, `loguru` (Logging), `joblib` (Serialization).

## 3. Architectural Patterns

### Overall Architecture: Modular Pipeline (M1 through M8)

The system is organized into sequential modules (M1-M8) orchestrated by `run_pipeline.py`.

*   `M1: Data Pipeline` — Fetch and aggregate OHLCV/VIX data.
*   `M2: Feature Engineering` — Compute ATR, Volatility, Bollinger Bands, and Macro features.
*   `M3: GARCH Volatility` — Fit GJR-GARCH(1,1,1) conditional volatility models.
*   `M4: Model Training` — Train per-regime LightGBM P10/P90 quantile models.
*   `M5: Calibration` — Apply MAPIE for empirical coverage guarantees.
*   `M6: Strike Generation` — Core logic converting predictions to option strikes.
*   `M7: Backtest` — Walk-forward P&L simulation and metrics.
*   `M8: Live Pipeline` — Sunday-night orchestration for weekly execution.
*   `M13: Monitor` — Model drift and coverage decay detection.

### Data Flow Contract

| Module | Reads From | Writes To | Purpose |
|--------|-----------|-----------|---------|
| M1 | Yahoo Finance | `data/*.parquet` | Raw market data storage |
| M2 | M1 outputs | `data/feature_matrix.parquet` | Engineered features |
| M3 | M1, M2 outputs | `models/garch_model.pkl` | Conditional volatility |
| M4 | M3 outputs | `models/lgb_*.pkl` | Quantile models |
| M5 | M4 models | `models/mapie_*.pkl` | Coverage calibration |
| M6 | M4/M5 models | `outputs/strikes_*.json` | Predicted strikes |
| M7 | M6 backtest | `outputs/backtest_*.csv/png`| P&L simulation |

### Directory Structure Philosophy

*   `/data`: Parquet files for raw and processed time-series data.
*   `/models`: Serialized model files (`.pkl`) and metadata (`.json`).
*   `/outputs`: Backtest results, charts, and live strike JSONs.
*   `/tests`: Unit tests (if applicable).
*   Root: Modular source files (`moduleX_...py`) and master entrypoint (`run_pipeline.py`).

## 4. Coding Conventions & Style Guide

*   **Formatting:** PEP 8 compliant. Uses `loguru` for structured, color-coded logging.
*   **Naming Conventions:**
    *   Modules: `moduleX_description.py` (e.g., `module1_data_pipeline.py`).
    *   Functions/Variables: `snake_case`.
    *   Classes: `PascalCase`.
*   **Data Integrity:**
    *   **No Lookahead Bias:** Strictly maintain chronological splits (80/20 for M4).
    *   **Strike Rounding:** All Nifty strikes must be rounded to the nearest 50 points using `round_to_50`.
*   **Error Handling:** Use `try-except` blocks with `loguru.logger.error` and `traceback` in orchestrators.
*   **Config Management:** Load parameters from `.env` via `config.py` using Pydantic `BaseSettings`.

## 5. Key Files & Entrypoints

*   **`run_pipeline.py`**: Master orchestrator. Supported modes:
    *   `--mode setup`: First-time full pipeline run (M1-M5).
    *   `--mode backtest`: Static 80/20 chronological backtest.
    *   `--mode walkforward`: Realistic expanding-window backtest with retraining.
    *   `--mode live`: Sunday-night live strike generation.
    *   `--mode macro`: Fetch latest US/global macro data.
    *   `--mode monitor`: Check for model drift and coverage decay.
    *   `--mode retrain`: Monthly retraining on latest data.
*   **`module6_strikes.py`**: Contains the core strike placement and VIX scaling logic.
*   **`config.py`**: Centralized configuration with range validation.
*   **`data/`**: Key parquets: `feature_matrix_with_garch.parquet`.

## 6. Development & Testing Workflow

### Local Development Environment
1.  `python -m venv venv`
2.  `pip install -r requirements.txt`
3.  `python run_pipeline.py --mode setup` (Runs M1 through M5).

### Testing & Validation
*   **Verification**: Run `python run_pipeline.py --mode backtest`.
*   **Metrics**: Check `outputs/backtest_summary.json` for:
    *   `win_rate_pct` (Target: >90%).
    *   `sharpe_ratio` (Target: >0.4 in walk-forward).
    *   `coverage` (Target: >85% per MAPIE).
*   **Drift Check**: Run `python run_pipeline.py --mode monitor` weekly.

## 7. Specific Instructions for AI Collaboration

### Strike Placement Logic (Mathematical Contract)
Any modifications to `module6_strikes.py` must respect this logic:
1.  **Regime Assignment**: Based on India VIX (Low < 14, Mid 14-18, High > 18).
2.  **Quantile Prediction**: Predicted log-ranges (P10/P90) from LightGBM.
3.  **Buffer Scaling**: `effective_buffer = clamp(base_buffer * (VIX/VIX_baseline), min_buffer, 150)`.
4.  **Strike Calculation**:
    *   `blended_range = (0.7 * P90 + 0.3 * P10) / 2`
    *   `short_put = round_to_50(spot - blended_range - effective_buffer - put_skew)`
    *   `short_call = round_to_50(spot + blended_range + effective_buffer + call_skew)`

### Maintenance & Monitoring
*   **Retraining**: Models should be retrained monthly or when the monitor flags `RETRAIN`.
*   **Explainability**: Maintain SHAP importance generation in `module4_model.py` to ensure signals remain economically grounded (e.g., VIX and Realized Vol should be top features).
*   **Data Quality**: Always validate for nulls or regime shifts when adding features to `module2_features.py`.

### Safety & Risk Mitigations
*   **Skip Condition**: Do not trade if VIX > 22 or expected premium < 20 pts.
*   **Position Sizing**: Max loss should be limited to 1-2% of capital per weekly trade.
*   **Lookahead Protection**: Never use `nifty_weekly.parquet` labels in the training features of `module2_features.py`.

### Commit Messages
*   Follow the Conventional Commits specification (e.g., `feat:`, `fix:`, `docs:`, `perf:`).

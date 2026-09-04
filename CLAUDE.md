# CLAUDE.md

Dev guidance for claude.ai/code working in this repo.

---

## Project Overview

**Nifty 50 Iron Condor Strategy Engine** — ML system predicting weekly price range for Nifty 50 (Indian stock index), generates conservative iron condor strike recommendations using LightGBM quantile models + conformal prediction calibration.

**Output**: Weekly strike levels (short put, short call, long put, long call) with calibrated probability coverage.

---

## Quick Commands

### First-Time Setup (runs M1→M5, ~10–30 min)
```bash
python run_pipeline.py --mode setup
```
Downloads 5 years data, engineers features, fits GARCH, trains LightGBM P10/P90, applies conformal calibration.

### Backtest Historical Performance
```bash
python run_pipeline.py --mode backtest
```
Walk-forward simulation of iron condor P&L. Outputs equity curve, win rate, drawdown to `outputs/`.

### Generate This Week's Strikes (Sunday night)
```bash
python run_pipeline.py --mode live
```
Fetches latest data, recomputes features, runs inference, outputs strikes to `outputs/strikes_live.json`.

### Retrain Models on Latest Data
```bash
python run_pipeline.py --mode retrain
```
Re-fit GARCH, LightGBM, and MAPIE without modifying backtest set boundaries.

---

## Architecture & Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│ Module 1: Data Pipeline                                      │
│ Fetch daily OHLCV, intraday, India VIX → parquets           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Module 2: Feature Engineering                                │
│ ATR, volatility, VIX, range, Bollinger Bands, calendar      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Module 3: GARCH(1,1) Volatility                             │
│ Fit conditional volatility, extract σ(t) weekly             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Module 4: LightGBM Quantile Models                          │
│ Train P10 & P90 quantile regressors (80% train, 20% test)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Module 5: Conformal Calibration (MAPIE)                     │
│ Guarantee ≥85% coverage on prediction intervals             │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────────┐          ┌──────────────────────┐
│ Module 6:            │          │ Module 7:            │
│ Strike Generator     │          │ Backtester           │
│ Convert predictions  │          │ Walk-forward P&L     │
│ → IC strikes        │          │ equity curve, metrics│
└──────────────────────┘          └──────────────────────┘
        ↑                                       
        └───────────────────┬───────────────────┘
                            ↓
                ┌──────────────────────┐
                │ Module 8: Live       │
                │ Sunday-night runner  │
                │ (reuses M1,2,3,5,6) │
                └──────────────────────┘
```

### Data Flow Contract

| Module | Reads From | Writes To | Purpose |
|--------|-----------|-----------|---------|
| M1 | Yahoo Finance / Kite API | `data/nifty_daily.parquet`, `data/nifty_intraday.parquet`, `data/india_vix_daily.parquet`, `data/nifty_weekly.parquet` | Fetch and store OHLCV history |
| M2 | M1 outputs (weekly data) | `data/feature_matrix.parquet` | Compute ATR, volatility, VIX features |
| M3 | M1 (daily), M2 outputs | `data/feature_matrix_with_garch.parquet`, `models/garch_model.pkl` | Extract GARCH conditional σ |
| M4 | M3 output (feature matrix) | `models/lgbm_p10.pkl`, `models/lgbm_p90.pkl`, `models/feature_columns.pkl`, `outputs/model_evaluation.json` | Train P10 & P90 quantile models |
| M5 | M4 models, M3 feature matrix | `models/mapie_calibrated.pkl`, `outputs/calibration_report.json` | Calibrate prediction intervals |
| M6 | M4/M5 models, live Nifty spot | `outputs/strikes_YYYY-MM-DD.json` | Generate iron condor strikes |
| M7 | M3/M4 data, M6 backtest | `outputs/backtest_results.csv`, `outputs/backtest_equity_curve.png`, `outputs/backtest_summary.json` | Simulate P&L, compute metrics |
| M8 | M1,M2,M3,M5 (incremental) | `outputs/strikes_live.json`, console log | Real-time Sunday-night predictions |

---

## Key Configuration (.env)

```bash
STRIKE_BUFFER_POINTS=50        # Points beyond P10/P90 to shift strikes
WING_WIDTH_POINTS=200          # Max loss width for iron condor
TARGET_COVERAGE=0.85           # Minimum coverage guarantee

# Optional: Zerodha Kite API (currently falls back to Yahoo Finance)
# KITE_API_KEY=...
# KITE_API_SECRET=...
# KITE_ACCESS_TOKEN=...
```

---

## Critical Design Decisions

### Time-Series Split (No Lookahead Bias)
- M4: 80% train (oldest), 20% test (last).
- M5: Calibration on test set only → generalization.
- M7: Walk-forward backtester never trains on future data.

### Conformal Prediction (MAPIE)
- Wraps quantile predictions with empirical coverage guarantees.
- If coverage < target, intervals widen automatically.
- See `outputs/calibration_report.json` for coverage @ 80%, 85%, 90%.

### Strike Rounding
- Nifty rounds to nearest 50 points (e.g., 22050, 22100, 22150).
- `module6_strikes.py:round_to_strike(price, interval=50)` handles this.

### Feature Importance & SHAP
- M4 saves `outputs/shap_importance.png` — top features driving P10/P90.
- Validate model relies on economically sensible features (VIX, realized vol, ATR, not noise).

### Backtest Metrics
- **Win rate**: % weeks Nifty closed within [short_put, short_call].
- **Max drawdown**: Largest peak-to-trough decline in equity curve.
- **Sharpe ratio**: Risk-adjusted returns (annualized).
- **Breach rate by VIX regime**: How often strikes failed (low/mid/high VIX).

---

## File Organization

```
research/
├── run_pipeline.py              # Master entry (4 modes)
├── module1_data_pipeline.py     # Fetch & aggregate OHLCV
├── module2_features.py          # Feature engineering
├── module3_garch.py             # GARCH volatility
├── module4_model.py             # LightGBM P10/P90 training
├── module5_calibration.py       # MAPIE conformal calibration
├── module6_strikes.py           # Strike generation logic
├── module7_backtest.py          # Historical P&L simulation
├── module8_live.py              # Live Sunday-night runner
├── requirements.txt             # Dependencies
├── .env                         # Config (gitignore'd)
├── .env.example                 # Template
├── CLAUDE.md                    # This file
├── data/                        # Parquet data files
├── models/                      # Serialized models (.pkl)
└── outputs/                     # Charts, backtest results, strike JSONs
```

---

## Development Notes

### Adding New Features
1. Compute in `module2_features.py`, save to `feature_matrix.parquet`.
2. Retrain M4 (models pick up new signal).
3. Re-run `--mode backtest` to validate.
4. Re-calibrate M5 if coverage drops.

### Debugging Model Failures
1. Check `feature_matrix_with_garch.parquet` shape + nulls: `pandas.read_parquet()`.
2. Verify P10 < P90 (quantile regression can violate if misconfigured).
3. Check SHAP plot (`outputs/shap_importance.png`) — sudden feature swaps signal data issues.
4. Check `outputs/backtest_summary.json` — if coverage drops, widen intervals in M5.

### Live Deployment
- Run `python module8_live.py` every Sunday (cron / Task Scheduler).
- Outputs `outputs/strikes_live.json` — parse + execute via broker API.
- Monitor `outputs/backtest_equity_curve.png` weekly — if Sharpe degrades, retrain needed.

---

## Dependencies & Setup

```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --mode setup
```

Key libraries:
- **Data**: `pandas`, `pyarrow` (parquet I/O)
- **Volatility**: `arch` (GARCH), `scipy`, `statsmodels`
- **ML**: `lightgbm`, `scikit-learn`, `shap` (explainability)
- **Calibration**: `MAPIE` (conformal prediction)
- **Utils**: `python-dotenv`, `loguru` (logging), `joblib` (serialization)

---

## Testing & Validation

- **Unit tests**: None (research). Validate via backtest equity curve.
- **Backtest**: `python run_pipeline.py --mode backtest` → check `outputs/backtest_summary.json`.
- **Visual checks**: Review `outputs/shap_importance.png` + calibration curve.
- **Data integrity**: No lookahead bias — backtester uses walk-forward with expanding window.

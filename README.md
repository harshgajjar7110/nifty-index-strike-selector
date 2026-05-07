# Nifty 50 Iron Condor Strategy Engine

ML-powered system for **conservative strike placement** in weekly iron condor options trading on Nifty 50 (Indian stock index). Uses per-VIX-regime LightGBM quantile models + GARCH volatility + macro features + conformal prediction for **calibrated probability guarantees**.

## Quick Start

```bash
# First-time setup (downloads 5yr data, engineers features, trains models, calibrates)
python run_pipeline.py --mode setup

# Static backtest (80/20 train/test split)
python run_pipeline.py --mode backtest

# Expanding-window walk-forward backtest (periodic retrain, most realistic)
python run_pipeline.py --mode walkforward

# Generate this week's strikes (Sunday night)
python run_pipeline.py --mode live

# Retrain models on latest data (monthly)
python run_pipeline.py --mode retrain

# Fetch US/global macro data
python run_pipeline.py --mode macro

# Check model drift & coverage decay
python run_pipeline.py --mode monitor
```

## Features

- **VIX-Aware Strike Placement** — Buffer scales dynamically based on current volatility (high VIX → wider, more conservative strikes)
- **Macro Feature Integration** — US VIX, S&P 500 returns, crude oil, USD/INR, US 10Y yields merged into feature matrix
- **Per-Regime LightGBM Quantile Models** — Separate P10/P90 models for low/mid/high VIX regimes
- **Conformal Prediction (MAPIE)** — Empirical coverage calibration on out-of-sample data
- **Static + Walk-Forward Backtests** — Compare idealized vs. realistic out-of-sample performance
- **Model Drift Monitor** — Automated alerts for coverage decay, feature drift, and win-rate degradation
- **Live Orchestration** — Single command fetches latest data, recomputes features, generates strikes

## Architecture

```
Data Pipeline (M1)
    ↓ (5yr daily OHLCV, intraday, India VIX)
Feature Engineering (M2)
    ↓ (ATR, volatility, Bollinger Bands, range, macro features)
GARCH Volatility (M3)
    ↓ (GJR-GARCH(1,1,1) conditional σ(t) weekly)
LightGBM Regime Models (M4)
    ↓ (per-VIX-regime: low/mid/high, quantile regression P10/P90)
Conformal Calibration (M5)
    ↓ (MAPIE empirical coverage guarantee)
Strike Generation (M6) ←→ Backtester (M7)
    ↓                          ↓
Live Pipeline (M8)         Equity Curve + Metrics
    ↓
Walk-Forward Backtest (M7b)
    ↓
Model Drift Monitor (M13)
```

### New Modules

| Module | Purpose |
|--------|---------|
| `module1b_macro.py` | Fetch US VIX, SPX, crude, USD/INR, US 10Y from Yahoo Finance; build weekly macro features |
| `module7b_walkforward.py` | Expanding-window walk-forward backtest with periodic retrain every 4 weeks |
| `module13_monitor.py` | Coverage decay detection, rolling win-rate alerts, KS feature drift tests |
| `config.py` | Pydantic `BaseSettings` with range validation and convenience properties |

## Configuration

All settings in `.env` (auto-loads via `python-dotenv`):

```env
# Strike buffer (points) — scaled by VIX dynamically
STRIKE_BUFFER_POINTS=100
MIN_BUFFER_POINTS=100

# Wing widths per VIX regime
WING_WIDTH_LOW_VIX=300
WING_WIDTH_MID_VIX=400
WING_WIDTH_HIGH_VIX=500

# Quantile alphas per regime (configurable via .env)
ALPHA_LOW_P10=0.10
ALPHA_LOW_P90=0.90
ALPHA_MID_P10=0.15
ALPHA_MID_P90=0.85
ALPHA_HIGH_P10=0.10
ALPHA_HIGH_P90=0.90

# Walk-forward settings
WF_INITIAL_TRAIN_WEEKS=120
WF_RETRAIN_EVERY_WEEKS=4
WF_CALIBRATION_WEEKS=40
WF_SL_MULTIPLIER=3.0
WF_MAX_VIX_TRADE=22
WF_MIN_PREMIUM_PTS=20

# Target coverage for conformal calibration
TARGET_COVERAGE=0.85

# Monitor thresholds
MONITOR_COVERAGE_DECAY=0.05
MONITOR_DRIFT_WINDOW_WEEKS=12
MONITOR_FEATURE_DRIFT_PVAL=0.01
```

## Usage Examples

### Weekly Iron Condors (Hold to Expiry)

```bash
python run_pipeline.py --mode live
# Output: strikes_live.json with short put, short call, long put, long call
```

### Backtesting

```bash
# Static backtest (fast, 80/20 split)
python run_pipeline.py --mode backtest

# Walk-forward backtest (slower, more realistic — periodic retrain)
python run_pipeline.py --mode walkforward
```

Generates:
- `outputs/backtest_equity_curve.png` / `outputs/walkforward_equity_curve.png`
- `outputs/backtest_results.csv` / `outputs/walkforward_results.csv`
- `outputs/backtest_summary.json` / `outputs/walkforward_summary.json`

Key metrics:
- `win_rate_pct` — % weeks Nifty closed within [short_put, short_call]
- `sharpe_ratio` — risk-adjusted returns (annualized)
- `max_drawdown_points` — peak-to-trough equity decline
- `breach_rate_low_vix_pct` / `mid` / `high` — breach rate by regime

### Model Health Check

```bash
python run_pipeline.py --mode monitor
```

Checks:
- Coverage decay vs. target
- Rolling 8-week win rate
- Feature distribution drift (Kolmogorov-Smirnov)
- POP prediction accuracy

Outputs `outputs/monitor_report_YYYY-MM-DD.json` with `RETRAIN` / `REVIEW_FEATURES` / `OK` recommendation.

## Strike Placement Logic

**Given:** Current spot, P10/P90 log-range predictions, VIX level, GARCH volatility

**Process:**

1. **Regime Assignment:**
   ```
   low  = VIX < low_thresh   (default ~14)
   mid  = low_thresh ≤ VIX < high_thresh (default ~14–18)
   high = VIX ≥ high_thresh
   ```

2. **Quantile Prediction:**
   ```
   log_range_p10 = LightGBM_p10.predict(feature_row)
   log_range_p90 = LightGBM_p90.predict(feature_row)
   ```

3. **Buffer Scaling:**
   ```
   vix_scalar = current_vix / vix_baseline
   effective_buffer = clamp(buffer_pts * vix_scalar, min_buffer_pts, 150)
   ```

4. **Range → Strikes:**
   ```
   range_pts_p10 = spot * (exp(log_range_p10) - 1)
   range_pts_p90 = spot * (exp(log_range_p90) - 1)
   blended_half_range = (0.70 * range_pts_p90 + 0.30 * range_pts_p10) / 2.0

   short_put  = round_to_50(spot - blended_half_range - effective_buffer - put_skew)
   short_call = round_to_50(spot + blended_half_range + effective_buffer + call_skew)
   long_put   = short_put - wing_width
   long_call  = short_call + wing_width
   ```

5. **Breach Probability:**
   ```
   log_range_mu    = (p10 + p90) / 2
   log_range_sigma = (p90 - p10) / (2 * z_0.90)
   breach_prob_call = 1 - Φ((ln(short_call/spot) - mu) / sigma)
   breach_prob_put  = Φ((ln(short_put/spot) - mu) / sigma)
   POP = 1 - breach_call - breach_put
   ```

## Data Files

| Path | Format | Purpose |
|------|--------|---------|
| `data/nifty_daily.parquet` | OHLCV | 5yr daily bars (yfinance) |
| `data/nifty_weekly.parquet` | OHLCV | Weekly aggregation |
| `data/india_vix_daily.parquet` | close | India VIX history |
| `data/nifty_intraday.parquet` | OHLCV | 1h bars (~730 days) |
| `data/macro_daily.parquet` | OHLCV | US/global macro data |
| `data/feature_matrix.parquet` | features | Engineered features per week |
| `data/feature_matrix_with_garch.parquet` | +σ | With GARCH + macro merged |
| `models/lgb_low.pkl` | serialized | LightGBM quantile models (VIX < low_thresh) |
| `models/lgb_mid.pkl` | serialized | LightGBM quantile models (mid regime) |
| `models/lgb_high.pkl` | serialized | LightGBM quantile models (VIX ≥ high_thresh) |
| `models/mapie_calibrated.pkl` | serialized | Global MAPIE conformal calibrator |
| `models/mapie_low.pkl` | serialized | Per-regime MAPIE (fallback) |
| `models/garch_model.pkl` | serialized | Fitted GJR-GARCH model |
| `models/regime_model_meta.json` | metadata | Model sources & training sizes |
| `models/feature_columns.pkl` | list | Feature names (training order) |

## Key Files

- **`run_pipeline.py`** — Master orchestrator (7 modes: setup, backtest, walkforward, live, macro, monitor, retrain)
- **`config.py`** — Pydantic `BaseSettings` with validation
- **`module1_data_pipeline.py`** — Fetch & aggregate OHLCV (yfinance)
- **`module1b_macro.py`** — Fetch US/global macro data
- **`module2_features.py`** — Feature engineering (ATR, volatility, VIX, macro merge)
- **`module3_garch.py`** — GJR-GARCH(1,1,1) conditional volatility
- **`module4_model.py`** — LightGBM per-regime quantile models (P10/P90)
- **`module5_calibration.py`** — MAPIE conformal calibration
- **`module6_strikes.py`** — Strike generation logic
- **`module7_backtest.py`** — Static walk-forward P&L simulation
- **`module7b_walkforward.py`** — Expanding-window walk-forward with periodic retrain
- **`module8_live.py`** — Sunday-night live runner
- **`module13_monitor.py`** — Drift & coverage monitoring

## Installation

```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas`, `numpy`, `pyarrow` — data handling
- `arch`, `scipy`, `statsmodels` — GARCH, statistics
- `lightgbm`, `scikit-learn`, `shap` — ML (quantile regression + explainability)
- `MAPIE` — conformal prediction
- `yfinance` — free market data (no API key)
- `pydantic`, `python-dotenv`, `loguru` — config + logging

## Validation

### Backtest Targets

```bash
python run_pipeline.py --mode backtest
python run_pipeline.py --mode walkforward
```

| Metric | Static Backtest | Walk-Forward |
|--------|----------------|--------------|
| Win rate | ~96% | ~91% |
| Sharpe | ~0.82 | ~0.44 |
| Max drawdown | ~421 pts | ~1,090 pts |
| Expectancy/trade | ~8.7 pts | ~5.5 pts |

> **Note:** Walk-forward is the more realistic benchmark because it simulates live retraining and out-of-sample prediction. The gap between static and walk-forward (~5pp win rate) quantifies overfitting.

### Tuning Guide

```
If win rate < 85% in walk-forward → increase STRIKE_BUFFER_POINTS or widen quantile alphas
If coverage < 80% in monitor → retrain models or increase buffer
If breach_rate_high_vix > 20% → lower WF_MAX_VIX_TRADE or widen wing width
If premium < 20 pts consistently → reduce buffer or tighten wing width
```

## Worst-Case Scenarios

1. **Gap Limit-Down** (>250 pts) → Full loss if exceeds wing width
2. **Earnings Collision** → Gap 300-500 pts on announcement
3. **Geopolitical Shock** → 5-10% move + IV crush
4. **Liquidity Evaporation** → Can't exit emergency position
5. **Model Drift** → Feature distributions shift (caught by monitor)
6. **Weekend Gap** → Can't exit Friday-to-Monday move

**Mitigations:**
- Skip entry on VIX > 22 (`WF_MAX_VIX_TRADE`)
- Skip entry on premium < 20 pts (`WF_MIN_PREMIUM_PTS`)
- Position size: max loss = 1-2% capital per trade
- Run monitor weekly after expiry
- Retrain monthly: `python run_pipeline.py --mode retrain`

## Monitoring & Maintenance

### Weekly (After Expiry)
- Run monitor: `python run_pipeline.py --mode monitor`
- Review `outputs/monitor_report_*.json`
- Check `outputs/backtest_equity_curve.png` for trend degradation

### Monthly (or After Regime Change)
- Retrain models: `python run_pipeline.py --mode retrain`
- Review SHAP importance (`outputs/shap_importance_*.png`) for feature drift
- Check calibration curve (`outputs/calibration_curve.png`)

### After Major Market Events
- Re-fetch macro data: `python run_pipeline.py --mode macro`
- Full pipeline retrain if monitor flags `RETRAIN`

## Disclaimers

⚠️ **Research Project** — Not intended as investment advice. Backtests assume realistic but not perfect execution.

⚠️ **Market Risk** — Options trading has significant downside. Position sizing critical.

⚠️ **Model Risk** — LightGBM quantile models assume continuity; real market tails are fatter. Conformal prediction widens intervals empirically but may still underestimate in unprecedented regimes.

## License

MIT License — Use freely. Attribution appreciated.

## Author

Built for conservative options traders targeting weekly theta decay + VIX-aware risk management.

---

**Questions?** Check `CLAUDE.md` for development notes or module docstrings for API details.

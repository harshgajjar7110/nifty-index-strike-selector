# Nifty 50 Iron Condor Strategy Engine

ML-powered system for **conservative strike placement** in weekly/short-duration iron condor options trading on Nifty 50 (Indian stock index). Uses LightGBM quantile models + GARCH volatility + conformal prediction for **calibrated probability guarantees**.

## Quick Start

```bash
# Setup (download 5yr data, train models, calibrate)
python run_pipeline.py --mode setup

# Backtest historical performance
python run_pipeline.py --mode backtest

# Generate this week's strikes (Sunday night)
python run_pipeline.py --mode live

# Retrain models on latest data
python run_pipeline.py --mode retrain
```

## Features

✅ **VIX-Aware Strike Placement** — Buffer scales dynamically based on current volatility (high VIX → wider, more conservative strikes)

✅ **Asymmetric Put Skew** — Put side widened to reflect NSE market premium for downside protection

✅ **Breach Probability Modeling** — Uses GARCH conditional volatility + lognormal CDF to compute actual breach risk (not just distance)

✅ **Conformal Prediction** — Guaranteed ≥85% coverage on prediction intervals (MAPIE calibration)

✅ **Walk-Forward Backtest** — Historical P&L simulation with equity curve, Sharpe ratio, max drawdown

✅ **Live Orchestration** — Single command updates all data, retrains GARCH, generates strikes every Sunday

## Architecture

```
Data Pipeline (M1)
    ↓ (5yr daily OHLCV, intraday, VIX)
Feature Engineering (M2)
    ↓ (ATR, volatility, Bollinger Bands, range)
GARCH Volatility (M3)
    ↓ (conditional σ(t) weekly)
LightGBM P10/P90 Models (M4)
    ↓ (quantile regressors, 80/20 train/test split)
Conformal Calibration (M5)
    ↓ (MAPIE empirical coverage guarantee)
Strike Generation (M6) ←→ Backtester (M7)
    ↓                          ↓
Live Pipeline (M8)         Equity Curve + Metrics

```

## Configuration

All settings in `.env` (auto-loads or inherits sensible defaults):

```env
# Strike buffer (points) — scaled by VIX dynamically
STRIKE_BUFFER_POINTS=50

# Iron condor wing width (short to long strike)
WING_WIDTH_POINTS=200

# Baseline VIX for scaling (if not set, computed from 52-week historical mean)
# VIX_BASELINE=16.0

# Put skew: extra OTM points for short put (0 = symmetric)
# Calibrate from backtest results; typical range 0-150 pts
PUT_SKEW_POINTS=0

# Base premium collected per trade (scaled per-row by VIX in backtest)
PREMIUM_POINTS_BASE=80

# Conformal prediction coverage target
TARGET_COVERAGE=0.85
```

## Usage Examples

### Weekly Iron Condors (Hold to Expiry)

```bash
python run_pipeline.py --mode live
# Output: strikes_live.json with short put, short call, long put, long call
# POP: 65-72% depending on VIX
```

### 2-3 Day Put Spreads (Trending Down Days)

On high-VIX, trending-down days:
1. Widen wing to 250-300 pts (in `.env` or code)
2. Set `PUT_SKEW_POINTS=50-100` (higher skew on trending days)
3. Implement 60% profit exit (not hold-to-expiry)
4. Skip entry on earnings/announcement days

**Expected metrics:**
- Win rate: 55-65% (vs 70-75% for weekly)
- POP: 60-68% depending on VIX and spread width
- Theta: ~60-70% captured in 2-3 days

### Backtesting & Calibration

```bash
python run_pipeline.py --mode backtest
```

Generates:
- `outputs/backtest_equity_curve.png` — cumulative P&L + POP overlay
- `outputs/backtest_results.csv` — per-week details
- `outputs/backtest_summary.json` — metrics:
  - `win_rate_pct` — % weeks both strikes held
  - `sharpe_ratio` — risk-adjusted returns (annualized)
  - `max_drawdown_points` — peak-to-trough loss
  - `avg_pop_pct` — average Prob of Profit
  - `breach_rate_up_pct`, `breach_rate_down_pct` — asymmetry
  - `recommended_put_skew_pts` — calibrated from breach imbalance

Use calibration to tune `.env`:
```
If breach_rate_down > breach_rate_up by 5% → set PUT_SKEW_POINTS = 125
If win rate < 65% → increase STRIKE_BUFFER_POINTS or WING_WIDTH_POINTS
```

## Strike Placement Logic

**Given:** Current spot, P10/P90 predictions, VIX level, GARCH volatility

**Process:**

1. **Buffer Scaling:**
   ```
   vix_scalar = current_vix / vix_baseline
   effective_buffer = clamp(buffer_pts * vix_scalar, [30, 150])
   ```

2. **Range Calculation:**
   ```
   half_range_p90 = spot * (exp(log_range_p90) - 1) / 2
   ```

3. **Asymmetric Placement:**
   ```
   short_put = round_to_50(spot - half_range_p90 - effective_buffer - put_skew_pts)
   short_call = round_to_50(spot + half_range_p90 + effective_buffer)  [no skew]
   ```

4. **Breach Probability (if GARCH vol available):**
   ```
   z_call = ln(short_call / spot) / garch_vol_weekly
   z_put = ln(spot / short_put) / garch_vol_weekly
   breach_prob_call = 1 - Φ(z_call)
   breach_prob_put = Φ(-z_put)
   prob_of_profit = 1 - breach_prob_call - breach_prob_put
   ```

**Output:** JSON with strikes, predicted range, VIX-aware metrics, breach probabilities

## Data Files

| Path | Format | Purpose |
|------|--------|---------|
| `data/nifty_daily.parquet` | OHLCV | 5yr daily bars (yfinance) |
| `data/nifty_weekly.parquet` | OHLCV | Weekly aggregation |
| `data/india_vix_daily.parquet` | close | India VIX history |
| `data/nifty_intraday.parquet` | OHLCV | 1h bars (~730 days) |
| `data/feature_matrix.parquet` | features | Engineered features per week |
| `data/feature_matrix_with_garch.parquet` | +σ | With GARCH conditional volatility |
| `models/lgbm_p10.pkl` | serialized | P10 quantile model |
| `models/lgbm_p90.pkl` | serialized | P90 quantile model |
| `models/feature_columns.pkl` | list | Feature names (training) |
| `models/mapie_calibrated.pkl` | serialized | Conformal calibrator (MAPIE) |

## Key Files

- **`run_pipeline.py`** — Master orchestrator (4 modes: setup, backtest, live, retrain)
- **`module1_data_pipeline.py`** — Fetch & aggregate OHLCV (yfinance)
- **`module2_features.py`** — Feature engineering (ATR, volatility, VIX, calendar)
- **`module3_garch.py`** — GARCH(1,1) conditional volatility
- **`module4_model.py`** — LightGBM P10/P90 quantile training
- **`module5_calibration.py`** — MAPIE conformal calibration
- **`module6_strikes.py`** — Strike generation logic (core)
- **`module7_backtest.py`** — Walk-forward P&L simulation
- **`module8_live.py`** — Sunday-night live runner (entry point)

## Installation

```bash
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas`, `numpy`, `pyarrow` — data handling
- `arch`, `scipy`, `statsmodels` — GARCH, statistics
- `lightgbm`, `scikit-learn`, `shap` — ML + explainability
- `MAPIE` — conformal prediction
- `yfinance` — free market data (no API key)
- `python-dotenv`, `loguru` — config + logging

## Validation

### Test on Mock VIX Scenarios

Create extreme-VIX test to verify scaling:

```python
from module6_strikes import generate_strikes

for vix in [5, 10, 16, 25, 40]:
    strikes = generate_strikes(
        current_close=24000,
        log_range_p10=0.0230,
        log_range_p90=0.0600,
        vix_level=vix,
        garch_vol_weekly=0.035
    )
    print(f"VIX={vix}: PUT={strikes['short_put']}, CALL={strikes['short_call']}, "
          f"Buffer={strikes['effective_buffer_pts']:.0f}, POP={strikes['prob_of_profit']*100:.1f}%")
```

**Expected behavior:**
- ✅ Buffer scales 30→140+ pts
- ✅ Strikes widen OTM as VIX rises
- ✅ POP improves (higher) with conservative placement
- ✅ Width increases with VIX

### Backtest Validation

```bash
python run_pipeline.py --mode backtest
cat outputs/backtest_summary.json | grep -E "win_rate|sharpe|breach"
```

Target metrics:
- Win rate: 70-75% (weekly) or 55-65% (2-3 day)
- Sharpe: 1.5+ (annual)
- Max drawdown: <500 pts
- Coverage: 85%+

## Worst-Case Scenarios

1. **Gap Limit-Down** (>250 pts) → Full loss if exceeds wing width
2. **Earnings Collision** → Gap 300-500 pts on announcement
3. **Geopolitical Shock** (war escalation) → 5-10% move + IV crush = premium decay halts
4. **Liquidity Evaporation** → Can't exit emergency position
5. **GARCH Model Failure** → Volatility regime shift (distribution tail fatter than expected)
6. **Weekend Gap** → Can't exit Friday-to-Monday move

**Mitigations:**
- Skip entry on earnings weeks
- Monitor economic calendar for events
- Position size: max loss = 1-2% capital per trade
- Add hard stop: exit if intraday move >3σ
- Check bid-ask spread before entry (skip if >20 pts)

## Strategy Notes

### Weekly Iron Condors
- Holds 5-7 days until Thursday expiry
- Win rate: 70-75%
- Slower theta decay but more time for error recovery

### 2-3 Day Put Spreads (Trending Days)
- Directional (puts only), short premium
- Win rate: 55-65% (vs 70-75% weekly)
- Faster capital rotation, higher gap risk
- Best on trending-down days with high VIX (volatility amplifies put premium)

**Tuning for 2-3 Day Spreads:**
- Increase `WING_WIDTH_POINTS` to 250-300 pts on gap-risk days
- Set `PUT_SKEW_POINTS = 50-100` (market pays premium for downside protection during high IV)
- Implement 60% profit exit instead of hold-to-expiry
- Exit immediately if intraday VIX spike (>25% move) or 3σ price move

## Monitoring & Maintenance

### Daily Checks (if deployed)
- Monitor option bid-ask spreads (NSE liquidity)
- Check for surprise earnings/events
- Track realized vs. predicted range

### Weekly (After Expiry)
- Review P&L vs. backtest expectations
- Check equity curve trend
- Compare realized vol to GARCH forecast

### Monthly (or After Market Regime Change)
- Retrain models: `python run_pipeline.py --mode retrain`
- Recalibrate skew from `recommended_put_skew_pts`
- Audit GARCH fit quality
- Review SHAP importance (feature drift?)

## Disclaimers

⚠️ **Research Project** — Not intended as investment advice. Backtests assume:
- Perfect execution (no slippage, instant fills)
- No transaction costs
- No circuit breakers or exchange halts
- Historical patterns continue (overfitting risk)

⚠️ **Market Risk** — Options trading has unlimited downside (short calls) or near-total loss (short puts + wide gaps). Position sizing critical.

⚠️ **Model Risk** — GARCH assumes normal distribution; tails fatter in real markets. Breach probability underestimates tail risk.

## License

MIT License — Use freely. Attribution appreciated.

## Author

Built for conservative options traders targeting 2-3 day theta decay + VIX-aware risk management.

---

**Questions?** Check CLAUDE.md for development notes or module docstrings for API details.

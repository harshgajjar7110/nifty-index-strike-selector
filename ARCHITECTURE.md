# Architecture & Design

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    NIFTY 50 IRON CONDOR ENGINE                  │
│                                                                   │
│  Input: Market data (OHLCV, VIX) → Output: Strike strikes       │
└─────────────────────────────────────────────────────────────────┘

                            ┌─────────────────┐
                            │   Data Layer    │
                            │                 │
                            │ ├─ Daily OHLCV  │
                            │ ├─ VIX history  │
                            │ ├─ Intraday 1h  │
                            │ └─ Weekly agg.  │
                            └────────┬────────┘
                                     │ (parquet)
                    ┌────────────────▼──────────────────┐
                    │   Feature Engineering (M2)       │
                    │                                   │
                    │ ├─ ATR (volatility)              │
                    │ ├─ Bollinger Bands               │
                    │ ├─ Realized vol                  │
                    │ ├─ VIX level                     │
                    │ ├─ Range (H-L)                   │
                    │ └─ Calendar features             │
                    └────────────┬───────────────────────┘
                                 │ (257 weekly rows)
                    ┌────────────▼──────────────────┐
                    │   GARCH Volatility (M3)       │
                    │                                │
                    │  GARCH(1,1) on daily returns  │
                    │  → σ(t) weekly conditional    │
                    │  → Used for breach prob       │
                    └────────────┬──────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            │                                         │
   ┌────────▼──────────┐                 ┌─────────▼────────┐
   │ Training (M4)     │                 │ Backtest (M7)    │
   │                   │                 │                  │
   │ ├─ Split 80/20    │                 │ ├─ Walk-forward  │
   │ ├─ P10 model      │                 │ ├─ P&L sim      │
   │ ├─ P90 model      │                 │ ├─ Metrics      │
   │ └─ LightGBM       │                 │ ├─ Equity curve │
   │    (quantile)     │                 │ └─ POP overlay  │
   └────────┬──────────┘                 └─────────────────┘
            │
   ┌────────▼──────────────────┐
   │ Calibration (M5)          │
   │                            │
   │ MAPIE conformal pred      │
   │ → ≥85% coverage guarantee │
   └────────┬──────────────────┘
            │
            └──────────┬───────────────────────┐
                       │ (trained models)      │
            ┌──────────▼──────────┐            │
            │  Strike Gen (M6)    │◄───────────┘
            │                     │
            │ ├─ Fetch live VIX   │
            │ ├─ Predict range    │
            │ ├─ Scale buffer     │
            │ ├─ Place strikes    │
            │ ├─ Breach prob      │
            │ └─ JSON output      │
            └────────┬────────────┘
                     │
                  ┌──▼──────────────────┐
                  │  Live Output (M8)   │
                  │                     │
                  │ ├─ Console display  │
                  │ ├─ JSON file        │
                  │ └─ Ready to trade   │
                  └────────────────────┘
```

## Data Flow (Detailed)

### Setup Mode: M1 → M5

```
M1: fetch_nifty_daily()
    └─→ yfinance ^NSEI
        Saves: data/nifty_daily.parquet (5 years)

M1: fetch_nifty_intraday()
    └─→ yfinance ^NSEI 1h bars (2 years)
        Saves: data/nifty_intraday.parquet

M1: fetch_india_vix()
    └─→ yfinance ^INDIAVIX
        Saves: data/india_vix_daily.parquet

M1: build_nifty_weekly()
    └─→ Aggregate daily → weekly OHLC
        Saves: data/nifty_weekly.parquet

M2: build_features()
    ├─ Read: nifty_weekly.parquet, india_vix_daily.parquet
    ├─ Compute: ATR, realized_vol, vix_level, log_range, etc.
    └─ Saves: data/feature_matrix.parquet (257 rows)

M3: run_garch_pipeline()
    ├─ Read: nifty_daily.parquet
    ├─ Fit GARCH(1,1) on daily returns
    ├─ Extract σ(t) weekly
    └─ Saves: data/feature_matrix_with_garch.parquet

M4: train_models()
    ├─ Read: feature_matrix_with_garch.parquet
    ├─ Split: 80% train (rows 0-205), 20% test (rows 206-256)
    ├─ Train P10 & P90 LightGBM quantile models
    └─ Saves: models/lgbm_p10.pkl, models/lgbm_p90.pkl,
               models/feature_columns.pkl

M5: run_calibration()
    ├─ Read: feature_matrix_with_garch.parquet (test set only)
    ├─ Apply MAPIE conformal wrapper
    ├─ Calibrate for ≥85% coverage
    └─ Saves: models/mapie_calibrated.pkl, calibration_report.json
```

### Backtest Mode: M7

```
M7: run_backtest()
    ├─ Read: feature_matrix_with_garch.parquet
    ├─ For each week in test set (rows 206-256):
    │  ├─ Predict P10/P90 via M4 models
    │  ├─ Extract VIX, GARCH vol from feature row
    │  ├─ Call M6: generate_strikes() with VIX scaling
    │  ├─ Compute breach probabilities
    │  ├─ Simulate P&L (if breached → loss, else → premium)
    │  └─ Store: pnl_points, won, breach_up, breach_down
    │
    ├─ Aggregate metrics:
    │  ├─ Win rate, total P&L, Sharpe, max drawdown
    │  ├─ Breach rate by VIX regime (low/mid/high)
    │  ├─ POP statistics
    │  └─ Recommend put skew from breach asymmetry
    │
    └─ Output:
       ├─ outputs/backtest_results.csv (per-week details)
       ├─ outputs/backtest_equity_curve.png (P&L + POP overlay)
       └─ outputs/backtest_summary.json (summary metrics)
    ```

    ### Module 9: Credit Spread Generator (`module9_spreads.py`)

    **Public API:**

    ```python
    def get_nse_expiries(today: date) -> list[dict]
    Input: Today's date
    Output: List of 3 expiry dicts {date, dte, type}

    def estimate_bs_price(S, K, T, sigma, r, type) -> float
    Input: Spot, Strike, Time (yrs), Vol (ann), Rate, Type
    Output: Theoretical option price

    def detect_direction(feature_row: pd.Series) -> dict
    Input: Feature row
    Output: {direction, confidence, composite, signals}

    def generate_all_spreads(feature_row, spot, vix, garch_vol) -> dict
    Input: Feature row and market data
    Output: JSON-serializable dict with multi-expiry candidates
    ```

    ## Configuration Management


```
M8: run_live_pipeline()
    ├─ M1: Incremental data fetch (daily, VIX, intraday)
    ├─ M2: Rebuild features (last row = this week)
    ├─ M3: Update GARCH (fast: only last row)
    │
    ├─ Load: feature_row = feature_matrix_with_garch.iloc[-1]
    ├─ Extract: vix_level, garch_sigma_mean
    │
    ├─ M6: Predict range (P10, P90) via M4 models
    ├─ M6: Fetch live Nifty spot (yfinance ^NSEI)
    ├─ M6: generate_strikes(spot, P10, P90, vix_level, garch_vol)
    │       ├─ Load VIX baseline from .env or parquet
    │       ├─ Scale buffer: effective = buffer * (vix / baseline), clamp [30, 150]
    │       ├─ Place puts/calls with asymmetric skew
    │       ├─ Compute breach probs if garch_vol > 0
    │       └─ Return strikes dict
    │
    └─ Output:
       ├─ Console: formatted table (spot, strikes, VIX, buffer, POP)
       └─ JSON: outputs/strikes_live.json (full details)
```

### Credit Spread Generator (M9)

```
┌─────────────────────────────────────────────────────────────┐
│ Module 9: Credit Spread Generation System                   │
│                                                             │
│ ├─ NSE Expiry Calc (Weekly/Monthly)                         │
│ ├─ Black-Scholes Premium Estimation                         │
│ ├─ Direction Detection (Momentum + VIX Trend)               │
│ ├─ DTE-Aware Buffer & Wing Scaling                          │
│ └─ EV-based Spread Ranking                                  │
└─────────────────────────────────────────────────────────────┘
```

**M9: generate_all_spreads()**
- **Input**: Feature row, spot, VIX, GARCH vol.
- **Process**:
  1. Calculate upcoming 3 expiries (NSE Thursdays).
  2. Detect direction signal from `prev_week_gap` and `vix_change_1w`.
  3. Predict price range for each expiry (DTE scaled).
  4. Estimate theoretical premiums via Black-Scholes.
  5. Rank spreads by `EV = R:R * POP`.
- **Output**: `outputs/spreads_live.json` with multi-expiry candidates.

## Module Interfaces

### Module 6: Strike Generation (`module6_strikes.py`)

**Public API:**

```python
def predict_range(feature_row: pd.Series) -> dict
    Input: Feature row with feature_columns
    Output: {log_range_p10, log_range_p90}

def generate_strikes(
    current_close: float,
    log_range_p10: float,
    log_range_p90: float,
    vix_level: float | None = None,
    put_skew_pts: int = 0,
    garch_vol_weekly: float | None = None,
) -> dict
    Input: Spot price, predicted ranges, optional VIX/GARCH
    Output: {
        current_close, short_put, short_call, long_put, long_call,
        predicted_range_p10, predicted_range_p90,
        buffer_pts, wing_width_pts,
        vix_level, vix_baseline, vix_scalar, effective_buffer_pts,
        put_skew_pts,
        breach_prob_call, breach_prob_put, prob_of_profit
    }

def fetch_live_spot() -> float
    Input: None (fetches live yfinance)
    Output: Current Nifty 50 spot price

def run_live_prediction(feature_row: pd.Series) -> dict
    Input: Feature row
    Output: Strikes dict + saves JSON file
```

**Key Implementation:**

- VIX scaling: `effective_buffer = clamp(buffer_pts * (vix / baseline), [30, 150])`
- Asymmetric placement: Put side gets `-put_skew_pts`, call side unchanged
- Breach prob: Lognormal CDF with GARCH vol, returns None if garch_vol ≤ 0
- Rounding: To nearest 50 pts (Nifty strike interval)

### Module 7: Backtest (`module7_backtest.py`)

**Public API:**

```python
def run_backtest() -> dict
    Input: None (reads feature_matrix_with_garch.parquet)
    Output: Summary dict with all metrics
            + CSV/PNG/JSON files to outputs/
```

**Key Metrics Computed:**

- Win rate: % weeks where price stayed within [short_put, short_call]
- Sharpe: annualized return / volatility = mean_pnl / std_pnl * √52
- Max drawdown: peak-to-trough in cumulative P&L
- Breach rates by VIX regime: separate wins/losses for low/mid/high VIX
- POP (Prob of Profit): average breach_prob_call + breach_prob_put across weeks
- Recommended put skew: calibrated from breach asymmetry (down - up) * 25

## Configuration Management

Single source of truth: `.env` file

```
STRIKE_BUFFER_POINTS         → module6_strikes._load_config()
WING_WIDTH_POINTS           → module6_strikes._load_config()
VIX_BASELINE                → auto-computed if not in .env (52-week mean)
PUT_SKEW_POINTS            → module6_strikes._load_config()
PREMIUM_POINTS_BASE        → module7_backtest (for per-row scaling)
TARGET_COVERAGE            → module5_calibration
```

All modules that need config import `_load_config()` from module6:

```python
from module6_strikes import _load_config
buffer, wing, vix_baseline, put_skew = _load_config()
```

## Model Training (Module 4)

**LightGBM Quantile Regression:**

- Two separate models: P10 (0.1 quantile) and P90 (0.9 quantile)
- Input features: 14 engineered features (ATR, VIX, range, etc.)
- Target: log(H/L) → log_range_p10 and log_range_p90
- Train/test split: 80/20 chronological (no lookahead bias)
- Hyperparameters: depth=5, leaves=31, learning_rate=0.01, iterations=1000

**Key Design Decisions:**

- Quantile regression (not mean) → captures range extremes, not average
- P10/P90 (not symmetric bounds) → allows asymmetric distributions
- Two separate models (not one) → independent optimization per quantile
- No feature scaling → LightGBM tree-based, invariant to scale

## Volatility Modeling (Module 3)

**GARCH(1,1) on Daily Returns:**

```
r_t = log(close_t / close_t-1)

σ²_t = ω + α*r²_t-1 + β*σ²_t-1

where ω, α, β are fitted parameters
```

**Output:** Weekly conditional σ (average of daily σ over the week)

**Used For:** Breach probability calculation via lognormal CDF

```
P(high > strike) = 1 - Φ( ln(strike/spot) / σ_weekly )
P(low < strike)  = Φ( ln(spot/strike) / σ_weekly )
```

## Conformal Prediction (Module 5)

**MAPIE (Multivariate Adaptive Prediction Intervals Estimator):**

- Wraps M4 models
- Returns prediction intervals with empirical coverage guarantees
- Calibrated on test set (20% of data)
- Targets ≥85% coverage

**Output:** Intervals guaranteed to contain true log_range with ≥85% probability

---

## Time Complexity

| Operation | Time | Notes |
|-----------|------|-------|
| M1 (fetch) | 5-10 min | Network I/O limited |
| M2 (features) | <1 sec | 257 rows, vectorized |
| M3 (GARCH) | 30 sec | Daily data fitting |
| M4 (train) | 5-10 sec | LightGBM, 2K rows |
| M5 (calibrate) | <1 sec | Empirical quantile |
| M6 (strike gen) | <100ms | Per-row, vectorizable |
| M7 (backtest) | 2-5 min | 51 weeks, breach sim |
| M8 (live full) | 10-15 min | Runs M1-M3 + M6 |

## Storage

| File | Size | Type |
|------|------|------|
| nifty_daily.parquet | ~5 MB | Compressed OHLCV |
| nifty_weekly.parquet | <1 MB | 257 weeks |
| india_vix_daily.parquet | ~2 MB | VIX history |
| feature_matrix_with_garch.parquet | <2 MB | 257 × 20 features |
| lgbm_p10.pkl | ~100 KB | Serialized model |
| lgbm_p90.pkl | ~100 KB | Serialized model |
| mapie_calibrated.pkl | <200 KB | Conformal predictor |

**Total footprint:** <20 MB (no external APIs)

---

**Questions?** See README for usage or module docstrings for implementation details.

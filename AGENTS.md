# AGENTS.md

Universal dev guidance for AI coding assistants (Claude Code, Gemini CLI, Cursor, Copilot, Kilo, etc.) working in this repo. Consolidates and supersedes `CLAUDE.md` and `GEMINI.md`.

---

## 1. Project Overview

**Nifty 50 Iron Condor Strategy Engine** — ML system that predicts the weekly price range for the Nifty 50 (Indian stock index) and generates **conservative** iron condor strike recommendations using per-VIX-regime LightGBM quantile models, GJR-GARCH volatility, macro features, and MAPIE conformal calibration.

**Output:** Weekly strike levels (short put, short call, long put, long call) with calibrated probability coverage, breach probability, and POP.

**Business domain:** Quantitative Finance / Options Trading (NSE weekly options).

---

## 2. Quick Commands

```bash
# First-time setup (downloads ~5yr data, engineers features, fits GARCH,
# trains per-regime LightGBM P10/P90, applies MAPIE calibration) — ~10–30 min
python run_pipeline.py --mode setup

# Static backtest (80/20 chronological split)
python run_pipeline.py --mode backtest

# Walk-forward backtest (expanding window, periodic retrain — most realistic)
python run_pipeline.py --mode walkforward

# Generate this week's strikes (Sunday night)
python run_pipeline.py --mode live

# Fetch US/global macro data
python run_pipeline.py --mode macro

# Check model drift, coverage decay, feature drift
python run_pipeline.py --mode monitor

# Retrain models on latest data (monthly)
python run_pipeline.py --mode retrain
```

---

## 3. Architecture (M1 → M13)

```
┌─────────────────────────────────────────────────────────────┐
│ M1  Data Pipeline        Yahoo Finance / Kite → parquets    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ M1b Macro Pipeline       US VIX, SPX, crude, USD/INR, UST  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ M2  Feature Engineering ATR, vol, VIX, BBands, calendar,   │
│                          macro merge                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ M3  GJR-GARCH(1,1,1)     Conditional σ(t) weekly            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ M4  LightGBM Quantile    Per-regime (low/mid/high VIX)      │
│     P10 & P90             quantile regressors                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ M5  MAPIE Calibration    ≥85% empirical coverage guarantee  │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────────┐          ┌──────────────────────┐
│ M6  Strike Generator │          │ M7  Static Backtest  │
│     spot+quantiles → │          │     80/20 P&L curve  │
│     IC strikes       │          └──────────────────────┘
└──────────────────────┘          ┌──────────────────────┐
        ↑                         │ M7b Walk-Forward     │
        │                         │     expanding window │
        │                         │     + periodic retrain│
        │                         └──────────────────────┘
        │                                       ↓
┌──────────────────────┐          ┌──────────────────────┐
│ M8  Live Runner       │          │ M13 Monitor          │
│     Sunday-night     │          │     drift/coverage   │
│     orchestration    │          │     decay detection  │
└──────────────────────┘          └──────────────────────┘
```

### Data Flow Contract

| Module | Reads From                                  | Writes To                                                              | Purpose                              |
|--------|---------------------------------------------|------------------------------------------------------------------------|--------------------------------------|
| M1     | Yahoo Finance / Kite                        | `data/nifty_daily.parquet`, `nifty_intraday.parquet`, `nifty_weekly.parquet`, `india_vix_daily.parquet` | Fetch & store OHLCV history          |
| M1b    | Yahoo Finance                               | `data/macro_daily.parquet`                                             | US/global macro features             |
| M2     | M1 + M1b outputs                            | `data/feature_matrix.parquet`                                          | Engineered features                  |
| M3     | M1 (daily) + M2                             | `data/feature_matrix_with_garch.parquet`, `models/garch_model.pkl`     | GARCH conditional σ                  |
| M4     | M3 output                                   | `models/lgb_low.pkl`, `lgb_mid.pkl`, `lgb_high.pkl`, `feature_columns.pkl`, `outputs/shap_importance_*.png` | Per-regime quantile models + SHAP    |
| M5     | M4 models + M3 features                     | `models/mapie_calibrated.pkl` (+ `mapie_low.pkl` fallback), `outputs/calibration_report.json`, `calibration_curve.png` | Conformal coverage calibration       |
| M6     | M4/M5 models, live spot + VIX               | `outputs/strikes_YYYY-MM-DD.json`                                      | Iron condor strikes                  |
| M7     | M3/M4/M6                                   | `outputs/backtest_results.csv`, `backtest_equity_curve.png`, `backtest_summary.json` | Static backtest P&L                  |
| M7b    | M3/M4/M6, retrain loop                      | `outputs/walkforward_*.csv/png/json`                                   | Walk-forward backtest                |
| M8     | M1, M2, M3, M5, M6 (incremental)            | `outputs/strikes_live.json`, console log                               | Sunday-night live strikes            |
| M13    | M5 calibration + backtest outputs          | `outputs/monitor_report_YYYY-MM-DD.json`                               | Drift / coverage decay detection     |

---

## 4. Tech Stack

- **Language:** Python 3.10+
- **Data:** `pandas`, `numpy`, `pyarrow` (Parquet), `yfinance`
- **Vol/Stats:** `arch` (GJR-GARCH), `scipy`, `statsmodels`
- **ML:** `lightgbm` (quantile regression), `scikit-learn`, `shap`
- **Calibration:** `MAPIE` (conformal prediction)
- **Infra:** `pydantic` (`BaseSettings` + range validation), `python-dotenv`, `loguru` (color-coded structured logs), `joblib`

---

## 5. Configuration (`.env`)

> **Source of truth:** the live `.env` file (read by `config.py` via Pydantic `BaseSettings`). The values shown below are the live runtime values. `config.py` holds safer range-validated defaults that `.env` overrides; `.env.example` mirrors `.env` for documentation. README.md and AGENTS.md numeric defaults may lag — always check `.env` and `config.py` for what the pipeline actually uses.

```env
# Strike buffer (points) — scaled by VIX dynamically
STRIKE_BUFFER_POINTS=100
MIN_BUFFER_POINTS=100

# Wing widths per VIX regime
WING_WIDTH_LOW_VIX=300
WING_WIDTH_MID_VIX=400
WING_WIDTH_HIGH_VIX=500

# Quantile alphas per regime (configurable)
ALPHA_LOW_P10=0.10
ALPHA_LOW_P90=0.90
ALPHA_MID_P10=0.15
ALPHA_MID_P90=0.85
ALPHA_HIGH_P10=0.10
ALPHA_HIGH_P90=0.90

# Walk-forward
WF_INITIAL_TRAIN_WEEKS=120
WF_RETRAIN_EVERY_WEEKS=4
WF_CALIBRATION_WEEKS=40
WF_SL_MULTIPLIER=3.0
WF_MAX_VIX_TRADE=22
WF_MIN_PREMIUM_PTS=20

# Conformal target
TARGET_COVERAGE=0.85

# Monitor thresholds
MONITOR_COVERAGE_DECAY=0.05
MONITOR_DRIFT_WINDOW_WEEKS=12
MONITOR_FEATURE_DRIFT_PVAL=0.01

# Optional Kite (falls back to yfinance if unset)
# KITE_API_KEY=...
# KITE_API_SECRET=...
# KITE_ACCESS_TOKEN=...
```

Config is loaded centrally via `config.py` using Pydantic `BaseSettings` with range validation.

---

## 6. Critical Design Decisions

### No Lookahead Bias
- M4: 80% train (oldest) / 20% test (newest).
- M5: calibration on test set only.
- M7 / M7b: walk-forward with expanding window — never train on future data.
- **Never** use `nifty_weekly.parquet` labels in `module2_features.py` training features.

### Per-Regime Models
- Train separate P10/P90 LightGBM models for `low` (VIX<14), `mid` (14–18), `high` (≥18) VIX regimes.
- Different quantile alphas per regime (see `.env`).

### Conformal Prediction (MAPIE)
- Wraps quantile predictions with empirical coverage guarantees.
- If coverage < target, intervals widen automatically.
- See `outputs/calibration_report.json` for coverage @ 80/85/90%.

### Strike Rounding
- Nifty rounds to nearest 50 points (22050, 22100, 22150).
- Use `round_to_strike(price, interval=50)` (a.k.a. `round_to_50`).

### Strike Placement (Mathematical Contract — `module6_strikes.py`)
Any change to strike logic **must** preserve this:

1. **Regime Assignment** from India VIX.
2. **Quantile Prediction** → P10/P90 log-ranges.
3. **Buffer Scaling:**
   ```
   vix_scalar        = current_vix / vix_baseline
   effective_buffer  = clamp(base_buffer * vix_scalar, min_buffer, 150)
   ```
4. **Range → Strikes:**
   ```
   blended_range       = (0.70 * P90 + 0.30 * P10) / 2
   short_put  = round_to_50(spot - blended_range - effective_buffer - put_skew)
   short_call = round_to_50(spot + blended_range + effective_buffer + call_skew)
   long_put   = short_put  - wing_width
   long_call  = short_call + wing_width
   ```
5. **Breach Probability (Gaussian on log-range):**
   ```
   mu    = (p10 + p90) / 2
   sigma = (p90 - p10) / (2 * z_0.90)
   breach_prob_call = 1 - Φ((ln(short_call/spot) - mu) / sigma)
   breach_prob_put  =     Φ((ln(short_put/spot)  - mu) / sigma)
   POP             = 1 - breach_call - breach_put
   ```

### Safety / Risk Mitigations
- **Skip trade** if VIX > `WF_MAX_VIX_TRADE` (22) **or** expected premium < `WF_MIN_PREMIUM_PTS` (20).
- **Position sizing:** max loss = 1–2% of capital per weekly trade.
- **Worst cases** (gap-down, earnings, geopolitics, liquidity gap, weekend) — see README §"Worst-Case Scenarios".

### Validation Targets (Walk-Forward — the realistic benchmark)

| Metric                    | Target     |
|---------------------------|------------|
| Win rate                  | ≥ 85% (~91% in WF) |
| Sharpe (annualized)       | > 0.40     |
| Coverage (MAPIE)          | ≥ 85%      |
| Breach rate (high-VIX)    | ≤ 20%      |

### Tuning Heuristics
```
win rate < 85% in walk-forward  → increase STRIKE_BUFFER_POINTS or widen quantile alphas
coverage < 80% in monitor      → retrain models or increase buffer
breach_rate_high_vix > 20%     → lower WF_MAX_VIX_TRADE or widen wing width
premium < 20 pts consistently  → reduce buffer or tighten wing width
```

---

## 7. Coding Conventions

- **Formatting:** PEP 8. `loguru` for structured color-coded logging.
- **Naming:**
  - Modules: `moduleN_description.py` (e.g. `module6_strikes.py`).
  - Functions / variables: `snake_case`. Classes: `PascalCase`.
- **Error handling:** `try-except` with `loguru.logger.error(...)` and `traceback` in orchestrators.
- **Config:** load from `.env` via `config.py` (Pydantic `BaseSettings`). Never hardcode tunables.
- **No comments unless asked** (code is self-documenting; docstrings on public functions are fine).
- **No emojis in code, logs, or files.**
- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`, `test:`).

---

## 8. Repository Layout

```
research/
├── run_pipeline.py            # Master orchestrator (7 modes)
├── config.py                  # Pydantic settings + validation
├── module1_data_pipeline.py   # M1: fetch & aggregate OHLCV
├── module1b_macro.py          # M1b: US/global macro data
├── module2_features.py        # M2: feature engineering + macro merge
├── module3_garch.py           # M3: GJR-GARCH(1,1,1)
├── module4_model.py           # M4: per-regime LightGBM P10/P90 + SHAP
├── module5_calibration.py     # M5: MAPIE conformal calibration
├── module6_strikes.py         # M6: strike generation (math contract here)
├── module7_backtest.py        # M7: static walk-forward backtest
├── module7b_walkforward.py    # M7b: expanding-window walk-forward
├── module8_live.py            # M8: Sunday-night live runner
├── module13_monitor.py        # M13: drift & coverage monitor
├── requirements.txt
├── .env / .env.example        # gitignore `.env`
├── data/                      # Parquet files
├── models/                    # .pkl models + metadata
└── outputs/                   # charts, backtest CSVs, strike JSONs
```

---

## 9. Development Workflow

### Setup
```bash
python -m venv venv
source venv/Scripts/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --mode setup
```

### Adding a New Feature
1. Compute in `module2_features.py`, persist to `feature_matrix.parquet`.
2. Retrain M4 (models pick up new signal).
3. Re-run `--mode backtest` to validate static; `--mode walkforward` for realistic.
4. Re-calibrate M5 if coverage drops.
5. Update `MONITOR_*` thresholds if needed and re-validate drift logic.

### Debugging Model Failures
1. `pandas.read_parquet('data/feature_matrix_with_garch.parquet')` — check shape & nulls.
2. Verify P10 < P90 for every regime's predictions.
3. Inspect `outputs/shap_importance_*.png` — sudden feature swaps signal data issues. Top features should be economically grounded (VIX, realized vol, ATR).
4. Check `outputs/backtest_summary.json` / `walkforward_summary.json` — if coverage drops, widen intervals in M5.
5. Run `python run_pipeline.py --mode monitor` for drift diagnostics.

### Live Deployment
- Run `module8_live.py` (or `run_pipeline.py --mode live`) every Sunday via cron / Task Scheduler.
- Output: `outputs/strikes_live.json` — parse + execute via broker API (Kite).
- Track `outputs/backtest_equity_curve.png` weekly — Sharpe degradation ⇒ retrain.

### Maintenance Cadence
- **Weekly (post-expiry):** `--mode monitor`. Review `outputs/monitor_report_*.json`.
- **Monthly (or after regime change):** `--mode retrain`. Review SHAP + calibration curve.
- **After major market events:** `--mode macro`, then full retrain if monitor flags `RETRAIN`.

---

## 10. AI Collaboration Rules

When making changes in this repo, AI assistants **must**:

1. **Preserve the strike-placement math contract** in `module6_strikes.py` (Section 6).
2. **Maintain no-lookahead discipline** — chronological splits, walk-forward only.
3. **Keep `.env`-driven configuration** — never hardcode tunables.
4. **Use `loguru`, not `print`** — keep logs structured.
5. **Use Pydantic config** — add range validation for new parameters.
6. **Validate VIX regime logic** when touching M4/M6 — regimes drive model selection.
7. **Round strikes to 50** — always via `round_to_strike`/`round_to_50`.
8. **Update SHAP generation** if features change in M2.
9. **Run `--mode backtest` after any model/feature change**; re-run `--mode monitor` after retraining.
10. **Follow Conventional Commits** for commit messages.
11. **Be concise** — no prose explanations in code, no emojis, no unnecessary comments.
12. **Respect safety guards:** `WF_MAX_VIX_TRADE` and `WF_MIN_PREMIUM_PTS` are hard skip conditions — do not bypass them.
13. **Don't introduce new top-level dependencies** without explicit ask; check `requirements.txt` first.

### What to Avoid (Antipatterns)
- Mixing train/test data across time boundaries.
- Hardcoding strike buffers, wing widths, or quantile alphas in module files.
- Adding features that leak future information (e.g. week-ahead labels).
- Skipping the monitor after retraining.
- Optimizing on walk-forward test set directly (would invalidate the no-lookahead invariant).

---

## 11. Testing & Validation

- **Unit tests:** None (research). Validation is via backtest equity curve + monitor.
- **Static backtest:** `python run_pipeline.py --mode backtest` → `outputs/backtest_summary.json`.
- **Walk-forward:** `python run_pipeline.py --mode walkforward` → `outputs/walkforward_summary.json` (the realistic benchmark).
- **Monitor:** `python run_pipeline.py --mode monitor` → drift / coverage / win-rate alerts.
- **Visual checks:** `outputs/shap_importance_*.png`, `outputs/calibration_curve.png`, `outputs/*_equity_curve.png`.

### Reference Backtest Numbers (from README)

| Metric              | Static | Walk-Forward |
|---------------------|--------|--------------|
| Win rate            | ~96%   | ~91%         |
| Sharpe              | ~0.82  | ~0.44        |
| Max drawdown (pts)  | ~421   | ~1,090       |
| Expectancy/trade    | ~8.7   | ~5.5         |

The static vs. walk-forward gap (~5pp win rate) quantifies overfitting — always trust walk-forward numbers.

# Anti-Pattern Report — Nifty 50 IC Strategy Engine

Generated: May 2026 | Files analyzed: 17 Python modules

---

## 1. `sys.path.insert(0, ...)` Runtime Import Hacking

**Files**: `run_pipeline.py:33`, `module7b_walkforward.py:46`

Manipulates `sys.path` at import time instead of using a proper package structure with `__init__.py`. Breaks IDE tooling and makes import resolution order-dependent.

**Fix**: Add `__init__.py` and use proper relative imports.

---

## 2. `mode_retrain()` Is a No-Op Alias

**File**: `run_pipeline.py:228-238`

`mode_retrain()` just calls `mode_setup()`. CLAUDE.md claims it should "re-fit without modifying backtest set boundaries" but it actually re-fetches all data, re-engineers features, and re-trains from scratch. Identical behavior, different label.

**Fix**: Either differentiate the retrain path (skip data fetch, only re-fit models on existing features), or remove the alias and document that retrain = setup.

---

## 3. Nifty Expiry Weekday Is Wrong

**Files**: `module9_spreads.py:105`, `module1_data_pipeline.py:260`

- `module9` sets `NSE_EXPIRY_WEEKDAY = 1` (Tuesday) and claims "changed from Thursday ~2023." This is incorrect — that was Bank Nifty's move from Thursday to Wednesday. Nifty 50 index options expire on **Thursday**.
- `module1` resamples to `W-FRI` (Friday) — yet a third day assumption.

All expiry-date calculations and weekly resampling are based on the wrong day of the week.

**Fix**: Set `NSE_EXPIRY_WEEKDAY = 3` (Thursday). Align `build_nifty_weekly` to `W-THU`.

---

## 4. Double Config Loading (Pydantic + `os.getenv`)

**Files**: `module7b_walkforward.py:62-63`, `module9_spreads.py:358-360,378`, `module6_strikes.py`

Some values go through `config.py` (Pydantic) while others read directly from `os.getenv()`:

| Config Key | Loaded Via |
|---|---|
| `wf_sl_multiplier`, `wf_slippage_entry`, etc. | `config.py` |
| `WF_MIN_PREMIUM_PTS` | `os.getenv()` |
| `WF_MAX_VIX_TRADE` | `os.getenv()` |
| `SPREAD_WING_WIDTH_LOW/MID/HIGH_VIX` | `os.getenv()` |
| `SPREAD_DELTA_TARGET` | `os.getenv()` |

Two sources of truth, with env vars not documented in `config.py`'s Pydantic schema.

**Fix**: Move all env-sourced configs into `config.py` as Pydantic fields.

---

## 5. Magic Numbers as Fallback Values

**Files**: `module6_strikes.py:179-182`, `module7b_walkforward.py:223`

Hardcoded fallback `log_range_p10=0.015`, `log_range_p90=0.035` that never adapt to changing market regimes. Also: `random_state=42` in 6+ places, `r=0.065` (risk-free rate), `q=0.015` (dividend yield) duplicated across `module4b_risk`, `module9`, `module10`, `module12`.

**Fix**: Centralize all calibrated defaults in `config.py`.

---

## 6. `_is_event_week` Has No Year Awareness

**File**: `module2_features.py:56-61`

Calendar events (Budget, RBI, earnings) map month names without years:

```
"Feb": ["Budget Week - Week 1"]
```

Every February week-1 across **all** years triggers "budget week." These become stale/false positives for future years.

**Fix**: Represent event dates as `(year, month, week)` tuples or load from an external calendar JSON.

---

## 7. `_clear_model_cache()` — Dead Code

**File**: `module6_strikes.py`

Defined but never called anywhere in the codebase. REVIEW_FIXES_SUMMARY.md claims cache invalidation was fixed, but the invalidation trigger is never wired into any pipeline flow (`run_live_prediction` at line 410 never calls it).

**Fix**: Call `_clear_model_cache()` after retraining, or add a TTL-based check inside `_load_models_cached`.

---

## 8. Black-Scholes Duplicated Across Modules

**Files**: `module9_spreads.py:190-219`, `module10_nse_costs.py:82-109`

Both modules implement BS pricing independently with slightly different interfaces. The inline `_bs()` closure in `module10:estimate_ic_premium` is a copy-paste of `module9:estimate_bs_price`.

**Fix**: Extract a single `black_scholes.py` utility module.

---

## 9. GARCH Refit Skipped in Walk-Forward

**File**: `module7b_walkforward.py:321-323`

Comment says: "refitting GARCH every 4 weeks is expensive and marginal. Skip for speed." But the walk-forward is the **only** path that should update GARCH estimates. Without GARCH updates, the retrain cycle trains LightGBM on stale volatility features. The `_fit_garch` function exists and works but is deliberately unreachable in the main loop.

**Fix**: Wire `_fit_garch()` into the retrain cycle, or deprecate the GARCH module entirely if it's truly unused.

---

## 10. Stringly-Typed Regime/Direction

**Files**: Nearly all modules

Regimes (`"low"/"mid"/"high"`), directions (`"bull"/"bear"/"neutral"`), spread types (`"bull_put"/"bear_call"`) are all bare strings. No Enum, no type safety. A typo propagates silently until runtime.

**Fix**: Define `StrEnum` classes in `utils_constants.py` and use them everywhere.

---

## 11. `_RegimeLGBWrapper.predict()` O(3n) Per-Row Fallback

**File**: `module7b_walkforward.py:107-124`

When a regime model is missing, the fallback at lines 118-123 loops over **all** remaining models for *every single row*, calling `.predict(row.reshape(1, -1))` individually:

```python
preds[i] = np.median([
    (float(m["p10"].predict(row.reshape(1, -1))[0]) +
     float(m["p90"].predict(row.reshape(1, -1))[0])) / 2.0
    for m in self.lgb_models.values()
])
```

That's 2 model calls per available regime per fallback row.

**Fix**: Vectorize fallback predictions — batch all fallback rows through each model, then take median across the batch.

---

## 12. YFinance Has No Retry / Rate-Limit Handling

**Files**: `module1_data_pipeline.py`, `module1b_macro.py`

Yahoo Finance API calls have no retry logic, exponential backoff, or circuit breaker. A transient network blip crashes the pipeline.

**Fix**: Add a simple `@retry` decorator or a try/except retry loop with backoff.

---

## 13. Calibration Evaluation Uses Only 4% of Data

**File**: `module5_calibration.py:143-147`

80/20 train/test split, then 80/20 conf/eval split on the test slice = `0.20 * 0.20 = 4%` of total data for final calibration evaluation. Coverage estimates from this tiny slice are noisy and unreliable.

**Fix**: Use k-fold or a larger eval fraction (e.g., 80/20 conf/eval or 70/30).

---

## 14. Inversion Guard Silently Masks Quantile Crossing

**File**: `module4_model.py:228-231`

If P10 > P90 (quantile regression crossing), the code silently clamps `P10 = min(P10, P90)`. This is a model bug that should log a hard warning or raise an alert — silently fixing it hides that the model is degenerate.

**Fix**: Log a `logger.critical()` or throw a `ValueError` when quantile crossing is detected.

---

## Summary

| Severity | Count | Top Issue |
|----------|-------|-----------|
| Critical | 2 | Wrong expiry weekday (Thursday, not Tuesday); mode_retrain = mode_setup |
| High | 5 | Duplicate BS, dual config sources, stale event-weeks, dead cache invalidation, skipped GARCH refit |
| Medium | 7 | Magic numbers, stringly-typed, 4% calib data, no retry, inversion mask, per-row fallback, sys.path hack |

# Remaining Tasks — Credit Spread Implementation

**Status**: Core implementation **COMPLETE**. Cleanup + testing needed.

---

## Implementation Verification ✅

All 5 core functions fully coded:

| Task | File | Lines | Status |
|------|------|-------|--------|
| T1 NSE Expiry Calculator | module9_spreads.py | 58-95 | ✅ Complete |
| T2 Black-Scholes Pricing | module9_spreads.py | 101-169 | ✅ Complete |
| T4 Direction Signal | module9_spreads.py | 175-220 | ✅ Complete |
| T3 DTE-Scaled Spreads | module9_spreads.py | 226-304 | ✅ Complete |
| T5 Multi-Expiry Orchestrator | module9_spreads.py | 310-387 | ✅ Complete |
| Module8 Integration | module8_live.py | 219-256 | ✅ Complete |

---

## Code Quality Issues

### High Priority

**Issue #1: Dead code (L73-74)**
```python
if expiry_1 == today: # Should not happen with modulo unless today is Thu
     pass
```
Remove dead code block.

**Issue #2: Config unpacking (L245)**
```python
buffer_pts, wing_config, vix_baseline, _, min_buffer_pts, _ = _load_config()
```
Verify `module6_strikes._load_config()` returns exactly 6 values + order matches.

**Issue #3: predict_range dependency (L323)**
```python
range_pred = predict_range(feature_row)
```
Verify `module6_strikes.predict_range()` exists, returns `{"log_range_p10", "log_range_p90"}`.

### Medium Priority

**.env.example validation**
Confirm 3 keys added (from plan):
```
MIN_RR_RATIO=0.15
RISK_FREE_RATE=0.065
DIRECTION_CONFIDENCE_THRESHOLD=0.35
```

---

## Testing Checklist

### Unit Tests

- [ ] **T1**: `get_nse_expiries(date(2026, 4, 21))` returns 3 expiries with correct DTE
  - Next Thu: 2026-04-24 (DTE=3)
  - 7 days later: 2026-05-01 (DTE=10)
  - Monthly last Thu: 2026-05-28 (DTE=37)

- [ ] **T2**: Black-Scholes sanity checks
  - estimate_bs_price(24500, 23700, 9/365, 0.1879, r=0.065, 'put') ≈ 42-45 pts
  - OTM puts decrease with increasing strike
  
- [ ] **T4**: Direction signal
  - Bullish (gap > 0.05): direction = "bull"
  - Bearish (vix_change > 2.0): direction = "bear"
  - Low confidence (< 0.35): direction = "neutral"

- [ ] **T3**: Strike placement
  - bull_put: short_strike < spot < long_strike (wrong!)
  - Correct: long_strike < short_strike < spot
  - bear_call: spot < short_strike < long_strike
  
- [ ] **T5**: Output JSON structure
  - Has "spreads" list sorted by ev_proxy descending
  - Has "summary" with "top_pick"
  - All spreads have: expiry_date, expiry_type, spread_type, premium_pts, max_loss_pts, rr_ratio, pop_pct, ev_proxy, meets_min_rr

### Integration Test

```bash
python run_pipeline.py --mode live
# Should create both outputs/strikes_live.json AND outputs/spreads_live.json
```

### Manual Validation

- [ ] `outputs/spreads_live.json` is valid JSON
- [ ] Top pick has reasonable metrics (premium > 0, rr_ratio > 0.15, pop_pct > 0.70)
- [ ] 3 expiries (if neutral) or 1 expiry per direction
- [ ] Wing widths scale with DTE (larger = longer duration)
- [ ] POP > 0.90 for short spreads (conservative)

---

## Documentation

- [ ] **module9_spreads.py**: Add module-level docstring with usage example
- [ ] **ARCHITECTURE.md**: Add credit spread module section (data flow, new outputs)
- [ ] **README.md**: Add credit spread feature to "Features"
- [ ] **Example output**: Show sample spreads_live.json in docs

---

## Deployment Checklist

- [ ] Dead code removed
- [ ] Config dependencies verified
- [ ] Tests passing (or skip if research project)
- [ ] Live pipeline runs: `python run_pipeline.py --mode live`
- [ ] Both files created: `strikes_live.json` and `spreads_live.json`
- [ ] Push to GitHub main (clean commit message)

---

## Commit Message (when ready)

```
feat(module9): add credit spread generation system

- Bull put / bear call spreads across 3 expiries (weekly + monthly)
- Black-Scholes option pricing with VIX scalar
- DTE-aware scaling for strike placement and buffers
- Direction detection (bull/bear/neutral) from gap + VIX trend
- Multi-expiry orchestration with risk/return ranking (ev_proxy)
- Module8 Step 8 integration (non-blocking, JSON output)
- Config: MIN_RR_RATIO, RISK_FREE_RATE, DIRECTION_CONFIDENCE_THRESHOLD

Verified: All 5 core functions (T1-T5) implemented.
Remaining: Code cleanup (dead code), dependency verification, integration testing.
```

---

## Next Immediate Steps

1. Remove dead code (Issue #1)
2. Verify module6 functions exist (Issues #2-3)
3. Run: `python run_pipeline.py --mode live`
4. Verify spreads_live.json valid + contains expected data
5. Push to GitHub (caveman-commit style)
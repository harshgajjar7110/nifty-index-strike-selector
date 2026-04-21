# Fix Plan: Indian Options Market Accuracy (Issues #1–#6)

## Context

Backtest and module9 have 6 accuracy issues for real-money Nifty IC/spread trading. Issues #1–#3 affect backtest P&L realism. Issues #4–#6 affect module9 credit spread correctness. All fixes additive/corrective — no architecture changes.

**Key finding**: Issue #1 (costs not applied) was misdiagnosed — module7 already applies `pnl_net_list` correctly (line 288). Fix becomes: expose avg_cost in summary.

---

## Critical Files

| File | Issues | Lines |
|------|--------|-------|
| `module10_nse_costs.py` | #2 — wrong rates | 38, 46, 50 |
| `module7_backtest.py` | #1, #3 — summary + premium | 186–189, 358–376 |
| `module9_spreads.py` | #4, #5, #6 — expiry, BS, DTE | 63–98, 104–132, 242–253 |
| `.env.example` | #5 — new DIVIDEND_YIELD key | — |

---

## Fix #1 — Expose Cost Breakdown in Backtest Summary

**File**: `module7_backtest.py`

Costs ARE applied (line 288 uses `pnl_net_list`). Gap: `backtest_summary.json` doesn't show avg cost per trade.

**Add 3 keys to summary dict** (lines 358–376):
```python
"avg_txn_cost_pts":     round(valid["txn_costs_pts"].mean(), 2),
"gross_expectancy_pts": round(valid["pnl_gross"].mean(), 2),
"net_expectancy_pts":   round(valid["pnl_points"].mean(), 2),
```
`txn_costs_pts` column already populated at line 294.

---

## Fix #2 — NSE Charge Rates in module10

**File**: `module10_nse_costs.py`

Three rate errors:

| Line | Item | Current | Correct |
|------|------|---------|---------|
| 46 | SEBI fee | `0.0000001` (₹1/crore) | `0.000001` (₹10/crore) |
| 50 | Stamp duty | seller pays (`is_sell=True`) | buyer pays → `not is_sell` |
| 42 | Exchange charges | `0.000505` (0.0505%) | `0.00053` (0.053%) |

**Changes**:
```python
exchange_charges = 0.00053 * turnover           # line 42
sebi_fees        = 0.000001 * turnover           # line 46
stamp_duty = (0.00003 * turnover) if not is_sell else 0.0  # line 50
```

---

## Fix #3 — Replace Hardcoded 80-pt Premium with Black-Scholes

**File**: `module7_backtest.py` lines 184–189 + `module10_nse_costs.py`

Current: `int(80 * vix/baseline)` clamped [60, 120] — overstates real weekly IC premium ~5–8x.
Strikes already available after `generate_strikes()` (line 176–182).

**Step A — Add `estimate_ic_premium()` to `module10_nse_costs.py`** (after `apply_slippage`):
```python
def estimate_ic_premium(
    spot: float,
    short_put: float, long_put: float,
    short_call: float, long_call: float,
    dte_days: int,
    vix_level: float,
    r: float = 0.065,
    q: float = 0.015,       # Nifty dividend yield ~1.5%
) -> float:
    """Black-Scholes IC net credit: sell short legs, buy wing legs."""
    from scipy.stats import norm
    import numpy as np

    T = max(dte_days, 1) / 365.0
    sigma = max(vix_level / 100.0, 0.05)

    def _bs(S, K, opt):
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt == 'put':
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    net = (_bs(spot, short_put,  'put')
         - _bs(spot, long_put,   'put')
         + _bs(spot, short_call, 'call')
         - _bs(spot, long_call,  'call'))
    return max(net, 0.0)
```

**Step B — Update module7 import line 38**:
```python
from module10_nse_costs import calculate_nse_charges, apply_slippage, estimate_ic_premium
```

**Step C — Replace lines 184–189 in module7**:
```python
vix_used = vix_level if not np.isnan(vix_level) else VIX_BASELINE
bs_premium = estimate_ic_premium(
    spot=current_close,
    short_put=strikes["short_put"],  long_put=strikes["long_put"],
    short_call=strikes["short_call"], long_call=strikes["long_call"],
    dte_days=5,
    vix_level=vix_used,
)
premium_pts    = max(bs_premium, 5.0)
wing_width_used = strikes.get("wing_width_pts", 200)
max_loss_pts   = max(wing_width_used - premium_pts, 1.0)
```

---

## Fix #4 — NSE Holiday Calendar in module9 Expiry Calculator

**File**: `module9_spreads.py`, `get_nse_expiries()` (lines 63–98)

**Problem**: No holiday handling. If Thursday = NSE holiday, expiry moves to preceding Wednesday.

**Add after imports** in module9:
```python
NSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 3, 8),   date(2025, 4, 11),  date(2025, 4, 14),
    date(2025, 8, 15),  date(2025, 10, 2),  date(2025, 10, 20),
    date(2025, 10, 21), date(2025, 11, 5),  date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),   date(2026, 3, 26),  date(2026, 4, 3),
    date(2026, 4, 14),  date(2026, 8, 15),  date(2026, 10, 2),
}

def _adjust_for_holiday(d: date) -> date:
    """If date is NSE holiday or weekend, move to previous trading day."""
    while d in NSE_HOLIDAYS or d.weekday() >= 5:
        d -= timedelta(days=1)
    return d
```

**Update `get_nse_expiries()`** — wrap each expiry with `_adjust_for_holiday()`:
```python
expiry_1 = _adjust_for_holiday(today + timedelta(days=days_ahead))
expiry_2 = _adjust_for_holiday(expiry_1 + timedelta(days=7))
month_after = (expiry_2.month % 12) + 1
year_after  = expiry_2.year + (1 if expiry_2.month == 12 else 0)
expiry_3 = _adjust_for_holiday(_last_thursday_of_month(year_after, month_after))
```

---

## Fix #5 — Add Dividend Yield to Black-Scholes in module9

**File**: `module9_spreads.py`, `estimate_bs_price()` (lines 104–132)

**Problem**: d1 uses `(r + 0.5σ²)` — missing dividend yield `q`. Nifty yield ≈ 1.5%.

**Update signature** (line 104–111):
```python
def estimate_bs_price(
    S: float, K: float, T_years: float, sigma_annual: float,
    r: float = 0.065, option_type: str = 'put',
    q: float = 0.015,    # ADD: Nifty dividend yield
) -> float:
```

**Update d1 formula** (line 123):
```python
d1 = (np.log(S / K) + (r - q + 0.5 * sigma_annual**2) * T_years) / (sigma_annual * np.sqrt(T_years))
```

**Update pricing formulas** (lines 128–130) — add `np.exp(-q * T_years)` discount on spot:
```python
if option_type == 'call':
    price = S * np.exp(-q * T_years) * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
else:
    price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * np.exp(-q * T_years) * norm.cdf(-d1)
```

**Add to `.env.example`**:
```ini
# Nifty 50 approximate dividend yield for Black-Scholes pricing (~1.5%)
DIVIDEND_YIELD=0.015
```

---

## Fix #6 — DTE Scaling Direction

**File**: `module9_spreads.py`, line 242

**Finding**: CONFIRMED CORRECT. `sqrt(DTE/5)` → longer DTE = wider buffer/wing. Correct (more time = more potential movement). No code change needed.

Add comment only:
```python
# sqrt(DTE/5): wider buffer/wing for longer-dated spreads (more time to breach)
dte_scalar = np.sqrt(max(dte_days, 1) / 5.0)
```

---

## Implementation Order

1. `module10_nse_costs.py` — Fix #2 (rates) + Add `estimate_ic_premium()` for Fix #3
2. `module7_backtest.py` — Fix #1 (summary keys) + Fix #3 (use BS premium, update import)
3. `module9_spreads.py` — Fix #4 (holidays) + Fix #5 (dividend yield) + Fix #6 (comment)
4. `.env.example` — Add `DIVIDEND_YIELD=0.015`

---

## Verification

```bash
# Test NSE costs with corrected rates
python -c "
from module10_nse_costs import calculate_nse_charges, estimate_ic_premium
entry = calculate_nse_charges(80, num_legs=4, is_sell=True)
exit_ = calculate_nse_charges(80, num_legs=4, is_sell=False)
print('Entry costs:', entry['cost_per_lot_pts'], 'pts')
print('Exit costs:', exit_['cost_per_lot_pts'], 'pts')
ic = estimate_ic_premium(24000, 23500, 23300, 24500, 24700, 5, 16.0)
print('BS IC premium:', round(ic, 2), 'pts')
"

# Run backtest — check new summary keys + realistic expectancy
python run_pipeline.py --mode backtest
# Expect: avg_txn_cost_pts ~2-4 pts, net_expectancy significantly lower than current 75 pts

# Test expiry calculator with holiday date
python -c "
from datetime import date
from module9_spreads import get_nse_expiries
expiries = get_nse_expiries(date(2025, 12, 22))
print(expiries)  # expiry_1 should be 2025-12-24 (Wed), not 2025-12-25 (Christmas)
"

# Run full live pipeline
python run_pipeline.py --mode live
```

---

## Expected Impact on Backtest Metrics

| Metric | Current (pre-fix) | Post-fix (realistic) |
|--------|------------------|----------------------|
| Avg premium/trade | ~78 pts | ~10–15 pts (BS-derived) |
| Avg NSE cost/trade | ~1 pt (understated) | ~3–4 pts |
| Net expectancy | ~75 pts | ~3–8 pts |
| Win rate | 98% (unchanged) | 98% (strike logic unchanged) |
| Sharpe ratio | 18.6 | ~3–5 |

# Detailed Implementation Plan — Credit Spread Generation System

**Date:** 2026-04-21  
**Scope:** 5 tasks implementing bull put / bear call credit spreads across 3 expiries (weekly, monthly)  
**Output:** `module9_spreads.py` (new), updated `module8_live.py`, updated `.env.example`

---

## Overview

Current system generates **iron condor (IC)** strikes: 4 legs, direction-neutral, single weekly expiry.

New system generates **credit spreads**: 2 legs, directional, multi-expiry (3 choices per Sunday):
- **Bull Put Spread**: sell OTM put, buy further OTM put (bullish bias)
- **Bear Call Spread**: sell OTM call, buy further OTM call (bearish bias)

Direction determined by feature signals (ROC, VIX trend). Premium estimated via Black-Scholes (no options chain needed).

---

## T1 — NSE Expiry Calculator

### Purpose
Compute upcoming Nifty options expiry dates. NSE expiries:
- **Weekly**: every Thursday
- **Monthly**: last Thursday of each month

From today (2026-04-21, Tuesday), return 3 expiries with DTE and type.

### Expected Output (Example)
```python
[
  {"date": date(2026, 4, 23), "dte": 2,  "type": "weekly"},      # next Thu
  {"date": date(2026, 4, 30), "dte": 9,  "type": "monthly"},     # last Thu Apr
  {"date": date(2026, 5, 28), "dte": 37, "type": "monthly"}      # last Thu May
]
```

### Algorithm

#### Helper 1: `_last_thursday_of_month(year: int, month: int) -> date`
```
Input: year=2026, month=5
Step 1: Find last day of month (May 31)
Step 2: Walk backward until weekday == 3 (Thursday)
  May 31 = Sunday (weekday=6)
  May 30 = Saturday (5)
  May 29 = Friday (4)
  May 28 = Thursday (3) ✓
Step 3: Return date(2026, 5, 28)

Implementation:
  from calendar import monthrange
  last_day = monthrange(year, month)[1]
  d = date(year, month, last_day)
  while d.weekday() != 3:
    d -= timedelta(days=1)
  return d
```

#### Helper 2: `_is_last_thursday(d: date) -> bool`
```
Check if date d is a Thursday AND no later Thursday exists in same month.

Implementation:
  if d.weekday() != 3:
    return False
  next_thursday = d + timedelta(days=7)
  return next_thursday.month != d.month
```

#### Main: `get_nse_expiries(today: date) -> list[dict]`
```
Step 1: Calculate days until next Thursday
  days_ahead = (3 - today.weekday()) % 7  # 3 = Thu
  If today IS Thursday (weekday == 3): days_ahead = 0 (use today)
  Else: if days_ahead == 0, set to 7 (next week's Thu)
  
  Example (today = 2026-04-21, Tue):
    (3 - 1) % 7 = 2
    expiry_1 = 2026-04-21 + 2 days = 2026-04-23 (Thu) ✓

Step 2: Calculate second expiry (7 days later)
  expiry_2 = expiry_1 + timedelta(days=7)
  Example: 2026-04-23 + 7 = 2026-04-30 (Thu) ✓

Step 3: Calculate third expiry (last Thursday of month AFTER expiry_2)
  next_month = (expiry_2.month % 12) + 1
  next_year = expiry_2.year + (1 if expiry_2.month == 12 else 0)
  expiry_3 = _last_thursday_of_month(next_year, next_month)
  
  Example:
    expiry_2.month = 4 (April)
    next_month = 5 (May)
    expiry_3 = last Thursday of May = 2026-05-28 ✓

Step 4: For each (expiry_1, expiry_2, expiry_3):
  dte = (expiry_date - today).days
  is_monthly = _is_last_thursday(expiry_date)
  type = "monthly" if is_monthly else "weekly"
  
  Results:
    2026-04-23: dte=2, is_last_thursday=False → weekly
    2026-04-30: dte=9, is_last_thursday=True (May 7 is Thu, not in April) → monthly
    2026-05-28: dte=37, is_last_thursday=True (Jun 4 is Thu, not in May) → monthly
```

### Edge Cases
| Today | Expected Expiry 1 | Notes |
|-------|---|---|
| Tuesday 04-21 | Thursday 04-23 | (3-1)%7=2 |
| Thursday 04-23 pre-market | Thursday 04-23 | (3-3)%7=0, use today |
| Friday 04-24 | Thursday 04-30 | (3-4)%7=-1%7=6 |
| Sunday 04-27 | Thursday 04-30 | (3-6)%7=-3%7=4 |
| Month boundary (Apr 29) | Thu 05-02 next month | Algorithm handles seamlessly |

### Testing
```python
# Test case 1: Tuesday → Thursday +2 days
from datetime import date
exp = get_nse_expiries(date(2026, 4, 21))
assert exp[0]["date"] == date(2026, 4, 23)
assert exp[0]["dte"] == 2
assert exp[0]["type"] == "weekly"

# Test case 2: Second expiry is monthly
assert exp[1]["date"] == date(2026, 4, 30)
assert exp[1]["type"] == "monthly"

# Test case 3: Third expiry is also monthly
assert exp[2]["type"] == "monthly"
assert exp[2]["date"].month == 5  # May
assert exp[2]["date"].weekday() == 3  # Thursday
```

---

## T2 — Black-Scholes Premium Estimator

### Purpose
Estimate theoretical option prices using Black-Scholes. Needed because no real options chain available.

### Black-Scholes Formula
```
Call = S * N(d1) - K * e^(-r*T) * N(d2)
Put  = K * e^(-r*T) * N(-d2) - S * N(-d1)

Where:
  d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
  d2 = d1 - σ*√T
  N(x) = cumulative standard normal
  S = spot price
  K = strike
  r = risk-free rate (0.065 for India)
  T = time to expiry in years (DTE / 365)
  σ = annualized volatility (VIX / 100)
```

### Function 1: `estimate_bs_price(...) -> float`

**Signature:**
```python
def estimate_bs_price(
    S: float,
    K: float,
    T_years: float,
    sigma_annual: float,
    r: float = 0.065,
    option_type: str = 'put'
) -> float:
```

**Inputs:**
- `S`: Nifty spot price (e.g., 24500)
- `K`: Strike price (e.g., 23700)
- `T_years`: DTE / 365 (e.g., 9/365 = 0.0247 years)
- `sigma_annual`: VIX / 100 (e.g., 18.79 → 0.1879)
- `r`: Risk-free rate (default 6.5%)
- `option_type`: 'put' or 'call'

**Implementation:**
```python
from scipy.stats import norm
import numpy as np

def estimate_bs_price(S, K, T_years, sigma_annual, r=0.065, option_type='put'):
    # Guard: at expiry or beyond
    if T_years <= 0:
        if option_type == 'put':
            return max(K - S, 0.0)
        else:
            return max(S - K, 0.0)
    
    # Guard: zero or negative vol
    if sigma_annual <= 0:
        sigma_annual = 0.10
    
    # Compute d1, d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma_annual**2) * T_years) / (sigma_annual * np.sqrt(T_years))
    d2 = d1 - sigma_annual * np.sqrt(T_years)
    
    # Black-Scholes pricing
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return max(price, 0.0)
```

**Example (Nifty 24500, put strike 23700, 9 DTE, VIX 18.79):**
```
S=24500, K=23700, T=9/365=0.0247, σ=0.1879, r=0.065

d1 = [ln(24500/23700) + (0.065 + 0.5*0.1879²)*0.0247] / (0.1879*√0.0247)
   = [0.0318 + 0.00166] / (0.1879*0.157)
   = 0.0335 / 0.0295 = 1.135

d2 = 1.135 - 0.1879*0.157 = 1.135 - 0.0295 = 1.105

P = 23700*e^(-0.065*0.0247)*N(-1.105) - 24500*N(-1.135)
  = 23700*0.9984*0.1346 - 24500*0.1283
  = 3,193 - 3,143 = ~50 points

At-the-money Nifty put at 9 DTE with 18.79% IV should be ~50-60 pts ✓
```

### Function 2: `estimate_spread_premium(...) -> dict`

**Signature:**
```python
def estimate_spread_premium(
    S: float,
    short_K: float,
    long_K: float,
    T_years: float,
    sigma_annual: float,
    r: float = 0.065,
    spread_type: str = 'bull_put'
) -> dict:
```

**Logic:**
```
Bull Put Spread:
  - SELL put at short_K (closer to ATM, higher premium)
  - BUY put at long_K (further OTM, lower premium, lower loss)
  - Constraint: long_K < short_K (both below spot)
  
  Example: spot=24500, short_K=23700, long_K=23500
    - Sell 23700 put: collect 50 pts
    - Buy 23500 put: pay 35 pts
    - Net credit: 50-35 = 15 pts
    - Max loss if Nifty goes below 23500: 23700-23500-15 = 185 pts
    - R:R = 15/185 = 0.081 = 8.1%

Bear Call Spread:
  - SELL call at short_K (closer to ATM, higher premium)
  - BUY call at long_K (further OTM, lower premium, lower loss)
  - Constraint: long_K > short_K (both above spot)
  
  Example: spot=24500, short_K=25300, long_K=25500
    - Sell 25300 call: collect 35 pts
    - Buy 25500 call: pay 20 pts
    - Net credit: 35-20 = 15 pts
    - Max loss if Nifty goes above 25500: 25500-25300-15 = 185 pts
    - R:R = 15/185 = 0.081 = 8.1%
```

**Implementation:**
```python
def estimate_spread_premium(S, short_K, long_K, T_years, sigma_annual, r=0.065, spread_type='bull_put'):
    if spread_type == 'bull_put':
        # Bull put: long_K < short_K
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'put')
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'put')
        premium_pts = short_price - long_price
        wing_width  = short_K - long_K
        
    else:  # bear_call
        # Bear call: long_K > short_K
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'call')
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'call')
        premium_pts = short_price - long_price
        wing_width  = long_K - short_K
    
    # Max loss = width - premium collected
    max_loss_pts = wing_width - premium_pts
    
    # R:R ratio
    if max_loss_pts > 0:
        rr_ratio = premium_pts / max_loss_pts
    else:
        rr_ratio = 0.0
    
    # Breakeven
    if spread_type == 'bull_put':
        breakeven = short_K - premium_pts
    else:
        breakeven = short_K + premium_pts
    
    return {
        "premium_pts":  round(premium_pts, 2),
        "max_loss_pts": round(max_loss_pts, 2),
        "rr_ratio":     round(rr_ratio, 4),
        "breakeven":    round(breakeven, 2),
        "wing_width":   wing_width,
    }
```

### Testing
```python
# Bull put spread
result = estimate_spread_premium(24500, 23700, 23500, 9/365, 0.1879)
assert result["wing_width"] == 200
assert result["premium_pts"] > 0
assert result["max_loss_pts"] == 200 - result["premium_pts"]
assert 0 < result["rr_ratio"] < 1  # typical OTM spread
assert result["breakeven"] == 23700 - result["premium_pts"]

# Bear call spread
result2 = estimate_spread_premium(24500, 25300, 25500, 9/365, 0.1879, spread_type='bear_call')
assert result2["wing_width"] == 200
assert result2["premium_pts"] > 0
```

---

## T3 — DTE-Aware Credit Spread Generator

### Purpose
Generate bull put or bear call strikes for a given expiry, with DTE-adjusted buffers and wings.

### DTE Scaling Rationale
Price range grows with sqrt(time):
```
Weekly range (5 DTE) = baseline
Monthly range (25 DTE) = baseline * sqrt(25/5) = baseline * sqrt(5) ≈ baseline * 2.24

Buffer and wing width scale the same way.
```

### Function: `generate_credit_spread(...) -> dict`

**Signature:**
```python
def generate_credit_spread(
    spot: float,
    log_range_p10: float,
    log_range_p90: float,
    dte_days: int,
    vix_level: float,
    garch_vol: float | None,
    direction: str,  # 'bull_put' or 'bear_call'
    r: float = 0.065
) -> dict:
```

**Inputs:**
- `spot`: Current Nifty level (e.g., 24354)
- `log_range_p10`, `log_range_p90`: Log-price range bounds (from module6 prediction)
- `dte_days`: Days to expiry (2, 9, or 37)
- `vix_level`: Current VIX (e.g., 18.79)
- `garch_vol`: Weekly GARCH volatility (e.g., 0.012)
- `direction`: 'bull_put' or 'bear_call'

**Algorithm:**

#### Step 1: DTE Scaling Factor
```python
dte_scalar = np.sqrt(dte_days / 5.0)
# For DTE=2:  sqrt(2/5) = 0.632
# For DTE=9:  sqrt(9/5) = 1.342
# For DTE=37: sqrt(37/5) = 2.722
```

#### Step 2: Convert Log-Price Range to Points
```python
half_range_p10 = spot * (np.exp(log_range_p10) - 1) / 2
half_range_p90 = spot * (np.exp(log_range_p90) - 1) / 2

# Example: spot=24354, log_range_p90=0.09 (9% range)
# half_range_p90 = 24354 * (e^0.09 - 1) / 2
#                = 24354 * (1.0942 - 1) / 2
#                = 24354 * 0.047 = ~1,145 points
```

#### Step 3: Load Base Configuration
```python
from module6_strikes import _load_config
buffer_pts, wing_config, vix_baseline, _, min_buffer_pts, _ = _load_config()
# Returns: e.g. (50, {'low':150, 'mid':200, 'high':250}, 16.0, 0, 75, 50)
```

#### Step 4: VIX-Scaled Buffer
```python
vix_scalar = vix_level / vix_baseline
base_buffer = np.clip(buffer_pts * vix_scalar, min_buffer_pts, 150)
# For VIX=18.79, vix_baseline=16: scalar=1.174
# base_buffer = clip(50 * 1.174, 75, 150) = 87.5 pts

# Apply DTE scaling
scaled_buffer = np.clip(base_buffer * dte_scalar, min_buffer_pts, 300)
# For DTE=9: scaled_buffer = clip(87.5 * 1.342, 75, 300) = 117 pts
```

#### Step 5: DTE/VIX-Scaled Wing Width
```python
# Select base wing width by VIX regime
if vix_level < 13:
    base_wing = wing_config['low']     # 150 pts
elif vix_level < 20:
    base_wing = wing_config['mid']     # 200 pts
else:
    base_wing = wing_config['high']    # 250 pts

# For VIX=18.79: base_wing = 200

# Scale by DTE
scaled_wing = base_wing * dte_scalar
# For DTE=9: scaled_wing = 200 * 1.342 = 268 pts

# Round to nearest 50 pts and apply floor
scaled_wing = round_to_strike(scaled_wing, interval=50)
scaled_wing = max(scaled_wing, 100)
# Result: 268 → rounds to 250 or 300 (assuming round_to_strike rounding logic)
```

#### Step 6: Strike Placement

**Bull Put:**
```
Short put = round_to_strike(spot - half_range_p90 - scaled_buffer)
Long put  = short_put - scaled_wing

Example: spot=24354, half_range_p90=1145, scaled_buffer=117, scaled_wing=250
  short_put = round_to_strike(24354 - 1145 - 117) = round_to_strike(23092) ≈ 23100
  long_put  = 23100 - 250 = 22850
```

**Bear Call:**
```
Short call = round_to_strike(spot + half_range_p90 + scaled_buffer)
Long call  = short_call + scaled_wing

Example: spot=24354, half_range_p90=1145, scaled_buffer=117, scaled_wing=250
  short_call = round_to_strike(24354 + 1145 + 117) = round_to_strike(25616) ≈ 25600
  long_call  = 25600 + 250 = 25850
```

#### Step 7: Probability of Profit (POP)
```
Using lognormal model from module6, scaled to DTE:

garch_vol_scaled = garch_vol * sqrt(dte_days / 5.0)

For bull put (probability spot stays above short_put):
  z = ln(spot / short_put) / garch_vol_scaled
  pop_pct = norm.cdf(z)

For bear call (probability spot stays below short_call):
  z = ln(short_call / spot) / garch_vol_scaled
  pop_pct = norm.cdf(z)

Example: spot=24354, short_put=23100, garch_vol=0.012, dte=9
  garch_scaled = 0.012 * sqrt(9/5) = 0.012 * 1.342 = 0.0161
  z = ln(24354/23100) / 0.0161 = 0.0534 / 0.0161 = 3.32
  pop_pct = norm.cdf(3.32) ≈ 0.9995 = 99.95%
```

#### Step 8: Call T2 for Premium/RR
```python
T_years = dte_days / 365.0
sigma_annual = vix_level / 100.0
spread_metrics = estimate_spread_premium(
    spot, short_strike, long_strike, T_years, sigma_annual, r, direction
)
# Returns: {premium_pts, max_loss_pts, rr_ratio, breakeven, wing_width}
```

#### Step 9: Build Output Dict
```python
return {
    "spot":           spot,
    "spread_type":    direction,  # 'bull_put' or 'bear_call'
    "short_strike":   short_strike,
    "long_strike":    long_strike,
    "wing_width":     scaled_wing,
    "scaled_buffer":  round(scaled_buffer, 2),
    "premium_pts":    spread_metrics["premium_pts"],
    "max_loss_pts":   spread_metrics["max_loss_pts"],
    "rr_ratio":       spread_metrics["rr_ratio"],
    "breakeven":      spread_metrics["breakeven"],
    "pop_pct":        round(pop_pct, 4) if pop_pct is not None else None,
    "dte_days":       dte_days,
}
```

### Testing
```python
# Test case 1: Weekly vs Monthly scaling
weekly = generate_credit_spread(24354, 0.045, 0.09, 2, 18.79, 0.012, 'bull_put')
monthly = generate_credit_spread(24354, 0.045, 0.09, 37, 18.79, 0.012, 'bull_put')

assert monthly["short_strike"] < weekly["short_strike"]  # more OTM for longer DTE
assert monthly["wing_width"] > weekly["wing_width"]      # wider wings
assert monthly["premium_pts"] > weekly["premium_pts"]    # more premium

# Test case 2: Bull put vs bear call
bp = generate_credit_spread(24354, 0.045, 0.09, 9, 18.79, 0.012, 'bull_put')
bc = generate_credit_spread(24354, 0.045, 0.09, 9, 18.79, 0.012, 'bear_call')

assert bp["short_strike"] < 24354  # below spot
assert bc["short_strike"] > 24354  # above spot
```

---

## T4 — Direction Signal Detector

### Purpose
Use feature row signals to determine bullish/bearish/neutral market direction.

### Available Feature Columns
From `module2_features.py`:
- ✓ `prev_week_gap`: (weekly_open - prior_close) / prior_close (momentum)
- ✓ `vix_level`: Current India VIX
- ✓ `vix_change_1w`: Change in VIX over 1 week
- ✓ `garch_sigma_mean`: Weekly GARCH volatility estimate
- ✓ `is_event_week`: 1 if budget/RBI/earnings week, 0 otherwise
- ✗ `roc_5_week` — DOES NOT EXIST, use `prev_week_gap` instead

### Function: `detect_direction(feature_row: pd.Series) -> dict`

**Algorithm:**

#### Step 1: Extract Signals (with Fallbacks)
```python
def _safe_get(row, col, default):
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default

roc_proxy    = _safe_get(row, "prev_week_gap", 0.0)      # gap up = bullish
vix_change   = _safe_get(row, "vix_change_1w", 0.0)      # rising = bearish
garch_vol    = _safe_get(row, "garch_sigma_mean", 0.012)
is_event     = int(_safe_get(row, "is_event_week", 0))
```

#### Step 2: Normalize Signals to [-1, +1]
```
Signal 1 — Momentum (ROC):
  roc_score = clip(prev_week_gap / 0.05, -1, 1)
  # ±5% gap = full signal strength
  # Example: gap +3% → 3%/5% = 0.6 (bullish)
  # Example: gap -2% → -2%/5% = -0.4 (bearish)

Signal 2 — VIX Trend:
  vix_trend_score = clip(-vix_change_1w / 2.0, -1, 1)
  # Rising VIX (positive change) → negative score (bearish)
  # Falling VIX (negative change) → positive score (bullish)
  # Example: VIX -2 pts → -(-2)/2 = +1.0 (bullish)
  # Example: VIX +3 pts → -(+3)/2 = -1.5 → clip to -1.0 (bearish)

Signal 3 — GARCH Acceleration (proxy):
  garch_acc_score = clip(-vix_change_1w / 3.0, -1, 1)
  # Similar to VIX trend but lower sensitivity (vol decel = bullish)
```

#### Step 3: Composite Score
```python
weights = {
    "roc_score":       0.45,  # primary signal
    "vix_trend_score": 0.35,  # secondary
    "garch_acc_score": 0.20,  # tertiary
}

composite = sum(score * weights[name] for name, score in signals.items())
# Result: composite in [-1, +1]
# > 0 = bullish, < 0 = bearish, ≈ 0 = neutral
```

#### Step 4: Confidence Calculation
```python
# Confidence = magnitude of composite score
# Event weeks reduce confidence by 30%
raw_confidence = abs(composite)
event_penalty = 0.70 if is_event else 1.0
confidence = raw_confidence * event_penalty

# Example 1: composite=0.6, not event → confidence=0.6
# Example 2: composite=0.6, is event → confidence=0.6*0.7=0.42
# Example 3: composite=0.2, not event → confidence=0.2 (too low)
```

#### Step 5: Direction Classification
```python
threshold = DIRECTION_CONFIDENCE_THRESHOLD  # from .env, default 0.35

if confidence < threshold:
    direction = "neutral"     # confidence too low
elif composite > 0:
    direction = "bull"        # composite positive
else:
    direction = "bear"        # composite negative
```

**Implementation:**
```python
import os
import numpy as np
import pandas as pd

def detect_direction(feature_row: pd.Series) -> dict:
    # Load config
    threshold = float(os.getenv("DIRECTION_CONFIDENCE_THRESHOLD", 0.35))
    
    # Extract signals
    roc = _safe_get(feature_row, "prev_week_gap", 0.0)
    vix_chg = _safe_get(feature_row, "vix_change_1w", 0.0)
    is_event = int(_safe_get(feature_row, "is_event_week", 0))
    
    # Normalize
    signals = {
        "roc_score":        np.clip(roc / 0.05, -1, 1),
        "vix_trend_score":  np.clip(-vix_chg / 2.0, -1, 1),
        "garch_acc_score":  np.clip(-vix_chg / 3.0, -1, 1),
    }
    
    # Composite
    weights = {"roc_score": 0.45, "vix_trend_score": 0.35, "garch_acc_score": 0.20}
    composite = sum(signals[k] * w for k, w in weights.items())
    
    # Confidence
    raw_confidence = abs(composite)
    event_penalty = 0.70 if is_event else 1.0
    confidence = raw_confidence * event_penalty
    
    # Direction
    if confidence < threshold:
        direction = "neutral"
    elif composite > 0:
        direction = "bull"
    else:
        direction = "bear"
    
    return {
        "direction":      direction,
        "confidence":     round(confidence, 4),
        "composite":      round(composite, 4),
        "signals":        {k: round(v, 4) for k, v in signals.items()},
        "is_event_week":  bool(is_event),
        "threshold":      threshold,
    }
```

### Testing
```python
# Test 1: Bullish row
row = pd.Series({
    "prev_week_gap": 0.03,           # +3% gap
    "vix_change_1w": -2.0,           # VIX down 2
    "vix_level": 16.0,
    "garch_sigma_mean": 0.010,
    "is_event_week": 0,
})
result = detect_direction(row)
assert result["direction"] == "bull"
assert result["confidence"] > 0.35

# Test 2: Bearish row
row2 = pd.Series({
    "prev_week_gap": -0.04,          # -4% gap
    "vix_change_1w": 3.5,            # VIX up 3.5
    "vix_level": 22.0,
    "garch_sigma_mean": 0.018,
    "is_event_week": 0,
})
result2 = detect_direction(row2)
assert result2["direction"] == "bear"

# Test 3: Neutral (event week)
row3 = pd.Series({
    "prev_week_gap": 0.02,
    "vix_change_1w": 0.0,
    "vix_level": 16.0,
    "garch_sigma_mean": 0.012,
    "is_event_week": 1,  # event week
})
result3 = detect_direction(row3)
# Confidence may drop below threshold due to event penalty
```

---

## T5 — Multi-Expiry Orchestrator + Module8 Integration

### Purpose
Main orchestrator: loop through 3 expiries, generate spreads for each, rank by expected value, write JSON output, integrate into module8.

### Function: `generate_all_spreads(...) -> dict`

**Signature:**
```python
def generate_all_spreads(
    feature_row: pd.Series,
    spot: float,
    vix_level: float,
    garch_vol: float | None,
    r: float | None = None
) -> dict:
```

**Algorithm:**

#### Step 1: Predict Price Range
```python
from module6_strikes import predict_range

range_pred = predict_range(feature_row)
log_p10 = range_pred["log_range_p10"]
log_p90 = range_pred["log_range_p90"]
# Example: log_p10=0.045, log_p90=0.090
```

#### Step 2: Get Expiries
```python
from datetime import date

expiries = get_nse_expiries(date.today())
# Returns: [
#   {"date": date(2026, 4, 23), "dte": 2, "type": "weekly"},
#   {"date": date(2026, 4, 30), "dte": 9, "type": "monthly"},
#   {"date": date(2026, 5, 28), "dte": 37, "type": "monthly"},
# ]
```

#### Step 3: Get Direction Signal
```python
direction_result = detect_direction(feature_row)
direction = direction_result["direction"]
# Result: "bull", "bear", or "neutral"

# Map to spread types
if direction == "bull":
    spread_types = ["bull_put"]        # bullish → sell puts
elif direction == "bear":
    spread_types = ["bear_call"]       # bearish → sell calls
else:  # neutral
    spread_types = ["bull_put", "bear_call"]  # generate both
```

#### Step 4: Generate Spreads for Each Expiry × Direction
```python
all_spreads = []

for expiry_dict in expiries:
    for spread_dir in spread_types:
        spread = generate_credit_spread(
            spot=spot,
            log_range_p10=log_p10,
            log_range_p90=log_p90,
            dte_days=expiry_dict["dte"],
            vix_level=vix_level,
            garch_vol=garch_vol,
            direction=spread_dir,
            r=r,
        )
        
        # Enhance with metadata
        spread["expiry_date"] = expiry_dict["date"].isoformat()
        spread["expiry_type"] = expiry_dict["type"]
        
        # Compute expected value proxy
        pop = spread["pop_pct"] if spread["pop_pct"] is not None else 0.70
        spread["ev_proxy"] = round(spread["rr_ratio"] * pop, 4)
        
        # Check if meets minimum R:R requirement
        min_rr = float(os.getenv("MIN_RR_RATIO", 0.15))
        spread["meets_min_rr"] = spread["rr_ratio"] >= min_rr
        
        all_spreads.append(spread)
```

#### Step 5: Rank by EV Proxy
```python
all_spreads.sort(key=lambda x: x["ev_proxy"], reverse=True)
# Highest expected value first
```

#### Step 6: Build Output Dict
```python
result = {
    "generated_at":     date.today().isoformat(),
    "spot":             round(spot, 2),
    "vix_level":        round(vix_level, 2),
    "direction_signal": direction_result,
    "expiries":         [e["date"].isoformat() for e in expiries],
    "spreads":          all_spreads,
    "summary": {
        "total_spreads": len(all_spreads),
        "top_pick":      all_spreads[0] if all_spreads else None,
    }
}
```

#### Step 7: Write to JSON
```python
import json
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

output_file = OUTPUTS_DIR / "spreads_live.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=2, default=str)

logger.success(f"Spreads written to {output_file}")
return result
```

### Module8 Integration

**File:** `module8_live.py`  
**Location:** After existing Step 7 (IC generation), before exit

**Code to Add:**
```python
# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Credit Spreads (module9) — non-blocking additive
# ─────────────────────────────────────────────────────────────────────────────

step = "credit spread generation (module9)"
logger.info(f"{'─'*60}")
logger.info(f"  {step}")
logger.info(f"{'─'*60}")

try:
    from module9_spreads import generate_all_spreads
    
    spreads_result = generate_all_spreads(
        feature_row=feature_row,
        spot=spot,
        vix_level=vix_level if vix_level else 16.0,
        garch_vol=garch_vol,
    )
    
    # Write output
    SPREADS_JSON = OUTPUTS_DIR / "spreads_live.json"
    with open(SPREADS_JSON, "w") as fh:
        json.dump(spreads_result, fh, indent=2, default=str)
    logger.success(f"Credit spreads → {SPREADS_JSON}")
    
    # Console summary
    top = spreads_result.get("summary", {}).get("top_pick")
    if top:
        print(f"\n  ★ Top Credit Spread")
        print(f"    Type:  {top['spread_type']} | Expiry: {top['expiry_date']}")
        print(f"    Legs:  {top['short_strike']} / {top['long_strike']} (±{top['wing_width']} pts)")
        print(f"    Premium: {top['premium_pts']:.1f} pts | Max Loss: {top['max_loss_pts']:.1f} pts")
        print(f"    R:R: {top['rr_ratio']:.2%} | POP: {top.get('pop_pct', 0)*100:.1f}% | EV: {top['ev_proxy']:.3f}")
    
except ImportError:
    logger.warning("module9_spreads not found — skipping credit spread generation")
except Exception as e9:
    logger.warning(f"Credit spread generation failed (non-fatal): {e9}")
    import traceback
    traceback.print_exc()
```

### .env.example Additions

```ini
# ─────────────────────────────────────────────────────────────────────────────
# Credit Spread Configuration (module9_spreads.py)
# ─────────────────────────────────────────────────────────────────────────────

# Minimum reward:risk ratio for a spread to be considered acceptable
# Example: 0.15 means "collect at least 15% of max loss as premium"
MIN_RR_RATIO=0.15

# Annualized risk-free rate for Black-Scholes option pricing
# (India 91-day T-bill yield, ~6.5%)
RISK_FREE_RATE=0.065

# Minimum direction confidence (0-1) before switching from neutral to bull/bear
# Below this, both bull_put and bear_call spreads generated
# (event weeks reduce confidence by 30%)
DIRECTION_CONFIDENCE_THRESHOLD=0.35
```

### Output JSON Example

```json
{
  "generated_at": "2026-04-21",
  "spot": 24354.00,
  "vix_level": 18.79,
  "direction_signal": {
    "direction": "bull",
    "confidence": 0.62,
    "composite": 0.62,
    "signals": {
      "roc_score": 0.6,
      "vix_trend_score": -0.75,
      "garch_acc_score": -0.5
    },
    "is_event_week": false,
    "threshold": 0.35
  },
  "expiries": [
    "2026-04-23",
    "2026-04-30",
    "2026-05-28"
  ],
  "spreads": [
    {
      "expiry_date": "2026-04-30",
      "expiry_type": "monthly",
      "dte": 9,
      "spread_type": "bull_put",
      "short_strike": 23700,
      "long_strike": 23500,
      "wing_width": 200,
      "scaled_buffer": 117.25,
      "premium_pts": 42.5,
      "max_loss_pts": 157.5,
      "rr_ratio": 0.2698,
      "breakeven": 23657.5,
      "pop_pct": 0.9234,
      "ev_proxy": 0.249,
      "meets_min_rr": true,
      "spot": 24354.0
    },
    {
      "expiry_date": "2026-04-23",
      "expiry_type": "weekly",
      "dte": 2,
      "spread_type": "bull_put",
      "short_strike": 24000,
      "long_strike": 23800,
      "wing_width": 200,
      "scaled_buffer": 82.15,
      "premium_pts": 35.2,
      "max_loss_pts": 164.8,
      "rr_ratio": 0.2137,
      "breakeven": 23964.8,
      "pop_pct": 0.8956,
      "ev_proxy": 0.191,
      "meets_min_rr": true,
      "spot": 24354.0
    }
  ],
  "summary": {
    "total_spreads": 3,
    "top_pick": "..."
  }
}
```

---

## File Structure — module9_spreads.py

```
1. Docstring
2. Imports
3. Constants (BASE_DIR, OUTPUTS_DIR, defaults)
4. Helper Functions
   - _safe_get(row, col, default)
   - _last_thursday_of_month(year, month)
   - _is_last_thursday(d)
5. T1 — get_nse_expiries(today)
6. T2a — estimate_bs_price(...)
7. T2b — estimate_spread_premium(...)
8. T4 — detect_direction(feature_row)
9. T3 — generate_credit_spread(...)
10. T5 — generate_all_spreads(feature_row, spot, vix_level, garch_vol, r)
11. if __name__ == "__main__": smoke test
```

---

## Testing & Verification Checklist

- [ ] T1: `get_nse_expiries(date(2026, 4, 21))` returns correct 3 expiries + DTE
- [ ] T2: `estimate_bs_price(24500, 23700, 9/365, 0.1879)` ≈ 50 pts (sanity check)
- [ ] T2: Bull put spread premium < wing width (always profitable at expiry if ITM)
- [ ] T3: Monthly wing width > weekly wing width (DTE scaling)
- [ ] T4: `detect_direction()` with bullish row → "bull", bearish → "bear"
- [ ] T4: Event weeks reduce confidence correctly
- [ ] T5: `generate_all_spreads()` returns 3-6 spreads, sorted by EV
- [ ] T5: `outputs/spreads_live.json` written correctly
- [ ] Module8: IC generation still works (non-breaking)
- [ ] Module8: Step 8 failure doesn't crash pipeline
- [ ] .env.example: all 3 new keys present with defaults

---

## Edge Case Handling

| Edge Case | Handling |
|-----------|----------|
| DTE = 0 (same-day Thursday) | T_years = max(dte/365, 1/365); BS returns intrinsic |
| VIX = 0 (impossible but guard) | sigma_annual floor = 0.10 in estimate_bs_price |
| premium_pts ≤ 0 (deep OTM) | Still return spread, mark meets_min_rr=False, log warning |
| max_loss_pts ≤ 0 (impossible for OTM) | Clamp to wing_width, rr_ratio = 0 |
| garch_vol = None | POP becomes None, fallback to 0.70 in EV calc |
| feature_row missing columns | _safe_get returns defaults; direction = "neutral" |
| predict_range() file missing | generate_all_spreads() catches FileNotFoundError, re-raise |
| module9 import fails | module8 try/except ImportError, IC unaffected |
| neutral direction | generates 6 spreads (3 expiries × 2 types), still ranks OK |

---

## Dependencies

**Existing (already in requirements.txt):**
- `numpy` — array math, sqrt, exp, log, clip
- `scipy.stats.norm` — cumulative normal CDF
- `pandas` — Series manipulation
- `loguru` — logging
- `python-dotenv` — load .env vars

**From existing modules (import, no circular deps):**
- `module6_strikes.py`: `round_to_strike()`, `_load_config()`, `predict_range()`

**Standard library:**
- `datetime`, `calendar` — date math
- `json` — output serialization
- `pathlib.Path` — file paths
- `os` — getenv

---

## Summary

| Task | Lines | Complexity | Dependencies |
|------|-------|-----------|---|
| T1 | ~50 | Low | stdlib (date, calendar) |
| T2 | ~80 | Medium | scipy.stats.norm, numpy |
| T3 | ~120 | High | T1, T2, module6 functions |
| T4 | ~70 | Medium | pandas, numpy |
| T5 | ~100 | High | T1-T4, module6, module8 integration |
| **Total** | **~420** | **High** | **All listed above** |

**Module8 addition:** ~40 lines (import-guarded, non-breaking)  
**.env.example addition:** 6 lines (3 keys + comments)

**Estimated development time with testing:** 2-3 hours

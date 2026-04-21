"""
Module 9: Credit Spread Generation System
Implements bull put and bear call credit spreads across multiple expiries (weekly, monthly).
Includes Black-Scholes premium estimation and direction signal detection.
"""

import os
import json
from datetime import date, timedelta
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger
from dotenv import load_dotenv

# Import from existing modules
from module6_strikes import round_to_strike, _load_config, predict_range

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_get(row, col, default):
    """Safely get a value from a pandas Series."""
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default

def _last_thursday_of_month(year: int, month: int) -> date:
    """Find the last Thursday of a given month and year."""
    last_day = monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != 3:  # 3 is Thursday
        d -= timedelta(days=1)
    return d

def _is_last_thursday(d: date) -> bool:
    """Check if date d is the last Thursday of its month."""
    if d.weekday() != 3:
        return False
    next_thursday = d + timedelta(days=7)
    return next_thursday.month != d.month

# ---------------------------------------------------------------------------
# T1 — NSE Expiry Calculator
# ---------------------------------------------------------------------------

def get_nse_expiries(today: date) -> list[dict]:
    """
    Compute upcoming Nifty options expiry dates (weekly and monthly).
    Returns 3 expiries: next Thu, 7 days later, and the following month's last Thu.
    """
    # Expiry 1: next Thursday (or today if today is Thursday)
    days_ahead = (3 - today.weekday()) % 7
    # If today is Thu, days_ahead is 0, which is correct for pre-market/intraday
    # But usually Sunday runner wants the upcoming Thu
    if days_ahead == 0 and today.weekday() == 3:
        expiry_1 = today
    else:
        # If today is Fri/Sat/Sun, it will find next Thu correctly
        # (3 - 4)%7 = 6 (Fri -> Thu), (3 - 6)%7 = 4 (Sun -> Thu)
        expiry_1 = today + timedelta(days=days_ahead)
        if expiry_1 == today: # Should not happen with modulo unless today is Thu
             pass

    # Expiry 2: 7 days after expiry 1
    expiry_2 = expiry_1 + timedelta(days=7)

    # Expiry 3: Last Thursday of month AFTER expiry_2
    next_month = (expiry_2.month % 12) + 1
    next_year = expiry_2.year + (1 if expiry_2.month == 12 else 0)
    expiry_3 = _last_thursday_of_month(next_year, next_month)

    expiries = [expiry_1, expiry_2, expiry_3]
    results = []
    for d in expiries:
        dte = (d - today).days
        is_monthly = _is_last_thursday(d)
        results.append({
            "date": d,
            "dte": max(dte, 0),
            "type": "monthly" if is_monthly else "weekly"
        })
    
    return results

# ---------------------------------------------------------------------------
# T2 — Black-Scholes Premium Estimator
# ---------------------------------------------------------------------------

def estimate_bs_price(
    S: float,
    K: float,
    T_years: float,
    sigma_annual: float,
    r: float = 0.065,
    option_type: str = 'put'
) -> float:
    """Estimate theoretical option price using Black-Scholes."""
    # Guard: at expiry or beyond
    if T_years <= 0:
        # Use 1 day as minimum for pricing if dte is 0
        T_years = 1.0 / 365.0
    
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

def estimate_spread_premium(
    S: float,
    short_K: float,
    long_K: float,
    T_years: float,
    sigma_annual: float,
    r: float = 0.065,
    spread_type: str = 'bull_put'
) -> dict:
    """Estimate net credit and metrics for a bull put or bear call spread."""
    if spread_type == 'bull_put':
        # Bull put: sell short_K, buy long_K (long_K < short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'put')
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'put')
        premium_pts = short_price - long_price
        wing_width  = short_K - long_K
    else:  # bear_call
        # Bear call: sell short_K, buy long_K (long_K > short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'call')
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'call')
        premium_pts = short_price - long_price
        wing_width  = long_K - short_K
    
    # Max loss = width - premium collected
    max_loss_pts = max(wing_width - premium_pts, 0.0)
    
    # R:R ratio
    rr_ratio = premium_pts / max_loss_pts if max_loss_pts > 0 else 0.0
    
    # Breakeven
    breakeven = (short_K - premium_pts) if spread_type == 'bull_put' else (short_K + premium_pts)
    
    return {
        "premium_pts":  round(premium_pts, 2),
        "max_loss_pts": round(max_loss_pts, 2),
        "rr_ratio":     round(rr_ratio, 4),
        "breakeven":    round(breakeven, 2),
        "wing_width":   wing_width,
    }

# ---------------------------------------------------------------------------
# T4 — Direction Signal Detector
# ---------------------------------------------------------------------------

def detect_direction(feature_row: pd.Series) -> dict:
    """
    Determine bullish/bearish/neutral direction based on feature signals.
    Uses momentum (gap) and VIX trend.
    """
    # Load config from env
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    threshold = float(os.getenv("DIRECTION_CONFIDENCE_THRESHOLD", 0.35))
    
    # Extract signals
    roc = _safe_get(feature_row, "prev_week_gap", 0.0)      # gap up = bullish
    vix_chg = _safe_get(feature_row, "vix_change_1w", 0.0) # rising = bearish
    is_event = int(_safe_get(feature_row, "is_event_week", 0))
    
    # Normalize signals to [-1, +1]
    signals = {
        "roc_score":        np.clip(roc / 0.05, -1, 1),
        "vix_trend_score":  np.clip(-vix_chg / 2.0, -1, 1),
        "garch_acc_score":  np.clip(-vix_chg / 3.0, -1, 1),
    }
    
    # Composite Score
    weights = {"roc_score": 0.45, "vix_trend_score": 0.35, "garch_acc_score": 0.20}
    composite = sum(signals[k] * w for k, w in weights.items())
    
    # Confidence (Event weeks reduce confidence)
    raw_confidence = abs(composite)
    event_penalty = 0.70 if is_event else 1.0
    confidence = raw_confidence * event_penalty
    
    # Direction Classification
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

# ---------------------------------------------------------------------------
# T3 — DTE-Aware Credit Spread Generator
# ---------------------------------------------------------------------------

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
    """Generate bull put or bear call strikes for a given expiry."""
    # Step 1: DTE Scaling Factor
    # Weekly (5 DTE) is baseline
    dte_scalar = np.sqrt(max(dte_days, 1) / 5.0)

    # Step 2: Half-ranges in points
    half_range_p90 = spot * (np.exp(log_range_p90) - 1) / 2

    # Step 3: Load Base Config
    buffer_pts, wing_config, vix_baseline, _, min_buffer_pts, _ = _load_config()

    # Step 4: VIX-Scaled and DTE-Scaled Buffer
    vix_scalar = vix_level / vix_baseline
    base_buffer = np.clip(buffer_pts * vix_scalar, min_buffer_pts, 150)
    scaled_buffer = np.clip(base_buffer * dte_scalar, min_buffer_pts, 300)

    # Step 5: DTE/VIX-Scaled Wing Width
    if vix_level < 13:
        base_wing = wing_config['low']
    elif vix_level < 20:
        base_wing = wing_config['mid']
    else:
        base_wing = wing_config['high']
    
    scaled_wing = base_wing * dte_scalar
    scaled_wing = round_to_strike(scaled_wing, interval=50)
    scaled_wing = max(scaled_wing, 100)

    # Step 6: Strike Placement
    if direction == 'bull_put':
        short_strike = round_to_strike(spot - half_range_p90 - scaled_buffer)
        long_strike = short_strike - scaled_wing
    else:
        short_strike = round_to_strike(spot + half_range_p90 + scaled_buffer)
        long_strike = short_strike + scaled_wing

    # Step 7: Probability of Profit (POP)
    pop_pct = None
    if garch_vol:
        garch_vol_scaled = garch_vol * dte_scalar
        if direction == 'bull_put':
            z = np.log(spot / short_strike) / garch_vol_scaled
            pop_pct = float(norm.cdf(z))
        else:
            z = np.log(short_strike / spot) / garch_vol_scaled
            pop_pct = float(norm.cdf(z))

    # Step 8: Premium and Metrics
    T_years = dte_days / 365.0
    sigma_annual = vix_level / 100.0
    metrics = estimate_spread_premium(
        spot, short_strike, long_strike, T_years, sigma_annual, r, direction
    )

    # Step 9: Return
    return {
        "spot":           round(spot, 2),
        "spread_type":    direction,
        "short_strike":   short_strike,
        "long_strike":    long_strike,
        "wing_width":     int(scaled_wing),
        "scaled_buffer":  round(scaled_buffer, 2),
        "premium_pts":    metrics["premium_pts"],
        "max_loss_pts":   metrics["max_loss_pts"],
        "rr_ratio":       metrics["rr_ratio"],
        "breakeven":      metrics["breakeven"],
        "pop_pct":        round(pop_pct, 4) if pop_pct is not None else None,
        "dte_days":       dte_days,
    }

# ---------------------------------------------------------------------------
# T5 — Multi-Expiry Orchestrator
# ---------------------------------------------------------------------------

def generate_all_spreads(
    feature_row: pd.Series,
    spot: float,
    vix_level: float,
    garch_vol: float | None,
    r: float | None = None
) -> dict:
    """Orchestrate spread generation for 3 expiries."""
    if r is None:
        load_dotenv(dotenv_path=BASE_DIR / ".env")
        r = float(os.getenv("RISK_FREE_RATE", 0.065))
    
    # Step 1: Predict Range
    range_pred = predict_range(feature_row)
    
    # Step 2: Get Expiries
    expiries = get_nse_expiries(date.today())
    
    # Step 3: Get Direction
    direction_result = detect_direction(feature_row)
    direction = direction_result["direction"]
    
    if direction == "bull":
        spread_types = ["bull_put"]
    elif direction == "bear":
        spread_types = ["bear_call"]
    else:
        spread_types = ["bull_put", "bear_call"]
    
    # Step 4: Generate
    all_spreads = []
    for exp_dict in expiries:
        for s_type in spread_types:
            try:
                spread = generate_credit_spread(
                    spot=spot,
                    log_range_p10=range_pred["log_range_p10"],
                    log_range_p90=range_pred["log_range_p90"],
                    dte_days=exp_dict["dte"],
                    vix_level=vix_level,
                    garch_vol=garch_vol,
                    direction=s_type,
                    r=r
                )
                
                spread["expiry_date"] = exp_dict["date"].isoformat()
                spread["expiry_type"] = exp_dict["type"]
                
                # EV Proxy
                pop = spread["pop_pct"] if spread["pop_pct"] is not None else 0.70
                spread["ev_proxy"] = round(spread["rr_ratio"] * pop, 4)
                
                # Min RR check
                min_rr = float(os.getenv("MIN_RR_RATIO", 0.15))
                spread["meets_min_rr"] = spread["rr_ratio"] >= min_rr
                
                all_spreads.append(spread)
            except Exception as e:
                logger.error(f"Failed to generate {s_type} for {exp_dict['date']}: {e}")

    # Step 5: Rank
    all_spreads.sort(key=lambda x: x["ev_proxy"], reverse=True)
    
    # Step 6: Result
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
    
    return result

# ---------------------------------------------------------------------------
# Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock data for testing
    mock_row = pd.Series({
        "prev_week_gap": 0.02,
        "vix_change_1w": -1.5,
        "is_event_week": 0,
        "vix_level": 16.0,
        "garch_sigma_mean": 0.012
    })
    
    # We need real model files for predict_range to work in generate_all_spreads
    # If they don't exist, this will fail gracefully or I can mock predict_range
    try:
        res = generate_all_spreads(mock_row, 24500, 16.0, 0.012)
        print(json.dumps(res, indent=2, default=str))
    except Exception as e:
        print(f"Smoke test failed: {e}")
        # At least test expiry calc
        print("\nTesting Expiry Calculator:")
        print(get_nse_expiries(date.today()))

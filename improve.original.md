"""
Module 9: Credit Spread Generation System
Implements bull put and bear call credit spreads across multiple expiries (weekly, monthly).
Includes Black-Scholes premium estimation and direction signal detection.

Usage:
    from module9_spreads import generate_all_spreads
    results = generate_all_spreads(feature_row, spot, vix_level, garch_vol)
    # Outputs to outputs/spreads_live.json
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

NSE_HOLIDAYS: set[date] = {
    # 2025
    date(2025, 3, 8),   date(2025, 4, 11),  date(2025, 4, 14),
    date(2025, 8, 15),  date(2025, 10, 2),  date(2025, 10, 20),
    date(2025, 10, 21), date(2025, 11, 5),  date(2025, 12, 25),
    # 2026
    date(2026, 1, 1),   date(2026, 3, 26),  date(2026, 4, 3),
    date(2026, 4, 14),  date(2026, 8, 15),  date(2026, 10, 2),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adjust_for_holiday(d: date) -> date:
    """If date is NSE holiday or weekend, move to previous trading day."""
    while d in NSE_HOLIDAYS or d.weekday() >= 5:
        d -= timedelta(days=1)
    return d

def _safe_get(row, col, default):
    """Safely get a value from a pandas Series."""
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return default

NSE_EXPIRY_WEEKDAY = 1  # NSE Nifty 50 options expire on Tuesday (changed from Thursday ~2023)
NIFTY_LOT_SIZE = 65    # NSE Nifty 50 lot size (as of 2024)

def _last_expiry_weekday_of_month(year: int, month: int) -> date:
    """Find the last NSE_EXPIRY_WEEKDAY of a given month."""
    last_day = monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != NSE_EXPIRY_WEEKDAY:
        d -= timedelta(days=1)
    return d

def _is_last_expiry_weekday(d: date) -> bool:
    """Check if d is the last expiry weekday (monthly expiry) of its month."""
    if d.weekday() != NSE_EXPIRY_WEEKDAY:
        return False
    return (d + timedelta(days=7)).month != d.month

# ---------------------------------------------------------------------------
# T1 — NSE Expiry Calculator
# ---------------------------------------------------------------------------

def get_nse_expiries(today: date, nse_expiry_list: list[str] | None = None) -> list[dict]:
    """
    Compute upcoming Nifty options expiry dates (weekly and monthly).

    If nse_expiry_list is provided (from live OI chain, format '28-Apr-2026'),
    uses those exact dates — most accurate, handles any NSE schedule change.
    Otherwise falls back to computed Tuesday-based calendar.

    Returns 3 expiries: nearest, next week, next month's last expiry.
    """
    # --- Path A: use actual NSE expiry dates from OI chain ---
    if nse_expiry_list:
        from datetime import datetime
        parsed = []
        for s in nse_expiry_list:
            try:
                parsed.append(datetime.strptime(s, "%d-%b-%Y").date())
            except ValueError:
                continue

        future = sorted(d for d in parsed if d >= today)
        if len(future) >= 3:
            results = []
            for i, d in enumerate(future[:3]):
                dte = (d - today).days
                results.append({
                    "date": d,
                    "dte": max(dte, 0),
                    "type": "monthly" if _is_last_expiry_weekday(d) else "weekly",
                    "source": "nse_live",
                })
            logger.info(f"Expiries from NSE live chain: {[r['date'].isoformat() for r in results]}")
            return results

    # --- Path B: compute Tuesday-based calendar (fallback) ---
    days_ahead = (NSE_EXPIRY_WEEKDAY - today.weekday()) % 7
    if days_ahead == 0 and today.weekday() == NSE_EXPIRY_WEEKDAY:
        expiry_1 = today
    else:
        expiry_1 = today + timedelta(days=days_ahead)

    expiry_2 = expiry_1 + timedelta(days=7)

    next_month = (expiry_2.month % 12) + 1
    next_year = expiry_2.year + (1 if expiry_2.month == 12 else 0)
    expiry_3 = _last_expiry_weekday_of_month(next_year, next_month)

    expiries = [_adjust_for_holiday(d) for d in [expiry_1, expiry_2, expiry_3]]
    results = []
    for d in expiries:
        dte = (d - today).days
        results.append({
            "date": d,
            "dte": max(dte, 0),
            "type": "monthly" if _is_last_expiry_weekday(d) else "weekly",
            "source": "computed",
        })
    logger.info(f"Expiries computed (fallback): {[r['date'].isoformat() for r in results]}")
    return results

# ---------------------------------------------------------------------------
# T2 — Black-Scholes Premium Estimator
# ---------------------------------------------------------------------------

def estimate_bs_price(
    S: float, K: float, T_years: float, sigma_annual: float,
    r: float = 0.065, option_type: str = 'put',
    q: float = 0.015,    # ADD: Nifty dividend yield
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
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_annual**2) * T_years) / (sigma_annual * np.sqrt(T_years))
    d2 = d1 - sigma_annual * np.sqrt(T_years)
    
    # Black-Scholes pricing
    if option_type == 'call':
        price = S * np.exp(-q * T_years) * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * np.exp(-q * T_years) * norm.cdf(-d1)
    
    return max(price, 0.0)

def estimate_spread_premium(
    S: float,
    short_K: float,
    long_K: float,
    T_years: float,
    sigma_annual: float,
    r: float = 0.065,
    spread_type: str = 'bull_put',
    q: float = 0.015,
) -> dict:
    """Estimate net credit and metrics for a bull put or bear call spread."""
    if spread_type == 'bull_put':
        # Bull put: sell short_K, buy long_K (long_K < short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'put', q)
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'put', q)
        premium_pts = short_price - long_price
        wing_width  = short_K - long_K
    else:  # bear_call
        # Bear call: sell short_K, buy long_K (long_K > short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'call', q)
        long_price  = estimate_bs_price(S, long_K,  T_years, sigma_annual, r, 'call', q)
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
    r: float = 0.065,
    q: float = 0.015,
    atm_iv: float | None = None,
) -> dict:
    """Generate bull put or bear call strikes for a given expiry."""
    # Step 1: DTE Scaling Factor
    # sqrt(DTE/5): wider buffer/wing for longer-dated spreads (more time to breach)
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
    sigma_annual = atm_iv if (atm_iv and atm_iv > 0.05) else vix_level / 100.0
    metrics = estimate_spread_premium(
        spot, short_strike, long_strike, T_years, sigma_annual, r, direction, q
    )

    # Step 9: Return
    return {
        "spot":              round(spot, 2),
        "spread_type":       direction,
        "short_strike":      short_strike,
        "long_strike":       long_strike,
        "wing_width":        int(scaled_wing),
        "scaled_buffer":     round(scaled_buffer, 2),
        "premium_pts":       metrics["premium_pts"],
        "max_loss_pts":      metrics["max_loss_pts"],
        "max_profit_inr":    round(metrics["premium_pts"]  * NIFTY_LOT_SIZE, 2),
        "max_loss_inr":      round(metrics["max_loss_pts"] * NIFTY_LOT_SIZE, 2),
        "rr_ratio":          metrics["rr_ratio"],
        "breakeven":         metrics["breakeven"],
        "pop_pct":           round(pop_pct, 4) if pop_pct is not None else None,
        "dte_days":          dte_days,
        "lot_size":          NIFTY_LOT_SIZE,
    }

# ---------------------------------------------------------------------------
# T5 — Multi-Expiry Orchestrator
# ---------------------------------------------------------------------------

def generate_all_spreads(
    feature_row: pd.Series,
    spot: float,
    vix_level: float,
    garch_vol: float | None,
    r: float | None = None,
    q: float | None = None,
    oi_data: dict | None = None,
) -> dict:
    """Orchestrate spread generation for 3 expiries."""
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", 0.065))
    if q is None:
        q = float(os.getenv("DIVIDEND_YIELD", 0.015))

    # Extract OI-derived inputs (all optional; None = fall back to GARCH/VIX)
    atm_iv         = oi_data.get("atm_iv")        if oi_data else None
    pcr            = oi_data.get("pcr")            if oi_data else None
    oi_strikes     = {int(k): v for k, v in oi_data.get("strikes", {}).items()} if oi_data else {}
    nse_expiry_list = oi_data.get("expiry_dates")  if oi_data else None
    min_oi         = int(os.getenv("MIN_OI_LIQUIDITY", "5000"))
    
    # Step 1: Predict Range
    range_pred = predict_range(feature_row)
    
    # Step 2: Get Expiries — use actual NSE schedule if OI chain available
    expiries = get_nse_expiries(date.today(), nse_expiry_list=nse_expiry_list)
    
    # Step 3: Get Direction
    direction_result = detect_direction(feature_row)

    # Override direction with PCR when market OI gives strong signal
    if pcr is not None:
        if pcr > 1.3:
            _pcr_dir = "bull"
        elif pcr < 0.7:
            _pcr_dir = "bear"
        else:
            _pcr_dir = direction_result.get("direction", "neutral")
        logger.info(f"PCR={pcr:.2f} → direction: {_pcr_dir}")
        direction_result = {**direction_result, "pcr": pcr, "pcr_direction": _pcr_dir}

    # Always generate both types for all expiries to provide a full market view,
    # regardless of the directional "opinion".
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
                    r=r,
                    q=q,
                    atm_iv=atm_iv,
                )
                
                spread["expiry_date"] = exp_dict["date"].isoformat()
                spread["expiry_type"] = exp_dict["type"]
                
                # EV Proxy
                pop = spread["pop_pct"] if spread["pop_pct"] is not None else 0.70
                spread["ev_proxy"] = round(spread["rr_ratio"] * pop, 4)
                
                # Min RR check
                min_rr = float(os.getenv("MIN_RR_RATIO", 0.15))
                spread["meets_min_rr"] = bool(spread["rr_ratio"] >= min_rr)
                
                all_spreads.append(spread)
            except Exception as e:
                logger.error(f"Failed to generate {s_type} for {exp_dict['date']}: {e}")

    # Step 5: Rank
    all_spreads.sort(key=lambda x: x["ev_proxy"], reverse=True)

    # Feasibility filter: drop spreads that don't meet min RR
    before_rr = len(all_spreads)
    all_spreads = [s for s in all_spreads if s["meets_min_rr"]]
    if len(all_spreads) < before_rr:
        logger.info(f"Min RR filter: {before_rr} → {len(all_spreads)} spreads (min_rr={os.getenv('MIN_RR_RATIO', '0.15')})")

    # OI liquidity filter
    if oi_strikes and min_oi > 0:
        def _liquid(strike: float) -> bool:
            k = int(round(strike / 50) * 50)
            row = oi_strikes.get(k, {})
            return (row.get("call_oi", 0) + row.get("put_oi", 0)) >= min_oi

        before = len(all_spreads)
        all_spreads = [s for s in all_spreads
                       if _liquid(s["short_strike"]) and _liquid(s["long_strike"])]
        logger.info(f"OI liquidity filter: {before} → {len(all_spreads)} spreads (min_oi={min_oi})")

    # Step 6: Result
    result = {
        "generated_at":     date.today().isoformat(),
        "spot":             round(spot, 2),
        "vix_level":        round(vix_level, 2),
        "direction_signal": direction_result,
        "expiries":         [e["date"].isoformat() for e in expiries],
        "spreads":          all_spreads,
        "oi_data_used":     oi_data is not None,
        "oi_filtered":      bool(oi_strikes and min_oi > 0),
        "atm_iv_pct":       round(atm_iv * 100, 2) if atm_iv else None,
        "pcr":              pcr,
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

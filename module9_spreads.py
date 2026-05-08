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

from utils_constants import REGIMES
from module6_strikes import round_to_strike, predict_range
from module10_nse_costs import apply_slippage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_nse_holidays() -> set[date]:
    """Load NSE holidays from JSON file; fallback to hardcoded defaults."""
    holidays_path = BASE_DIR / "data" / "nse_holidays.json"
    if holidays_path.exists():
        try:
            with open(holidays_path) as f:
                data = json.load(f)
                return {date.fromisoformat(d) for d in data.get("holidays", [])}
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load NSE holidays from JSON: {e}. Using fallback.")

    # Hardcoded fallback (update data/nse_holidays.json for future years)
    return {
        date(2025, 3, 8),   date(2025, 4, 11),  date(2025, 4, 14),
        date(2025, 8, 15),  date(2025, 10, 2),  date(2025, 10, 20),
        date(2025, 10, 21), date(2025, 11, 5),  date(2025, 12, 25),
        date(2026, 1, 1),   date(2026, 3, 26),  date(2026, 4, 3),
        date(2026, 4, 14),  date(2026, 8, 15),  date(2026, 10, 2),
    }


NSE_HOLIDAYS: set[date] = _load_nse_holidays()

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

def _get_strike_iv(
    oi_strikes: dict,
    strike: float,
    spot: float,
    side: str,
    atm_iv_fallback: float | None = None,
) -> float | None:
    """Look up per-strike IV from chain. Interpolates between neighbours; falls back to atm_iv."""
    iv_key = "put_iv" if side == "put" else "call_iv"
    k = int(round(strike / 50) * 50)

    # Direct lookup
    row = oi_strikes.get(k, {})
    iv = row.get(iv_key, 0.0)
    if iv > 0.01:
        return iv

    # Linear interpolation between two nearest strikes with valid IV
    ks = sorted(oi_strikes.keys())
    lower = [x for x in ks if x < k and oi_strikes[x].get(iv_key, 0.0) > 0.01]
    upper = [x for x in ks if x > k and oi_strikes[x].get(iv_key, 0.0) > 0.01]
    if lower and upper:
        k_lo, k_hi = lower[-1], upper[0]
        iv_lo = oi_strikes[k_lo][iv_key]
        iv_hi = oi_strikes[k_hi][iv_key]
        frac = (k - k_lo) / (k_hi - k_lo)
        return iv_lo + frac * (iv_hi - iv_lo)

    return atm_iv_fallback

NSE_EXPIRY_WEEKDAY = 1  # NSE Nifty 50 options expire on Tuesday (changed from Thursday ~2023)
NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", "65"))

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
    q: float = 0.015,
    vol_skew_factor: float = 0.0,
) -> float:
    """Estimate theoretical option price via Black-Scholes with optional vol skew."""
    if T_years <= 0:
        T_years = 1.0 / 365.0
    if sigma_annual <= 0:
        sigma_annual = 0.10
    # Apply skew: OTM puts/calls trade at higher IV proportional to OTM distance
    if vol_skew_factor > 0:
        if option_type == 'put' and K < S:
            otm_pct = (S - K) / S
            sigma_annual = sigma_annual + vol_skew_factor * otm_pct
        elif option_type == 'call' and K > S:
            otm_pct = (K - S) / S
            sigma_annual = sigma_annual + vol_skew_factor * otm_pct
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
    vol_skew_factor: float = 0.0,
    long_iv_override: float | None = None,
) -> dict:
    """Estimate net credit and metrics for a bull put or bear call spread."""
    long_sigma = long_iv_override if (long_iv_override and long_iv_override > 0.05) else sigma_annual
    if spread_type == 'bull_put':
        # Bull put: sell short_K, buy long_K (long_K < short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'put', q, vol_skew_factor)
        long_price  = estimate_bs_price(S, long_K,  T_years, long_sigma,   r, 'put', q, 0.0)
        premium_pts = short_price - long_price
        wing_width  = short_K - long_K
    else:  # bear_call
        # Bear call: sell short_K, buy long_K (long_K > short_K)
        call_skew_factor = float(os.getenv("CALL_SKEW_FACTOR", "0.01"))
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'call', q, call_skew_factor)
        long_price  = estimate_bs_price(S, long_K,  T_years, long_sigma,   r, 'call', q, 0.0)
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
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
    threshold = float(os.getenv("DIRECTION_CONFIDENCE_THRESHOLD", 0.35))
    
    # Extract signals
    roc = _safe_get(feature_row, "prev_week_gap", 0.0)      # gap up = bullish
    vix_chg = _safe_get(feature_row, "vix_change_1w", 0.0) # rising = bearish
    is_event = int(_safe_get(feature_row, "is_event_week", 0))
    # GARCH acceleration: rising vol = bearish. garch_sigma_mean is current week's
    # mean daily vol; garch_sigma_max is peak daily vol this week (proxy for prior stress).
    # Positive diff = vol receding = less fear = bullish signal.
    garch_cur  = _safe_get(feature_row, "garch_sigma_mean", 0.0)
    garch_prev = _safe_get(feature_row, "garch_sigma_max",  0.0)
    garch_acc  = garch_prev - garch_cur  # positive = vol falling = bullish

    # Normalize signals to [-1, +1]
    signals = {
        "roc_score":        np.clip(roc / 0.05, -1, 1),
        "vix_trend_score":  np.clip(-vix_chg / 2.0, -1, 1),
        "garch_acc_score":  np.clip(garch_acc / 0.005, -1, 1),
    }
    
    # Composite Score
    roc_w = float(os.getenv("WEIGHT_ROC", "0.45"))
    vix_w = float(os.getenv("WEIGHT_VIX", "0.35"))
    garch_w = float(os.getenv("WEIGHT_GARCH", "0.20"))
    
    weights = {"roc_score": roc_w, "vix_trend_score": vix_w, "garch_acc_score": garch_w}
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
    log_range_mu: float | None = None,
    log_range_sigma: float | None = None,
    oi_strikes: dict | None = None,
    pcr: float | None = None,
) -> dict:
    """Generate bull put or bear call strikes for a given expiry."""
    # Step 0: PCR Skew
    put_skew_pts, call_skew_pts = 0, 0
    if pcr is not None:
        from module6_strikes import compute_pcr_skew
        put_skew_pts, call_skew_pts = compute_pcr_skew(pcr)
        if put_skew_pts != 0 or call_skew_pts != 0:
            logger.debug(f"PCR={pcr:.2f} → put_skew={put_skew_pts}, call_skew={call_skew_pts}")

    # Step 1: DTE Scaling Factor
    dte_scalar = np.sqrt(max(dte_days, 1) / 5.0)

    # Step 2: Wing Width (tighter than IC — spread-specific)
    spread_wing_low  = int(os.getenv("SPREAD_WING_WIDTH_LOW_VIX",  "100"))
    spread_wing_mid  = int(os.getenv("SPREAD_WING_WIDTH_MID_VIX",  "150"))
    spread_wing_high = int(os.getenv("SPREAD_WING_WIDTH_HIGH_VIX", "200"))
    if vix_level < 13:
        base_wing = spread_wing_low
    elif vix_level < 20:
        base_wing = spread_wing_mid
    else:
        base_wing = spread_wing_high

    scaled_wing = base_wing * dte_scalar
    scaled_wing = round_to_strike(scaled_wing, interval=50)
    scaled_wing = max(scaled_wing, 50)

    # Step 3: Strike Placement — 25-delta for meaningful premium collection
    # K = S * exp(±z * σ√T)  where z≈0.674 for ~25-delta in lognormal
    sigma = atm_iv if (atm_iv and atm_iv > 0.05) else (garch_vol if garch_vol else vix_level / 100.0)
    T = dte_days / 365.0
    sigma_dte = sigma * np.sqrt(T)
    z_target = float(os.getenv("SPREAD_DELTA_TARGET", "0.674"))

    if direction == 'bull_put':
        short_strike = round_to_strike(spot * np.exp(-z_target * sigma_dte) - put_skew_pts)
        long_strike = short_strike - scaled_wing
    else:
        short_strike = round_to_strike(spot * np.exp(z_target * sigma_dte) + call_skew_pts)
        long_strike = short_strike + scaled_wing

    # Step 7: Probability of Profit (POP)
    # Primary: per-strike chain IV → N(d2) [risk-neutral P(expires OTM)]
    # Secondary: log_range model breach_probability
    # Tertiary: GARCH-based lognormal
    pop_pct = None
    side = "put" if direction == "bull_put" else "call"
    _strikes = oi_strikes or {}

    # Primary — chain IV
    strike_iv = _get_strike_iv(_strikes, short_strike, spot, side, atm_iv_fallback=None)
    if strike_iv and strike_iv > 0.01:
        from module4b_risk import pop_from_chain_iv
        pop_pct = pop_from_chain_iv(short_strike, spot, dte_days, strike_iv, r, q, side)

    # Secondary — log_range model
    if pop_pct is None and log_range_mu is not None and log_range_sigma is not None and log_range_sigma > 0:
        try:
            from module4b_risk import breach_probability
            breach_p = breach_probability(short_strike, log_range_mu, log_range_sigma, spot, side)
            pop_pct = float(1 - breach_p)
        except Exception as e:
            logger.warning(f"breach_probability failed: {e}")

    # Tertiary — GARCH lognormal (unified via pop_from_chain_iv)
    if pop_pct is None and garch_vol:
        from module4b_risk import pop_from_chain_iv
        # Annualize the daily GARCH vol: garch_vol * sqrt(252)
        garch_vol_annual = garch_vol * np.sqrt(252)
        pop_pct = pop_from_chain_iv(short_strike, spot, dte_days, garch_vol_annual, r, q, side)

    # Step 8: Premium — use per-strike IV for short leg; fall back to atm_iv + skew
    T_years = dte_days / 365.0
    # Short-leg IV (chain primary, atm_iv fallback)
    short_iv = _get_strike_iv(_strikes, short_strike, spot, side, atm_iv_fallback=None)
    if not short_iv or short_iv < 0.05:
        short_iv = atm_iv if (atm_iv and atm_iv > 0.05) else vix_level / 100.0

    # Long-leg IV (chain primary, interpolated, atm_iv fallback)
    long_side = side  # same option type for vertical spread
    long_iv = _get_strike_iv(_strikes, long_strike, spot, long_side, atm_iv_fallback=None)
    if not long_iv or long_iv < 0.05:
        # FIX: apply conservative vol skew (deeper OTM = higher IV)
        long_iv = short_iv * 1.05 if short_iv else (atm_iv if atm_iv else vix_level/100.0)

    metrics = estimate_spread_premium(
        spot, short_strike, long_strike, T_years, short_iv, r, direction, q,
        vol_skew_factor=0.0,  # skew already baked into per-strike IV
        long_iv_override=long_iv,
    )

    # Step 9: Return
    return {
        "spot":              round(spot, 2),
        "spread_type":       direction,
        "short_strike":      short_strike,
        "long_strike":       long_strike,
        "wing_width":        int(scaled_wing),
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
    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
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
    
    # Step 3.5: Filter out low-DTE expiries (gamma risk)
    min_dte_to_trade = int(os.getenv("MIN_DTE_TO_TRADE", "7"))
    expiries_filtered = [e for e in expiries if e["dte"] >= min_dte_to_trade]

    # Step 3.6: Augment with calendar fallback if only 1-2 expiries remain (need 3 for 6 spreads)
    if len(expiries_filtered) < 3:
        fallback_expiries = get_nse_expiries(date.today(), nse_expiry_list=None)  # Calendar-based
        fallback_filtered = [e for e in fallback_expiries if e["dte"] >= min_dte_to_trade]
        for fb in fallback_filtered:
            if not any(e["date"] == fb["date"] for e in expiries_filtered):
                expiries_filtered.append(fb)
                logger.info(f"Added fallback expiry {fb['date']} ({fb['dte']}d, {fb['type']})")

    expiries = expiries_filtered
    if not expiries:
        logger.warning(f"No expiries >= {min_dte_to_trade} DTE. Minimum DTE filter too strict.")
        return {"generated_at": date.today().isoformat(), "spreads": [], "summary": {"total_spreads": 0}}

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
                    log_range_mu=range_pred.get("log_range_mu"),
                    log_range_sigma=range_pred.get("log_range_sigma"),
                    oi_strikes=oi_strikes,
                    pcr=pcr,
                )
                
                spread["expiry_date"] = exp_dict["date"].isoformat()
                spread["expiry_type"] = exp_dict["type"]

                # Net premium after slippage (2 legs per credit spread)
                net_premium = apply_slippage(spread["premium_pts"], num_legs=2)
                net_premium = max(net_premium, 0.0)
                spread["net_premium_pts"]  = round(net_premium, 2)
                spread["net_profit_inr"]   = round(net_premium * NIFTY_LOT_SIZE, 2)
                spread["slippage_pts"]     = round(spread["premium_pts"] - net_premium, 2)

                # RR and EV use net premium
                spread["rr_ratio"] = round(
                    net_premium / spread["max_loss_pts"], 4
                ) if spread["max_loss_pts"] > 0 else 0.0

                # EV Proxy using net premium
                if spread["pop_pct"] is None:
                    logger.warning(
                        f"pop_pct unavailable for {spread['spread_type']} "
                        f"{int(spread['short_strike'])}/{int(spread['long_strike'])} "
                        f"DTE={spread['dte_days']} — falling back to pop=0.70 (unreliable EV)"
                    )
                    pop = 0.70
                else:
                    pop = spread["pop_pct"]
                spread["ev_proxy"] = round(
                    net_premium * pop - spread["max_loss_pts"] * (1 - pop), 4
                )
                
                # Min RR check
                min_rr = float(os.getenv("MIN_RR_RATIO", 0.15))
                spread["meets_min_rr"] = bool(spread["rr_ratio"] >= min_rr)
                
                all_spreads.append(spread)
            except Exception as e:
                logger.error(f"Failed to generate {s_type} for {exp_dict['date']}: {e}")

    # Step 5: Rank by EV
    all_spreads.sort(key=lambda x: x["ev_proxy"], reverse=True)

    # Step 5.5: Hard RR and Premium filters
    min_rr = float(os.getenv("MIN_RR_RATIO", "0.15"))
    
    # Regime-specific premium floor
    from module6_strikes import _load_regime_thresholds
    low_thresh, high_thresh = _load_regime_thresholds()
    
    if vix_level < low_thresh:
        min_premium = float(os.getenv("MIN_PREMIUM_LOW_VIX_PTS", "15.0"))
    elif vix_level < high_thresh:
        min_premium = float(os.getenv("MIN_PREMIUM_MID_VIX_PTS", "25.0"))
    else:
        min_premium = float(os.getenv("MIN_PREMIUM_HIGH_VIX_PTS", "40.0"))

    logger.info(f"VIX={vix_level:.1f} → hard filters: RR >= {min_rr}, premium >= {min_premium} pts")

    for spread in all_spreads:
        spread["meets_min_rr"] = bool(spread["rr_ratio"] >= min_rr)

    pre_filter_count = len(all_spreads)
    all_spreads = [
        s for s in all_spreads
        if s["rr_ratio"] >= min_rr and s["net_premium_pts"] >= min_premium
    ]
    dropped = pre_filter_count - len(all_spreads)
    if dropped > 0:
        logger.info(f"Hard filter: dropped {dropped} spreads (RR < {min_rr} or premium < {min_premium} pts)")

    # Step 5.7: OI liquidity filter (skip if OI data is sparse)
    if oi_strikes and min_oi > 0 and len(oi_strikes) >= 200:
        def _liquid(strike: float) -> bool:
            k = int(round(strike / 50) * 50)
            row = oi_strikes.get(k, {})
            return (row.get("call_oi", 0) + row.get("put_oi", 0)) >= min_oi

        before = len(all_spreads)
        all_spreads = [s for s in all_spreads
                       if _liquid(s["short_strike"]) and _liquid(s["long_strike"])]
        logger.info(f"OI liquidity filter: {before} → {len(all_spreads)} spreads (min_oi={min_oi})")
    elif oi_strikes and len(oi_strikes) < 200:
        logger.warning(f"OI data sparse ({len(oi_strikes)} strikes) — skipping OI liquidity filter")

    # Step 5.8: Cap to top N spreads (default 6)
    max_spreads_output = int(os.getenv("MAX_SPREADS_OUTPUT", "6"))
    if len(all_spreads) > max_spreads_output:
        logger.info(f"Limiting output to top {max_spreads_output} spreads by EV")
        all_spreads = all_spreads[:max_spreads_output]

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

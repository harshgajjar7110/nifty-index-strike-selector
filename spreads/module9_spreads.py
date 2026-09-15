"""
Module 9: Credit Spread Generation System
Implements bull put and bear call credit spreads across multiple expiries (weekly, monthly).
Includes Black-Scholes premium estimation and direction signal detection.

Usage:
    from spreads.module9_spreads import generate_all_spreads
    results = generate_all_spreads(feature_row, spot, vix_level, garch_vol)
    # Outputs to outputs/spreads_live.json
"""

import hashlib
import json
from datetime import date, timedelta
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger

from config import REPO_ROOT, cfg
from utils.utils_constants import REGIMES, load_regime_thresholds
from module6_strikes import round_to_strike, predict_range
from spreads.module10_nse_costs import apply_slippage

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = REPO_ROOT
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
    """Look up per-strike IV from chain. Uses stored impliedVolatility primarily;
    falls back to implied_volatility() from market price, then interpolation."""
    from utils.black_scholes import implied_volatility
    iv_key = "put_iv" if side == "put" else "call_iv"
    price_key = "put_price" if side == "put" else "call_price"
    k = int(round(strike / 50) * 50)

    row = oi_strikes.get(k, {})

    # Primary: stored IV from chain (most reliable from jugaad-data)
    iv = row.get(iv_key, 0.0)
    if iv > 0.01:
        return iv

    # Secondary: back out IV from actual market price (when stored IV is 0/missing)
    mkt_price = row.get(price_key, 0.0)
    if mkt_price > 0.5:
        expiry_str = row.get("expiry_date")
        if expiry_str:
            from datetime import date as _date
            exp_d = _date.fromisoformat(expiry_str)
            T = max((exp_d - _date.today()).days, 1) / 365.0
        else:
            T = 11 / 365.0
        computed_iv = implied_volatility(mkt_price, spot, k, T, side, cfg.risk_free_rate, cfg.dividend_yield)
        if computed_iv == computed_iv and computed_iv > 0.01:
            return computed_iv

    # Tertiary: linear interpolation between nearest strikes with valid IV
    ks = sorted(oi_strikes.keys())
    lower = [x for x in ks if x < k and oi_strikes[x].get(iv_key, 0.0) > 0.01]
    upper = [x for x in ks if x > k and oi_strikes[x].get(iv_key, 0.0) > 0.01]
    if lower and upper:
        k_lo, k_hi = lower[-1], upper[0]
        iv_lo = oi_strikes[k_lo][iv_key]
        iv_hi = oi_strikes[k_hi][iv_key]
        frac = (k - k_lo) / (k_hi - k_lo)
        return iv_lo + frac * (iv_hi - iv_lo)

    # Quaternary: interpolate from market prices of neighbours
    price_lower = [x for x in ks if x < k and oi_strikes[x].get(price_key, 0.0) > 0.5]
    price_upper = [x for x in ks if x > k and oi_strikes[x].get(price_key, 0.0) > 0.5]
    if price_lower and price_upper:
        k_lo, k_hi = price_lower[-1], price_upper[0]
        p_lo = oi_strikes[k_lo][price_key]
        p_hi = oi_strikes[k_hi][price_key]
        expiry_str = oi_strikes[k_lo].get("expiry_date")
        if expiry_str:
            from datetime import date as _date
            exp_d = _date.fromisoformat(expiry_str)
            T = max((exp_d - _date.today()).days, 1) / 365.0
        else:
            T = 11 / 365.0
        iv_lo = implied_volatility(p_lo, spot, k_lo, T, side, cfg.risk_free_rate, cfg.dividend_yield)
        iv_hi = implied_volatility(p_hi, spot, k_hi, T, side, cfg.risk_free_rate, cfg.dividend_yield)
        if iv_lo == iv_lo and iv_hi == iv_hi and iv_lo > 0.01 and iv_hi > 0.01:
            frac = (k - k_lo) / (k_hi - k_lo)
            return iv_lo + frac * (iv_hi - iv_lo)

    return atm_iv_fallback

NSE_EXPIRY_WEEKDAY = cfg.nse_expiry_weekday  # NSE Nifty 50 weekly options expiry weekday (0=Mon)
NIFTY_LOT_SIZE = cfg.nifty_lot_size

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

def get_nse_expiries(today: date, nse_expiry_list: list[str] | None = None, max_expiries: int = 8) -> list[dict]:
    """
    Compute upcoming Nifty options expiry dates (weekly and monthly).

    If nse_expiry_list is provided (from live OI chain, format '28-Apr-2026'),
    uses those exact dates — most accurate, handles any NSE schedule change.
    Otherwise falls back to computed Tuesday-based calendar.

    Returns up to max_expiries forward expiries for 21-50 DTE filtering downstream.
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
        if future:
            results = []
            for d in future[:max_expiries]:
                dte = (d - today).days
                results.append({
                    "date": d,
                    "dte": max(dte, 0),
                    "type": "monthly" if _is_last_expiry_weekday(d) else "weekly",
                    "source": "nse_live",
                })
            logger.info(f"Expiries from NSE live chain: {[r['date'].isoformat() for r in results]}")
            return results

    # --- Path B: compute Tuesday-based calendar (fallback, 8 weeklies) ---
    days_ahead = (NSE_EXPIRY_WEEKDAY - today.weekday()) % 7
    if days_ahead == 0 and today.weekday() == NSE_EXPIRY_WEEKDAY:
        expiry_1 = today
    else:
        expiry_1 = today + timedelta(days=days_ahead)

    weekly = [expiry_1 + timedelta(days=7 * i) for i in range(max_expiries)]
    next_month = (weekly[-1].month % 12) + 1
    next_year = weekly[-1].year + (1 if weekly[-1].month == 12 else 0)
    monthly_last = _last_expiry_weekday_of_month(next_year, next_month)
    if monthly_last not in weekly:
        weekly.append(monthly_last)

    expiries = sorted({_adjust_for_holiday(d) for d in weekly})
    results = []
    for d in expiries[:max_expiries]:
        dte = (d - today).days
        results.append({
            "date": d,
            "dte": max(dte, 0),
            "type": "monthly" if _is_last_expiry_weekday(d) else "weekly",
            "source": "computed",
        })
    logger.info(f"Expiries computed (fallback): {[r['date'].isoformat() for r in results]}")
    return results


def filter_expiries_to_tenor(
    expiries: list[dict],
    min_dte: int | None = None,
    max_dte: int | None = None,
    target_dte: int | None = None,
    blend_enabled: bool | None = None,
) -> list[dict]:
    """Filter expiries to [min_dte, max_dte]; blend tenors if band is empty.

    Blend logic: when no single expiry sits in-band and blending is enabled,
    return the nearest straddling pair (one below, one above) each tagged
    ``blended=True, size_mult=0.5`` so two half-size spreads synthesize
    ~target_dte exposure. Falls back to closest single expiry if only one
    side exists. All in-band expiries get ``blended=False, size_mult=1.0``.
    """
    if min_dte is None:
        min_dte = cfg.min_dte_to_trade
    if max_dte is None:
        max_dte = cfg.max_dte_to_trade
    if target_dte is None:
        target_dte = cfg.target_dte
    if blend_enabled is None:
        blend_enabled = cfg.dte_blend_enabled

    in_band = [dict(e, blended=False, size_mult=1.0) for e in expiries if min_dte <= e["dte"] <= max_dte]
    if in_band:
        logger.info(f"Tenor filter [{min_dte},{max_dte}] DTE: {len(in_band)}/{len(expiries)} kept")
        return sorted(in_band, key=lambda e: abs(e["dte"] - target_dte))

    if not blend_enabled or not expiries:
        logger.warning(f"No expiries in [{min_dte},{max_dte}] DTE and blending disabled.")
        return []

    below = [e for e in expiries if e["dte"] < min_dte]
    above = [e for e in expiries if e["dte"] > max_dte]
    if below and above:
        lo = max(below, key=lambda e: e["dte"])
        hi = min(above, key=lambda e: e["dte"])
        pair = [dict(lo, blended=True, size_mult=0.5), dict(hi, blended=True, size_mult=0.5)]
        logger.info(f"Blended tenor: {lo['date']} ({lo['dte']}d) + {hi['date']} ({hi['dte']}d) -> ~{target_dte}d target")
        return pair
    closest = min(expiries, key=lambda e: abs(e["dte"] - target_dte))
    logger.warning(f"Single-sided blend fallback: {closest['date']} ({closest['dte']}d)")
    return [dict(closest, blended=True, size_mult=0.5)]

# ---------------------------------------------------------------------------
# T2 — Black-Scholes Premium Estimator
# ---------------------------------------------------------------------------

def estimate_bs_price(
    S: float, K: float, T_years: float, sigma_annual: float,
    r: float | None = None, option_type: str = 'put',
    q: float | None = None,
    vol_skew_factor: float = 0.0,
) -> float:
    """Estimate theoretical option price via Black-Scholes with optional vol skew."""
    if r is None:
        r = cfg.risk_free_rate
    if q is None:
        q = cfg.dividend_yield
    from utils.black_scholes import bs_price_with_skew
    return bs_price_with_skew(S, K, T_years, sigma_annual, option_type, r, q, vol_skew_factor)

def estimate_spread_premium(
    S: float,
    short_K: float,
    long_K: float,
    T_years: float,
    sigma_annual: float,
    r: float | None = None,
    spread_type: str = 'bull_put',
    q: float | None = None,
    vol_skew_factor: float = 0.0,
    long_iv_override: float | None = None,
) -> dict:
    """Estimate net credit and metrics for a bull put or bear call spread."""
    if r is None:
        r = cfg.risk_free_rate
    if q is None:
        q = cfg.dividend_yield
    long_sigma = long_iv_override if (long_iv_override and long_iv_override > 0.05) else sigma_annual
    if spread_type == 'bull_put':
        # Bull put: sell short_K, buy long_K (long_K < short_K)
        short_price = estimate_bs_price(S, short_K, T_years, sigma_annual, r, 'put', q, vol_skew_factor)
        long_price  = estimate_bs_price(S, long_K,  T_years, long_sigma,   r, 'put', q, 0.0)
        premium_pts = short_price - long_price
        wing_width  = short_K - long_K
    else:  # bear_call
        # Bear call: sell short_K, buy long_K (long_K > short_K)
        call_skew_factor = cfg.call_skew_factor
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
    Determine bullish/bearish/neutral direction based on trend + VIX regime signals.
    All inputs are lagged weekly features (no lookahead).
    """
    from utils.utils_constants import assign_regime
    threshold = cfg.direction_confidence_threshold

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
    ret_4w = _safe_get(feature_row, "return_4w", 0.0)
    trend_strength = _safe_get(feature_row, "trend_strength_proxy", 0.5)
    close_pos = _safe_get(feature_row, "close_position_in_range", 0.5)

    # Normalize signals to [-1, +1]
    signals = {
        "roc_score":        np.clip(roc / 0.05, -1, 1),
        "vix_trend_score":  np.clip(-vix_chg / 2.0, -1, 1),
        "garch_acc_score":  np.clip(garch_acc / 0.005, -1, 1),
        "trend_4w_score":   np.clip(ret_4w / 0.08, -1, 1),
        "trend_strength_score": np.clip((trend_strength - 0.5) * 2, -1, 1) * np.sign(ret_4w if ret_4w != 0 else roc if roc != 0 else 1),
        "close_pos_score":  np.clip((close_pos - 0.5) * 2, -1, 1),
    }

    # Composite Score
    weights = {
        "roc_score": cfg.weight_roc,
        "vix_trend_score": cfg.weight_vix,
        "garch_acc_score": cfg.weight_garch,
        "trend_4w_score": cfg.weight_trend_4w,
        "trend_strength_score": cfg.weight_trend_strength,
        "close_pos_score": 0.0,
    }
    w_sum = sum(weights.values()) or 1.0
    composite = float(sum(signals[k] * w for k, w in weights.items()) / w_sum)
    
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

    try:
        vix_regime = assign_regime(float(_safe_get(feature_row, "vix_level", 16.0)))
    except Exception:
        vix_regime = "mid"

    return {
        "direction":      direction,
        "confidence":     round(float(confidence), 4),
        "composite":      round(float(composite), 4),
        "signals":        {k: round(float(v), 4) for k, v in signals.items()},
        "is_event_week":  bool(is_event),
        "threshold":      threshold,
        "vix_regime":     vix_regime,
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
    r: float | None = None,
    q: float | None = None,
    atm_iv: float | None = None,
    log_range_mu: float | None = None,
    log_range_sigma: float | None = None,
    oi_strikes: dict | None = None,
    pcr: float | None = None,
    is_neutral: bool = False,
) -> dict:
    """Generate bull put or bear call strikes for a given expiry."""
    if r is None:
        r = cfg.risk_free_rate
    if q is None:
        q = cfg.dividend_yield
    # Step 0: PCR Skew
    put_skew_pts, call_skew_pts = 0, 0
    if pcr is not None:
        from module6_strikes import compute_pcr_skew
        put_skew_pts, call_skew_pts = compute_pcr_skew(pcr)
        if put_skew_pts != 0 or call_skew_pts != 0:
            logger.debug(f"PCR={pcr:.2f} → put_skew={put_skew_pts}, call_skew={call_skew_pts}")

    # Step 1: DTE Scaling Factor (normalized to TARGET_DTE for 21-50DTE regime)
    dte_scalar = np.sqrt(max(dte_days, 1) / max(cfg.target_dte, 1))

    # Step 2: Wing Width (tighter than IC — spread-specific)
    spread_wing_low  = cfg.spread_wing_width_low_vix
    spread_wing_mid  = cfg.spread_wing_width_mid_vix
    spread_wing_high = cfg.spread_wing_width_high_vix
    low_thresh, high_thresh = load_regime_thresholds()
    if vix_level < low_thresh:
        base_wing = spread_wing_low
    elif vix_level < high_thresh:
        base_wing = spread_wing_mid
    else:
        base_wing = spread_wing_high

    scaled_wing = base_wing * dte_scalar
    scaled_wing = round_to_strike(scaled_wing, interval=50)
    scaled_wing = max(scaled_wing, 50)

    # Step 3: Strike Placement — regime-aware delta
    # K = S * exp(±z * σ√T); z≈0.674 trending (~25Δ), z≈0.45 neutral (safer, more OTM)
    sigma = atm_iv if (atm_iv and atm_iv > 0.05) else (garch_vol if garch_vol else vix_level / 100.0)
    T = dte_days / 365.0
    sigma_dte = sigma * np.sqrt(T)
    z_target = cfg.spread_delta_target_neutral if is_neutral else cfg.spread_delta_target

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
        from spreads.module4b_risk import pop_from_chain_iv
        pop_pct = pop_from_chain_iv(short_strike, spot, dte_days, strike_iv, r, q, side)

    # Secondary — log_range model
    if pop_pct is None and log_range_mu is not None and log_range_sigma is not None and log_range_sigma > 0:
        try:
            from spreads.module4b_risk import breach_probability
            breach_p = breach_probability(short_strike, log_range_mu, log_range_sigma, spot, side)
            pop_pct = float(1 - breach_p)
        except Exception as e:
            logger.warning(f"breach_probability failed: {e}")

    # Tertiary — GARCH lognormal (unified via pop_from_chain_iv)
    if pop_pct is None and garch_vol:
        from spreads.module4b_risk import pop_from_chain_iv
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
# Risk-floor version stamp + empty-result helper
# ---------------------------------------------------------------------------

def _risk_floor_stamp() -> dict:
    """Snapshot the active POP/breach/EV floors plus a config hash.

    The hash lets downstream consumers detect when a stored JSON was produced
    under different floors than the current ``.env``/``config.py``.
    """
    payload = {
        "min_pop": cfg.ic_min_pop,
        "max_breach_per_leg": cfg.ic_max_breach_prob_per_leg,
        "min_ev_proxy_pts": cfg.min_ev_proxy_pts,
        "min_rr": cfg.min_rr_ratio,
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return {**payload, "config_hash": digest}


def _empty_spreads_result(reason: str) -> dict:
    """Stamped empty result for weeks with no viable trade (never emits spreads)."""
    stamp = _risk_floor_stamp()
    logger.warning(f"No viable trade this week: {reason}")
    return {
        "generated_at": date.today().isoformat(),
        "spreads": [],
        "no_trade_reason": reason,
        "summary": {
            "total_spreads": 0,
            "top_pick": None,
            "risk_floors": {**stamp, "floors_overridden": False},
        },
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
    allow_below_floor_override: bool = False,
) -> dict:
    """Orchestrate spread generation for 3 expiries.

    Risk floors (POP/breach/EV) always apply unless ``allow_below_floor_override``
    is explicitly set, in which case the override is stamped in the output.
    """
    if r is None:
        r = cfg.risk_free_rate
    if q is None:
        q = cfg.dividend_yield

    # Extract OI-derived inputs (all optional; None = fall back to GARCH/VIX)
    atm_iv         = oi_data.get("atm_iv")        if oi_data else None
    pcr            = oi_data.get("pcr")            if oi_data else None
    oi_strikes     = {int(k): v for k, v in oi_data.get("strikes", {}).items()} if oi_data else {}
    nse_expiry_list = oi_data.get("expiry_dates")  if oi_data else None
    min_oi         = cfg.min_oi_liquidity
    
    # Step 1: Predict Range
    range_pred = predict_range(feature_row)
    
    # Step 2: Get Expiries — use actual NSE schedule if OI chain available
    expiries = get_nse_expiries(date.today(), nse_expiry_list=nse_expiry_list)
    
    # Step 3: Get Direction
    direction_result = detect_direction(feature_row)

    # Override direction with PCR when market OI gives strong signal
    if pcr is not None:
        if pcr > cfg.pcr_bull_threshold:
            _pcr_dir = "bull"
        elif pcr < cfg.pcr_bear_threshold:
            _pcr_dir = "bear"
        else:
            _pcr_dir = direction_result.get("direction", "neutral")
        logger.info(f"PCR={pcr:.2f} → direction: {_pcr_dir}")
        direction_result = {**direction_result, "pcr": pcr, "pcr_direction": _pcr_dir}

    # Directional single-side selection: follow trend (model + strong PCR).
    # Bull -> bull_put only, bear -> bear_call only, neutral -> both small.
    effective_dir = direction_result.get("direction", "neutral")
    if pcr is not None:
        if pcr > cfg.pcr_bull_threshold:
            effective_dir = "bull"
        elif pcr < cfg.pcr_bear_threshold:
            effective_dir = "bear"
    direction_result = {**direction_result, "effective_direction": effective_dir}

    if effective_dir == "bull":
        spread_types = ["bull_put"]
    elif effective_dir == "bear":
        spread_types = ["bear_call"]
    else:
        spread_types = ["bull_put", "bear_call"]
    logger.info(f"Directional selection: trend={effective_dir} -> {spread_types}")

    # Step 3.5: 21-50 DTE tenor filter with blend fallback
    expiries = filter_expiries_to_tenor(expiries)
    if not expiries:
        return _empty_spreads_result("no expiries in 21-50 DTE band and blending disabled")

    trend_dir = effective_dir
    is_neutral = trend_dir == "neutral"
    boost = cfg.trend_boost_ev
    neutral_mult = cfg.neutral_size_mult

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
                    is_neutral=is_neutral,
                )

                spread["expiry_date"] = exp_dict["date"].isoformat()
                spread["expiry_type"] = exp_dict["type"]
                spread["blended"] = bool(exp_dict.get("blended", False))
                spread["tenor_size_mult"] = float(exp_dict.get("size_mult", 1.0))

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
                spread["breach_prob_per_leg"] = round(1.0 - pop, 4)

                # Trend-regime EV boost (both sides kept, aligned side ranked higher)
                aligned = (trend_dir == "bull" and s_type == "bull_put") or (trend_dir == "bear" and s_type == "bear_call")
                spread["trend_aligned"] = bool(aligned)
                spread["ev_boosted"] = round(spread["ev_proxy"] * (1 + boost) if aligned and not is_neutral else spread["ev_proxy"], 4)

                # Neutral weeks: small size on both legs (synthetic condor)
                size_mult = float(exp_dict.get("size_mult", 1.0))
                if is_neutral:
                    size_mult *= neutral_mult
                spread["size_mult"] = round(size_mult, 3)
                spread["is_neutral"] = bool(is_neutral)

                # Min RR check
                min_rr = cfg.min_rr_ratio
                spread["meets_min_rr"] = bool(spread["rr_ratio"] >= min_rr)

                all_spreads.append(spread)
            except Exception as e:
                logger.error(f"Failed to generate {s_type} for {exp_dict['date']}: {e}")

    # Step 5: Rank by trend-boosted EV
    all_spreads.sort(key=lambda x: x["ev_boosted"], reverse=True)

    # Step 5.5: Hard RR and Premium filters
    min_rr = cfg.min_rr_ratio
    
    # Regime-specific premium floor
    low_thresh, high_thresh = load_regime_thresholds()
    
    if vix_level < low_thresh:
        min_premium = cfg.min_premium_low_vix_pts
    elif vix_level < high_thresh:
        min_premium = cfg.min_premium_mid_vix_pts
    else:
        min_premium = cfg.min_premium_high_vix_pts

    logger.info(f"VIX={vix_level:.1f} → hard filters: RR >= {min_rr}, premium >= {min_premium} pts")

    n_candidates = len(all_spreads)
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

    # Step 5.6: Risk floor filters (POP and breach probability per leg).
    # No spread below the configured floors is emitted unless the caller sets
    # allow_below_floor_override explicitly (stamped + logged loudly below).
    stamp = _risk_floor_stamp()
    min_pop = stamp["min_pop"]
    max_breach_per_leg = stamp["max_breach_per_leg"]
    min_ev = stamp["min_ev_proxy_pts"]
    floors_overridden = bool(allow_below_floor_override)

    logger.info(f"Risk floors: POP >= {min_pop:.0%}, breach/leg <= {max_breach_per_leg:.0%}, EV >= {min_ev:.1f} pts")

    if floors_overridden:
        logger.warning(
            "Risk floors OVERRIDDEN by explicit flag: emitting spreads below "
            f"POP >= {min_pop:.0%} / breach/leg <= {max_breach_per_leg:.0%} / EV >= {min_ev:.1f} pts"
        )
    else:
        pre_filter_count = len(all_spreads)
        all_spreads = [
            s for s in all_spreads
            if s.get("pop_pct", 0) >= min_pop
               and s.get("breach_prob_per_leg", 1) <= max_breach_per_leg
               and s["ev_proxy"] >= min_ev
        ]
        dropped = pre_filter_count - len(all_spreads)
        if dropped > 0:
            logger.info(f"Risk filter: dropped {dropped} spreads (POP < {min_pop:.0%} or breach/leg > {max_breach_per_leg:.0%} or EV < {min_ev:.1f} pts)")

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
    max_spreads_output = cfg.max_spreads_output
    if len(all_spreads) > max_spreads_output:
        logger.info(f"Limiting output to top {max_spreads_output} spreads by EV")
        all_spreads = all_spreads[:max_spreads_output]

    # Step 6: Result
    if all_spreads:
        top = all_spreads[0]
        logger.info(
            f"TOP PICK: {top['spread_type']} {int(top['short_strike'])}/{int(top['long_strike'])} "
            f"DTE={top['dte_days']} exp={top['expiry_date']} "
            f"premium={top['net_premium_pts']:.1f} max_loss={top['max_loss_pts']:.1f} "
            f"pop={top['pop_pct']:.1%} ev_proxy={top['ev_proxy']:.2f}pts"
        )
        neg_ev = [s for s in all_spreads if s["ev_proxy"] < 0]
        if neg_ev:
            logger.warning(
                f"{len(neg_ev)}/{len(all_spreads)} surviving spreads have negative ev_proxy. "
                f"Consider raising min_ev_proxy_pts (currently {cfg.min_ev_proxy_pts}) or skipping this week."
            )

    if cfg.wf_soft_vix_lower <= vix_level <= cfg.wf_max_vix_trade:
        logger.warning(
            f"Soft-VIX band [{cfg.wf_soft_vix_lower:.1f}, {cfg.wf_max_vix_trade:.1f}] "
            f"active (vix={vix_level:.1f}): reduce size to {cfg.wf_soft_size_mult:.2f}x for this week."
        )

    if all_spreads:
        no_trade_reason = None
    else:
        no_trade_reason = (
            f"all {n_candidates} candidate spreads dropped by filters "
            f"(RR >= {min_rr}, premium >= {min_premium} pts, POP >= {min_pop:.0%}, "
            f"breach/leg <= {max_breach_per_leg:.0%}, EV >= {min_ev:.1f} pts, OI liquidity)"
        )
        logger.warning(f"No viable trade this week: {no_trade_reason}")

    result = {
        "generated_at":     date.today().isoformat(),
        "spot":             round(spot, 2),
        "vix_level":        round(vix_level, 2),
        "direction_signal": direction_result,
        "expiries":         [e["date"].isoformat() for e in expiries],
        "tenor_config": {
            "min_dte": cfg.min_dte_to_trade,
            "max_dte": cfg.max_dte_to_trade,
            "target_dte": cfg.target_dte,
            "blend_enabled": cfg.dte_blend_enabled,
        },
        "spreads":          all_spreads,
        "no_trade_reason":  no_trade_reason,
        "oi_data_used":     oi_data is not None,
        "oi_filtered":      bool(oi_strikes and min_oi > 0),
        "atm_iv_pct":       round(atm_iv * 100, 2) if atm_iv else None,
        "pcr":              pcr,
        "summary": {
            "total_spreads": len(all_spreads),
            "top_pick":      all_spreads[0] if all_spreads else None,
            "min_ev_proxy_pts": cfg.min_ev_proxy_pts,
            "risk_floors": {**stamp, "floors_overridden": floors_overridden},
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
        "garch_sigma_mean": 0.012,
        "garch_sigma_max": 0.014,
        "return_4w": 0.03,
        "trend_strength_proxy": 0.6,
        "close_position_in_range": 0.7,
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

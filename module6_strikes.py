"""
Module 6: Strike Generation
Convert P10/P90 log-range predictions into Nifty 50 iron condor strike levels.
"""

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from scipy.stats import norm

from config import cfg
from utils_constants import REGIMES, load_regime_thresholds, extract_vix

try:
    from module4b_risk import breach_probability
except ImportError:
    breach_probability = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_models_cached() -> tuple[dict, list, dict, object]:
    """Load per-regime LightGBM models, MAPIE models, and feature columns from disk."""
    meta_path = MODELS_DIR / "regime_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run module4_model.py first")

    with open(meta_path) as f:
        regime_meta = json.load(f)

    lgb_models = {}
    for regime in REGIMES:
        if regime in regime_meta:
            model_file = MODELS_DIR / regime_meta[regime]["model_file"]
            if (MODELS_DIR / model_file.name).exists():
                lgb_models[regime] = joblib.load(model_file)

    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

    # Load MAPIE models
    mapie_per_regime = {}
    for regime in REGIMES:
        regime_path = MODELS_DIR / f"mapie_{regime}.pkl"
        if regime_path.exists():
            mapie_per_regime[regime] = joblib.load(regime_path)

    # Fallback global MAPIE
    mapie_global = None
    global_path = MODELS_DIR / "mapie_calibrated.pkl"
    if global_path.exists():
        mapie_global = joblib.load(global_path)

    logger.info("Regime LightGBM and MAPIE models loaded.")
    return lgb_models, feature_columns, mapie_per_regime, mapie_global


def _clear_model_cache() -> None:
    """Invalidate the model cache."""
    _load_models_cached.cache_clear()


def _load_models(force_reload: bool = False) -> tuple[dict, list, dict, object]:
    """Load models with optional cache invalidation."""
    if force_reload:
        _clear_model_cache()
    return _load_models_cached()


@lru_cache(maxsize=1)
def _get_vix_baseline() -> float:
    """Compute VIX baseline from .env or historical data."""
    vix_baseline = None
    vix_baseline_env = cfg.vix_baseline if hasattr(cfg, "vix_baseline") else None
    if vix_baseline_env:
        try:
            vix_baseline = float(vix_baseline_env)
            logger.info(f"VIX_BASELINE loaded from .env: {vix_baseline:.2f}")
        except ValueError:
            logger.warning(f"Invalid VIX_BASELINE in .env: {vix_baseline_env}, will compute from data")

    if vix_baseline is None:
        vix_path = BASE_DIR / "data" / "india_vix_daily.parquet"
        if vix_path.exists():
            try:
                vix_df = pd.read_parquet(vix_path)
                close_col = None
                for col in ("close", "Close", "CLOSE", "vix", "VIX"):
                    if col in vix_df.columns:
                        close_col = col
                        break
                if close_col is None and len(vix_df.columns) > 0:
                    close_col = vix_df.columns[0]
                if close_col is not None:
                    vix_series = pd.to_numeric(vix_df[close_col], errors="coerce")
                    vix_baseline = float(vix_series.tail(252).mean())
                    logger.info(f"VIX_BASELINE computed from historical data: {vix_baseline:.2f}")
                else:
                    raise ValueError("No VIX column found in parquet")
            except Exception as e:
                logger.warning(f"Could not load VIX baseline from parquet: {e}, using default 16.0")
                vix_baseline = 16.0
        else:
            logger.warning(f"VIX data file not found at {vix_path}, using default 16.0")
            vix_baseline = 16.0

    if vix_baseline <= 0:
        logger.warning("VIX baseline is <= 0; using default 16.0")
        vix_baseline = 16.0

    return vix_baseline


def round_to_strike(price: float, interval: int = 50) -> int:
    """Round price to nearest Nifty strike interval."""
    return round(price / interval) * interval


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_range(feature_row: pd.Series) -> dict:
    """
    Predict P10 and P90 log-range + mu/sigma for a single feature row.
    Uses per-regime MAPIE for calibrated interval prediction.

    Returns dict with keys: log_range_p10, log_range_p90, log_range_mu, log_range_sigma, regime
    """
    lgb_models, feature_columns, mapie_per_regime, mapie_global = _load_models()

    X = feature_row[feature_columns].values.reshape(1, -1)
    vix = float(feature_row["vix_level"])
    low_thresh, high_thresh = load_regime_thresholds()
    regime = "low" if vix < low_thresh else ("mid" if vix < high_thresh else "high")

    # Determine which MAPIE model to use
    mapie_model = mapie_per_regime.get(regime, mapie_global)

    if mapie_model is not None:
        # Use MAPIE for calibrated intervals
        _, y_pis = mapie_model.predict_interval(X)
        if len(y_pis.shape) == 3:
            log_range_p10 = float(y_pis[0, 0, 0])
            log_range_p90 = float(y_pis[0, 1, 0])
        else:
            log_range_p10 = float(y_pis[0, 0])
            log_range_p90 = float(y_pis[0, 1])
        
        # Compute mu/sigma from P10 and P90 assuming Normal distribution
        log_range_mu = (log_range_p10 + log_range_p90) / 2.0
        log_range_sigma = (log_range_p90 - log_range_p10) / (2 * norm.ppf(0.90))
    elif regime in lgb_models:
        # Fallback to raw LightGBM if MAPIE missing
        models = lgb_models[regime]
        log_range_p10 = float(models["p10"].predict(X)[0])
        log_range_p90 = float(models["p90"].predict(X)[0])
        log_range_mu = (log_range_p10 + log_range_p90) / 2.0
        log_range_sigma = (log_range_p90 - log_range_p10) / (2 * norm.ppf(0.90))
    else:
        logger.warning(f"No model for regime {regime}, using fallback percentiles")
        log_range_p10 = cfg.fallback_log_range_p10
        log_range_p90 = cfg.fallback_log_range_p90
        log_range_mu = cfg.fallback_log_range_mu
        log_range_sigma = cfg.fallback_log_range_sigma

    logger.info(f"Predicted log_range: p10={log_range_p10:.4f}, p90={log_range_p90:.4f}, mu={log_range_mu:.4f}, sigma={log_range_sigma:.4f}, regime={regime}")
    return {
        "log_range_p10": log_range_p10,
        "log_range_p90": log_range_p90,
        "log_range_mu": log_range_mu,
        "log_range_sigma": log_range_sigma,
        "regime": regime,
    }


def compute_pcr_skew(pcr: float | None) -> tuple[int, int]:
    """Returns (put_skew_pts, call_skew_pts) based on PCR.
    Negative skew -> tighten (strike moves closer to spot - more premium).
    """
    if pcr is None:
        return 0, 0
    pcr_bear = cfg.pcr_bear_threshold
    pcr_bull = cfg.pcr_bull_threshold
    put_tighten = cfg.pcr_put_tighten_pts
    call_tighten = cfg.pcr_call_tighten_pts

    if pcr < pcr_bear:
        # Bearish market: put heavy buying -> tighten puts (negative = closer to spot)
        return -put_tighten, 0
    elif pcr > pcr_bull:
        # Bullish market: call heavy -> tighten calls
        return 0, -call_tighten
    return 0, 0


def generate_strikes(
    current_close: float,
    log_range_p10: float,
    log_range_p90: float,
    vix_level: float | None = None,
    put_skew_pts: int = 0,
    call_skew_pts: int = 0,
    garch_vol_weekly: float | None = None,
    atm_iv: float | None = None,
    max_pain: float | None = None,
    log_range_mu: float | None = None,
    log_range_sigma: float | None = None,
    pcr: float | None = None,
) -> dict:
    """
    Convert log-range predictions and current spot into iron condor strikes.
    Buffer is dynamically scaled based on VIX level.
    Put side can be widened (OTM) via put_skew_pts to reflect market skew.
    Call side can be widened (OTM) via call_skew_pts to reflect asymmetric skew.
    """
    if pcr is not None and put_skew_pts == 0 and call_skew_pts == 0:
        put_skew_pts, call_skew_pts = compute_pcr_skew(pcr)
        logger.debug(f"PCR={pcr:.2f} -> put_skew={put_skew_pts}, call_skew={call_skew_pts}")

    buffer_pts = cfg.strike_buffer_points
    wing_width_config = cfg.wing_width_by_regime
    vix_baseline = _get_vix_baseline()
    loaded_put_skew = cfg.put_skew_points
    min_buffer_pts = cfg.min_buffer_points
    loaded_call_skew = cfg.call_skew_points

    # Validate min_buffer_pts is in reasonable range
    if min_buffer_pts < 0 or min_buffer_pts > 150:
        logger.warning(f"MIN_BUFFER_POINTS {min_buffer_pts} out of range [0,150]; using 75")
        min_buffer_pts = 75

    # Use parameter if explicitly provided and non-zero, else use loaded config
    if put_skew_pts == 0:
        put_skew_pts = loaded_put_skew
    if call_skew_pts == 0:
        call_skew_pts = loaded_call_skew

    # Fetch live VIX if not provided
    if vix_level is None:
        try:
            ticker = yf.Ticker("^INDIAVIX")
            vix_level = float(ticker.fast_info["last_price"])
            logger.info(f"Live India VIX fetched: {vix_level:.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch live VIX: {e}, using baseline {vix_baseline:.2f}")
            vix_level = vix_baseline

    # Select wing width based on VIX regime
    low_thresh, high_thresh = load_regime_thresholds()
    if vix_level < low_thresh:
        wing_width = wing_width_config["low"]
    elif vix_level < high_thresh:
        wing_width = wing_width_config["mid"]
    else:
        wing_width = wing_width_config["high"]
    logger.info(f"Dynamic wing width: {wing_width} pts (VIX={vix_level:.1f})")

    # Compute VIX scalar and effective buffer
    vix_scalar = vix_level / vix_baseline
    effective_buffer = np.clip(buffer_pts * vix_scalar, min_buffer_pts, 150)

    # Calm-market tighten: if VIX low AND GARCH vol low, reduce buffer by 15%
    if (
        vix_level is not None
        and vix_level < 12
        and garch_vol_weekly is not None
        and garch_vol_weekly < 0.020
    ):
        effective_buffer = max(effective_buffer * 0.85, min_buffer_pts)
        logger.info(
            f"Calm-market modifier applied (VIX={vix_level:.1f}, "
            f"garch={garch_vol_weekly:.4f}): effective_buffer -> {effective_buffer:.0f} pts"
        )

    # Log if minimum buffer floor was applied
    if effective_buffer == min_buffer_pts:
        logger.info(f"Minimum buffer floor applied: {min_buffer_pts} pts (VIX scalar {vix_scalar:.2f} < 1.0)")

    logger.info(
        f"VIX: {vix_level:.1f} (baseline: {vix_baseline:.1f}), "
        f"scalar: {vix_scalar:.2f}, effective_buffer: {effective_buffer:.0f} pts"
    )

    # log_range is ln(High/Low). Typical weekly range is ~2.5%.
    # half_range is roughly the distance from center to H or L.
    range_pts_p10 = current_close * (np.exp(log_range_p10) - 1)
    range_pts_p90 = current_close * (np.exp(log_range_p90) - 1)
    
    # Use a blended range: 70% P90 + 30% P10 for a robust 'half-range'
    blended_half_range = (0.70 * range_pts_p90 + 0.30 * range_pts_p10) / 2.0

    lower_price = current_close - blended_half_range - effective_buffer - put_skew_pts
    upper_price = current_close + blended_half_range + effective_buffer + call_skew_pts

    short_put = round_to_strike(lower_price)
    short_call = round_to_strike(upper_price)

    # Blend strikes 30% toward max pain
    if max_pain is not None:
        _MP_BLEND = 0.30
        short_put  = short_put  + _MP_BLEND * (max_pain - short_put)
        short_call = short_call + _MP_BLEND * (max_pain - short_call)
        short_put  = round_to_strike(short_put)
        short_call = round_to_strike(short_call)
        logger.info(f"Max pain blend applied: max_pain={max_pain}, short_put->{short_put}, short_call->{short_call}")

    long_put = short_put - wing_width
    long_call = short_call + wing_width

    # Compute breach probabilities
    breach_prob_call = None
    breach_prob_put = None
    prob_of_profit = None

    if breach_probability and log_range_mu is not None and log_range_sigma is not None and log_range_sigma > 0:
        breach_prob_call = breach_probability(short_call, log_range_mu, log_range_sigma, current_close, "call")
        breach_prob_put = breach_probability(short_put, log_range_mu, log_range_sigma, current_close, "put")
        prob_of_profit = float(1 - breach_prob_call - breach_prob_put)
        logger.info(
            f"POP: {prob_of_profit:.1%}, breach_call: {breach_prob_call:.1%}, "
            f"breach_put: {breach_prob_put:.1%}  [src=dist_model]"
        )
    else:
        # Fallback: lognormal model using GARCH/ATM IV
        weekly_vol_garch = garch_vol_weekly * np.sqrt(5) if garch_vol_weekly else None
        weekly_vol_iv    = atm_iv * np.sqrt(5 / 252) if (atm_iv and atm_iv > 0) else None
        vol_for_pop = weekly_vol_iv if weekly_vol_iv else weekly_vol_garch
        
        if vol_for_pop is not None and vol_for_pop > 0:
            z_call = np.log(short_call / current_close) / vol_for_pop
            z_put  = np.log(current_close / short_put)  / vol_for_pop
            breach_prob_call = float(1 - norm.cdf(z_call))
            breach_prob_put  = float(norm.cdf(-z_put))
            prob_of_profit   = float(1 - breach_prob_call - breach_prob_put)
            vol_src = "market_iv" if (atm_iv and atm_iv > 0) else "garch"
            logger.info(
                f"POP: {prob_of_profit:.1%}, breach_call: {breach_prob_call:.1%}, "
                f"breach_put: {breach_prob_put:.1%}  [vol_src={vol_src}_weekly]"
            )

    result = {
        "current_close": current_close,
        "short_put": short_put,
        "short_call": short_call,
        "long_put": long_put,
        "long_call": long_call,
        "predicted_range_p10": round(range_pts_p10 * 2, 2),
        "predicted_range_p90": round(range_pts_p90 * 2, 2),
        "buffer_pts": buffer_pts,
        "wing_width_pts": wing_width,
        "vix_level": vix_level,
        "vix_baseline": vix_baseline,
        "vix_scalar": vix_scalar,
        "effective_buffer_pts": effective_buffer,
        "put_skew_pts": put_skew_pts,
        "call_skew_pts": call_skew_pts,
        "breach_prob_call": breach_prob_call,
        "breach_prob_put": breach_prob_put,
        "prob_of_profit": prob_of_profit,
        "atm_iv_used":   round(atm_iv * 100, 2) if atm_iv else None,
        "max_pain_used": int(max_pain) if max_pain else None,
    }

    if put_skew_pts > 0:
        logger.info(f"Put skew applied: {put_skew_pts} pts (put {put_skew_pts} OTM vs call)")
    if call_skew_pts > 0:
        logger.info(f"Call skew applied: {call_skew_pts} pts (call {call_skew_pts} OTM vs put)")
    if put_skew_pts == 0 and call_skew_pts == 0:
        logger.debug("No skew applied (both skews == 0)")

    logger.info(
        f"Strikes: long_put={long_put}, short_put={short_put}, "
        f"short_call={short_call}, long_call={long_call}"
    )
    return result


def fetch_live_spot() -> float:
    """
    Fetch current Nifty 50 spot price via yfinance (no API key required).

    Returns
    -------
    float - last traded price of Nifty 50
    """
    ticker = yf.Ticker("^NSEI")
    spot = ticker.fast_info["last_price"]
    logger.info(f"Live Nifty spot (yfinance): {spot:.2f}")
    return float(spot)


def run_live_prediction(feature_row: pd.Series) -> dict:
    """
    End-to-end live prediction: predict range, fetch spot, generate strikes,
    save to outputs/strikes_YYYY-MM-DD.json.
    """
    range_pred = predict_range(feature_row)
    spot = fetch_live_spot()

    # Extract VIX from feature_row if available
    vix_level = extract_vix(feature_row)

    strikes = generate_strikes(
        current_close=spot,
        log_range_p10=range_pred["log_range_p10"],
        log_range_p90=range_pred["log_range_p90"],
        vix_level=vix_level,
    )

    out_path = OUTPUTS_DIR / f"strikes_{date.today().isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(strikes, f, indent=2)
    logger.info(f"Strikes saved to {out_path}")

    return strikes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("This module is called by module8_live.py. For testing, run module8_live.py.")

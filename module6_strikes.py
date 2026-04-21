"""
Module 6: Strike Generation
Convert P10/P90 log-range predictions into Nifty 50 iron condor strike levels.
"""

import json
import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loguru import logger
from scipy.stats import norm

# load_dotenv only needed for strike config (buffer/wing points), not for credentials

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

_MODEL_CACHE: dict = {}

def _load_models():
    """Load P10/P90 models and feature columns. Raises FileNotFoundError if missing."""
    if _MODEL_CACHE:
        return _MODEL_CACHE["p10"], _MODEL_CACHE["p90"], _MODEL_CACHE["cols"]

    p10_path = MODELS_DIR / "lgbm_p10.pkl"
    p90_path = MODELS_DIR / "lgbm_p90.pkl"
    feat_path = MODELS_DIR / "feature_columns.pkl"

    for path in (p10_path, p90_path, feat_path):
        if not path.exists():
            raise FileNotFoundError("Run module4_model.py first")

    lgbm_p10 = joblib.load(p10_path)
    lgbm_p90 = joblib.load(p90_path)
    feature_columns = joblib.load(feat_path)
    logger.info("Models and feature columns loaded.")
    _MODEL_CACHE["p10"] = lgbm_p10
    _MODEL_CACHE["p90"] = lgbm_p90
    _MODEL_CACHE["cols"] = feature_columns
    return lgbm_p10, lgbm_p90, feature_columns


_CONFIG_CACHE: dict = {}

def _load_config():
    """Load strike config from .env, including VIX baseline, put skew, call skew, and minimum buffer.

    Returns 6-tuple: (buffer_pts, wing_dict, vix_baseline, put_skew_pts, min_buffer_pts, call_skew_pts)
    Note: As of Task 5, wing_dict is {"low": 150, "mid": 200, "high": 250}.
    """
    if _CONFIG_CACHE:
        return _CONFIG_CACHE["buffer"], _CONFIG_CACHE["wing"], _CONFIG_CACHE["vix_baseline"], _CONFIG_CACHE["put_skew"], _CONFIG_CACHE["min_buffer"], _CONFIG_CACHE["call_skew"]

    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    buffer_pts = int(os.getenv("STRIKE_BUFFER_POINTS", 50))

    # Check if new regime-specific vars are set (they should all be set together for consistency)
    wing_low_env = os.getenv("WING_WIDTH_LOW_VIX")
    wing_mid_env = os.getenv("WING_WIDTH_MID_VIX")
    wing_high_env = os.getenv("WING_WIDTH_HIGH_VIX")

    # If any new regime var is set, use them; else check for legacy WING_WIDTH_POINTS
    if wing_low_env is not None or wing_mid_env is not None or wing_high_env is not None:
        # New regime-specific config
        wing_pts_low  = int(wing_low_env or 150)
        wing_pts_mid  = int(wing_mid_env or 200)
        wing_pts_high = int(wing_high_env or 250)
        logger.info("Using dynamic wing width by VIX regime")
    else:
        # Legacy single-value config
        wing_pts_legacy = int(os.getenv("WING_WIDTH_POINTS", 200))
        wing_pts_low = wing_pts_mid = wing_pts_high = wing_pts_legacy
        logger.info(f"Using legacy fixed wing width: {wing_pts_legacy} pts (set WING_WIDTH_LOW/MID/HIGH_VIX for dynamic)")

    put_skew_pts = int(os.getenv("PUT_SKEW_POINTS", 0))
    call_skew_pts = int(os.getenv("CALL_SKEW_POINTS", 0))
    min_buffer_pts = int(os.getenv("MIN_BUFFER_POINTS", 75))

    # Validate min_buffer_pts is in reasonable range
    if min_buffer_pts < 0 or min_buffer_pts > 150:
        logger.warning(f"MIN_BUFFER_POINTS {min_buffer_pts} out of range [0,150]; using 75")
        min_buffer_pts = 75

    # Try to load VIX_BASELINE from .env, otherwise compute from historical data
    vix_baseline = None
    vix_baseline_env = os.getenv("VIX_BASELINE", None)
    if vix_baseline_env:
        try:
            vix_baseline = float(vix_baseline_env)
            logger.info(f"VIX_BASELINE loaded from .env: {vix_baseline:.2f}")
        except ValueError:
            logger.warning(f"Invalid VIX_BASELINE in .env: {vix_baseline_env}, will compute from data")

    if vix_baseline is None:
        # Compute 52-week mean from historical VIX data
        vix_path = BASE_DIR / "data" / "india_vix_daily.parquet"
        if vix_path.exists():
            try:
                vix_df = pd.read_parquet(vix_path)
                # Try common column names
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

    # Guard against zero or negative VIX baseline (prevents division by zero)
    if vix_baseline <= 0:
        logger.warning("VIX baseline is <= 0; using default 16.0")
        vix_baseline = 16.0

    _CONFIG_CACHE["buffer"] = buffer_pts
    _CONFIG_CACHE["wing"] = {"low": wing_pts_low, "mid": wing_pts_mid, "high": wing_pts_high}
    _CONFIG_CACHE["vix_baseline"] = vix_baseline
    _CONFIG_CACHE["put_skew"] = put_skew_pts
    _CONFIG_CACHE["min_buffer"] = min_buffer_pts
    _CONFIG_CACHE["call_skew"] = call_skew_pts
    return buffer_pts, _CONFIG_CACHE["wing"], vix_baseline, put_skew_pts, min_buffer_pts, call_skew_pts


def round_to_strike(price: float, interval: int = 50) -> int:
    """Round price to nearest Nifty strike interval."""
    return round(price / interval) * interval


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_range(feature_row: pd.Series) -> dict:
    """
    Predict P10 and P90 log-range for a single feature row.

    Parameters
    ----------
    feature_row : pd.Series
        Features matching the training feature set.

    Returns
    -------
    dict with keys: log_range_p10, log_range_p90
    """
    model_p10, model_p90, feature_columns = _load_models()

    X = feature_row[feature_columns].values.reshape(1, -1)
    log_range_p10 = float(model_p10.predict(X)[0])
    log_range_p90 = float(model_p90.predict(X)[0])

    logger.info(f"Predicted log_range_p10={log_range_p10:.4f}, log_range_p90={log_range_p90:.4f}")
    return {"log_range_p10": log_range_p10, "log_range_p90": log_range_p90}


def generate_strikes(
    current_close: float,
    log_range_p10: float,
    log_range_p90: float,
    vix_level: float | None = None,
    put_skew_pts: int = 0,
    call_skew_pts: int = 0,
    garch_vol_weekly: float | None = None,
) -> dict:
    """
    Convert log-range predictions and current spot into iron condor strikes.
    Buffer is dynamically scaled based on VIX level.
    Put side can be widened (OTM) via put_skew_pts to reflect market skew.
    Call side can be widened (OTM) via call_skew_pts to reflect asymmetric skew.

    Parameters
    ----------
    current_close : float
        Current Nifty spot price.
    log_range_p10 : float
        Predicted P10 of log(H/L).
    log_range_p90 : float
        Predicted P90 of log(H/L).
    vix_level : float | None
        Current India VIX level. If None, fetches live via yfinance.
    put_skew_pts : int
        Extra OTM points for put strike. If 0 (default), uses value
        from .env (PUT_SKEW_POINTS). If > 0, overrides config.
    call_skew_pts : int
        Extra OTM points for call strike. If 0 (default), uses value
        from .env (CALL_SKEW_POINTS). If > 0, overrides config.
    garch_vol_weekly : float | None
        GARCH conditional volatility (weekly). If provided, computes
        breach probabilities via lognormal model.

    Returns
    -------
    dict with strike levels and metadata.
    """
    buffer_pts, wing_width_config, vix_baseline, loaded_put_skew, min_buffer_pts, loaded_call_skew = _load_config()
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
    if vix_level < 13:
        wing_width = wing_width_config["low"]
    elif vix_level < 20:
        wing_width = wing_width_config["mid"]
    else:
        wing_width = wing_width_config["high"]
    logger.info(f"Dynamic wing width: {wing_width} pts (VIX={vix_level:.1f})")

    # Compute VIX scalar and effective buffer
    vix_scalar = vix_level / vix_baseline
    effective_buffer = np.clip(buffer_pts * vix_scalar, min_buffer_pts, 150)

    # Calm-market tighten: if VIX low AND GARCH vol low, reduce buffer by 15%
    # but always respect the min_buffer floor.
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

    # Absolute half-ranges
    half_range_p10 = current_close * (np.exp(log_range_p10) - 1) / 2
    half_range_p90 = current_close * (np.exp(log_range_p90) - 1) / 2

    # Use P90 (wider) range for strike placement with VIX-scaled buffer
    lower_price = current_close - half_range_p90 - effective_buffer - put_skew_pts
    upper_price = current_close + half_range_p90 + effective_buffer + call_skew_pts

    short_put = round_to_strike(lower_price)
    short_call = round_to_strike(upper_price)

    long_put = short_put - wing_width
    long_call = short_call + wing_width

    # Compute breach probabilities using lognormal model
    breach_prob_call = None
    breach_prob_put = None
    prob_of_profit = None

    if garch_vol_weekly is not None and garch_vol_weekly > 0:
        z_call = np.log(short_call / current_close) / garch_vol_weekly
        z_put = np.log(current_close / short_put) / garch_vol_weekly
        breach_prob_call = float(1 - norm.cdf(z_call))
        breach_prob_put = float(norm.cdf(-z_put))
        prob_of_profit = float(1 - breach_prob_call - breach_prob_put)
        logger.info(
            f"POP: {prob_of_profit:.1%}, breach_call: {breach_prob_call:.1%}, "
            f"breach_put: {breach_prob_put:.1%}"
        )

    result = {
        "current_close": current_close,
        "short_put": short_put,
        "short_call": short_call,
        "long_put": long_put,
        "long_call": long_call,
        "predicted_range_p10": round(half_range_p10 * 2, 2),
        "predicted_range_p90": round(half_range_p90 * 2, 2),
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
    float — last traded price of Nifty 50
    """
    import yfinance as yf
    ticker = yf.Ticker("^NSEI")
    spot = ticker.fast_info["last_price"]
    logger.info(f"Live Nifty spot (yfinance): {spot:.2f}")
    return float(spot)


def run_live_prediction(feature_row: pd.Series) -> dict:
    """
    End-to-end live prediction: predict range, fetch spot, generate strikes,
    save to outputs/strikes_YYYY-MM-DD.json.

    Parameters
    ----------
    feature_row : pd.Series
        Features matching the training feature set.

    Returns
    -------
    dict — strikes result
    """
    range_pred = predict_range(feature_row)
    spot = fetch_live_spot()

    # Extract VIX from feature_row if available
    vix_level = None
    for col in ("vix_level", "vix", "india_vix", "VIX", "INDIA_VIX"):
        if col in feature_row.index:
            try:
                vix_level = float(feature_row[col])
                if vix_level > 0:
                    break
            except (ValueError, TypeError):
                pass

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

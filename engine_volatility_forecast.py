"""
Volatility Forecast Engine
===========================

Multi-model volatility forecasting for options strategies:
  Primary: GARCH(1,1) — from existing module3_garch
  Secondary: EWMA (Exponential Weighted Moving Average)
  Fallback: Historical Volatility

Outputs:
  - Forecast volatility by horizon (weekly, swing, monthly)
  - Volatility regime classification
  - Vol term structure (vol changes over forecast horizon)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from loguru import logger
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VOL_FORECAST_PATH = DATA_DIR / "vol_forecast_latest.json"

# Volatility forecast horizons
FORECAST_HORIZONS = {
    "weekly": 7,      # 1 week
    "swing": 30,      # ~1 month
    "monthly": 60,    # ~2 months
}

# Volatility regime thresholds
VOL_REGIMES = {
    "low": (0.0, 0.15),
    "medium": (0.15, 0.25),
    "high": (0.25, 0.40),
    "extreme": (0.40, 1.0),
}


# ---------------------------------------------------------------------------
# GARCH(1,1) Forecasting (leverage existing Module 3)
# ---------------------------------------------------------------------------

def forecast_volatility_garch(
    returns: pd.Series,
    periods_ahead: int = 7,
    garch_params: dict | None = None,
) -> np.ndarray:
    """
    Forecast volatility using GARCH(1,1) model.
    
    Args:
        returns: Daily log returns (annualized vol = returns.std() * sqrt(252))
        periods_ahead: Number of periods to forecast
        garch_params: Optional dict with p, q, params. If None, fit fresh.
        
    Returns:
        Array of forecast volatilities (annualized) for each period
    """
    logger.info(f"GARCH(1,1) forecast: {periods_ahead} periods ahead")
    
    if len(returns) < 30:
        logger.warning(f"Not enough returns ({len(returns)}) for GARCH fitting. Using historical vol.")
        hist_vol = returns.std() * np.sqrt(252)
        return np.array([hist_vol] * periods_ahead)
    
    try:
        # Fit GARCH(1,1)
        model = arch_model(returns.dropna() * 100, vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        
        # Forecast conditional volatility
        forecast = res.forecast(horizon=periods_ahead)
        cond_var = forecast.variance.values[-1, :]  # Get most recent forecast row
        cond_vol = np.sqrt(cond_var) / 100  # Convert back to decimal (daily vol)
        
        # Annualize
        ann_vol = cond_vol * np.sqrt(252)
        
        logger.success(f"GARCH forecast: {ann_vol[0]:.4f} (next period)")
        
        return ann_vol
    
    except Exception as e:
        logger.warning(f"GARCH fitting failed: {e}. Falling back to historical vol.")
        hist_vol = returns.std() * np.sqrt(252)
        return np.array([hist_vol] * periods_ahead)


# ---------------------------------------------------------------------------
# EWMA Forecasting
# ---------------------------------------------------------------------------

def forecast_volatility_ewma(
    returns: pd.Series,
    periods_ahead: int = 7,
    lambda_param: float = 0.94,
) -> np.ndarray:
    """
    Forecast volatility using Exponential Weighted Moving Average (RiskMetrics).
    
    EWMA assumes conditional vol mean-reverts to long-term vol at exponential rate.
    
    Args:
        returns: Daily log returns
        periods_ahead: Number of periods to forecast
        lambda_param: Decay factor (0.94 = standard RiskMetrics)
        
    Returns:
        Array of forecast volatilities (annualized)
    """
    logger.info(f"EWMA forecast: {periods_ahead} periods ahead (lambda={lambda_param})")
    
    if len(returns) < 10:
        hist_vol = returns.std() * np.sqrt(252)
        return np.array([hist_vol] * periods_ahead)
    
    # Current conditional vol (from EWMA)
    returns_clean = returns.dropna()
    sq_returns = returns_clean ** 2
    
    # EWMA of squared returns
    ewma_var = sq_returns.ewm(span=int(1 / (1 - lambda_param))).mean().iloc[-1]
    current_vol_daily = np.sqrt(ewma_var)
    current_vol_ann = current_vol_daily * np.sqrt(252)
    
    # Long-term vol (historical)
    lt_vol_ann = returns_clean.std() * np.sqrt(252)
    
    # Mean reversion rate
    mean_reversion_speed = 1 - lambda_param
    
    # Forecast: exponential approach from current to long-term
    forecasts = []
    current = current_vol_ann
    for t in range(periods_ahead):
        current = current * (1 - mean_reversion_speed) + lt_vol_ann * mean_reversion_speed
        forecasts.append(current)
    
    logger.success(f"EWMA forecast: {forecasts[0]:.4f} (next period), LT: {lt_vol_ann:.4f}")
    
    return np.array(forecasts)


# ---------------------------------------------------------------------------
# Historical Volatility (Fallback)
# ---------------------------------------------------------------------------

def historical_volatility(
    returns: pd.Series,
    periods_ahead: int = 7,
    window: int = 30,
) -> np.ndarray:
    """
    Fallback: Use historical rolling volatility as constant forecast.
    
    Args:
        returns: Daily log returns
        periods_ahead: Number of periods to forecast
        window: Rolling window (default 30 days)
        
    Returns:
        Array with same value repeated periods_ahead times
    """
    if len(returns) < window:
        window = len(returns)
    
    hist_vol = returns.iloc[-window:].std() * np.sqrt(252)
    logger.info(f"Historical volatility: {hist_vol:.4f}")
    
    return np.array([hist_vol] * periods_ahead)


# ---------------------------------------------------------------------------
# Multi-Model Ensemble
# ---------------------------------------------------------------------------

def forecast_volatility_ensemble(
    returns: pd.Series,
    periods_ahead: int = 7,
    weights: dict | None = None,
) -> dict:
    """
    Ensemble forecast combining GARCH, EWMA, and historical volatility.
    
    Args:
        returns: Daily log returns
        periods_ahead: Number of periods
        weights: dict with keys 'garch', 'ewma', 'hist'. Must sum to 1.
                 Default: {'garch': 0.5, 'ewma': 0.3, 'hist': 0.2}
        
    Returns:
        dict with forecasts from each model + ensemble
    """
    if weights is None:
        weights = {"garch": 0.5, "ewma": 0.3, "hist": 0.2}
    
    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    logger.info(f"Ensemble forecast (periods={periods_ahead})")
    
    # Get individual forecasts
    try:
        garch_vol = forecast_volatility_garch(returns, periods_ahead)
    except Exception as e:
        logger.warning(f"GARCH failed: {e}")
        garch_vol = None
    
    try:
        ewma_vol = forecast_volatility_ewma(returns, periods_ahead)
    except Exception as e:
        logger.warning(f"EWMA failed: {e}")
        ewma_vol = None
    
    hist_vol = historical_volatility(returns, periods_ahead)
    
    # Combine available models
    forecasts = {}
    if garch_vol is not None:
        forecasts["garch"] = garch_vol
    if ewma_vol is not None:
        forecasts["ewma"] = ewma_vol
    forecasts["hist"] = hist_vol
    
    # Weighted average
    ensemble = np.zeros(periods_ahead)
    active_weight = 0
    
    if "garch" in forecasts:
        ensemble += weights["garch"] * forecasts["garch"]
        active_weight += weights["garch"]
    
    if "ewma" in forecasts:
        ensemble += weights["ewma"] * forecasts["ewma"]
        active_weight += weights["ewma"]
    
    ensemble += weights["hist"] * forecasts["hist"]
    active_weight += weights["hist"]
    
    # Re-normalize to account for missing models
    if active_weight > 0:
        ensemble /= active_weight
    
    forecasts["ensemble"] = ensemble
    
    logger.success(
        f"Ensemble vol (next period): {ensemble[0]:.4f} "
        f"(GARCH: {garch_vol[0] if garch_vol is not None else 'N/A'}, "
        f"EWMA: {ewma_vol[0] if ewma_vol is not None else 'N/A'}, "
        f"Hist: {hist_vol[0]:.4f})"
    )
    
    return forecasts


# ---------------------------------------------------------------------------
# Volatility Regime Classification
# ---------------------------------------------------------------------------

def classify_vol_regime(vol: float) -> str:
    """
    Classify annualized volatility into regime buckets.
    
    Args:
        vol: Annualized volatility (e.g., 0.18 = 18%)
        
    Returns:
        Regime name ('low', 'medium', 'high', 'extreme')
    """
    for regime, (lower, upper) in VOL_REGIMES.items():
        if lower <= vol < upper:
            return regime
    return "extreme" if vol >= 0.40 else "low"


# ---------------------------------------------------------------------------
# Term Structure of Volatility
# ---------------------------------------------------------------------------

def estimate_vol_term_structure(
    forecasts: dict,
    horizons_days: list | None = None,
) -> pd.DataFrame:
    """
    Build term structure of volatility (vol smile across time horizons).
    
    Args:
        forecasts: Dict with 'ensemble' key containing array of forecast vols
        horizons_days: List of day numbers corresponding to forecasts
        
    Returns:
        DataFrame with DTE and corresponding volatility
    """
    if horizons_days is None:
        horizons_days = [1, 7, 14, 30, 60]
    
    if "ensemble" not in forecasts:
        raise ValueError("forecasts must contain 'ensemble' key")
    
    ensemble = forecasts["ensemble"]
    
    # Interpolate/extrapolate to match horizons
    term_structure = []
    for i, dte in enumerate(horizons_days):
        if i < len(ensemble):
            term_structure.append(ensemble[i])
        else:
            # Assume final forecast vol persists
            term_structure.append(ensemble[-1])
    
    df = pd.DataFrame({
        "dte": horizons_days,
        "forecast_vol": term_structure,
        "regime": [classify_vol_regime(v) for v in term_structure],
    })
    
    return df


# ---------------------------------------------------------------------------
# Main Interface: Generate Forecast Report
# ---------------------------------------------------------------------------

def generate_forecast_report(
    daily_ohlcv: pd.DataFrame,
    lookback_days: int = 252,
    ensemble_weights: dict | None = None,
) -> dict:
    """
    Comprehensive volatility forecast for current market.
    
    Args:
        daily_ohlcv: DataFrame with 'close' column (indexed by date)
        lookback_days: Historical window for vol calculation
        ensemble_weights: Weights for GARCH/EWMA/Hist
        
    Returns:
        dict with all forecast data for downstream engines
    """
    logger.info(f"Generating volatility forecast report (lookback={lookback_days} days)")
    
    # Compute returns
    prices = daily_ohlcv["close"]
    returns = np.log(prices / prices.shift(1))
    
    # Use recent data
    if len(returns) > lookback_days:
        returns = returns.iloc[-lookback_days:]
    
    # Generate ensemble forecasts
    forecasts = {
        "weekly": forecast_volatility_ensemble(
            returns, periods_ahead=7, weights=ensemble_weights
        ),
        "swing": forecast_volatility_ensemble(
            returns, periods_ahead=30, weights=ensemble_weights
        ),
        "monthly": forecast_volatility_ensemble(
            returns, periods_ahead=60, weights=ensemble_weights
        ),
    }
    
    # Build term structure
    term_structure = estimate_vol_term_structure(
        forecasts["monthly"],
        horizons_days=[7, 14, 30, 45, 60],
    )
    
    # Current volatility regime
    current_vol = returns.std() * np.sqrt(252)
    current_regime = classify_vol_regime(current_vol)
    
    # Prepare report
    report = {
        "generated_at": datetime.now().isoformat(),
        "current_vol": float(current_vol),
        "current_regime": current_regime,
        "forecast_weekly": {
            "ensemble": forecasts["weekly"]["ensemble"].tolist(),
            "regime": classify_vol_regime(forecasts["weekly"]["ensemble"][0]),
        },
        "forecast_swing": {
            "ensemble": forecasts["swing"]["ensemble"].tolist(),
            "regime": classify_vol_regime(forecasts["swing"]["ensemble"][0]),
        },
        "forecast_monthly": {
            "ensemble": forecasts["monthly"]["ensemble"].tolist(),
            "regime": classify_vol_regime(forecasts["monthly"]["ensemble"][0]),
        },
        "term_structure": term_structure.to_dict(orient="records"),
    }
    
    logger.success(
        f"Forecast report ready. Current vol: {current_vol:.4f} ({current_regime}), "
        f"Weekly forecast: {forecasts['weekly']['ensemble'][0]:.4f}"
    )
    
    return report


def save_forecast_report(report: dict) -> None:
    """Persist forecast report to JSON."""
    with open(VOL_FORECAST_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Forecast report saved → {VOL_FORECAST_PATH}")


def load_forecast_report() -> dict:
    """Load most recent forecast report."""
    if not VOL_FORECAST_PATH.exists():
        logger.warning(f"No forecast report found at {VOL_FORECAST_PATH}")
        return {}
    
    with open(VOL_FORECAST_PATH, "r") as f:
        report = json.load(f)
    
    logger.info(f"Forecast report loaded from {VOL_FORECAST_PATH}")
    return report

"""
Shared constants and utilities for Nifty 50 Iron Condor pipeline.
Centralizes regime definitions, thresholds, and common helper functions.
"""

import json
from pathlib import Path
from typing import Tuple

import pandas as pd

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# Regime Constants
# ─────────────────────────────────────────────────────────────────────────────

REGIMES = ["low", "mid", "high"]
DEFAULT_REGIME_LOW_THRESH = 15.0
DEFAULT_REGIME_HIGH_THRESH = 20.0


def load_regime_thresholds() -> Tuple[float, float]:
    """Load optimized VIX regime thresholds from models/; fall back to defaults (15, 20)."""
    thresh_path = BASE_DIR / "models" / "regime_thresholds.json"
    if thresh_path.exists():
        with open(thresh_path) as f:
            t = json.load(f)
        return t["low_thresh"], t["high_thresh"]
    return DEFAULT_REGIME_LOW_THRESH, DEFAULT_REGIME_HIGH_THRESH


def assign_regime(vix: float, low_thresh: float = None, high_thresh: float = None) -> str:
    """Assign VIX regime: low / mid / high.

    Parameters
    ----------
    vix : float
        VIX level
    low_thresh : float, optional
        Lower threshold (default: 15.0)
    high_thresh : float, optional
        Upper threshold (default: 20.0)

    Returns
    -------
    str
        Regime name: 'low', 'mid', or 'high'
    """
    if low_thresh is None or high_thresh is None:
        low_thresh, high_thresh = load_regime_thresholds()

    if vix < low_thresh:
        return "low"
    if vix < high_thresh:
        return "mid"
    return "high"


# ─────────────────────────────────────────────────────────────────────────────
# Volatility Utilities
# ─────────────────────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_WEEK = 5


def annualize_vol(daily_vol: float, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Convert daily volatility to annualized."""
    import numpy as np
    return daily_vol * np.sqrt(periods_per_year)


def weekly_from_daily(daily_vol: float) -> float:
    """Convert daily volatility to weekly."""
    import numpy as np
    return daily_vol * np.sqrt(TRADING_DAYS_PER_WEEK)


def extract_vix(feature_row: pd.Series, default: float | None = None) -> float | None:
    """Extract VIX level from a feature row using known column aliases.

    Parameters
    ----------
    feature_row : pd.Series
        A pandas Series (e.g. a feature matrix row) that may contain VIX data.
    default : float | None, optional
        Value to return if no valid VIX column is found (default: None).

    Returns
    -------
    float | None
        The first valid positive VIX level found, or *default* if none found.
    """
    for col in ("vix_level", "vix", "india_vix", "VIX", "INDIA_VIX"):
        if col in feature_row.index:
            try:
                val = float(feature_row[col])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return default

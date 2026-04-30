"""
Shared constants and utilities for Nifty 50 Iron Condor pipeline.
Centralizes regime definitions, thresholds, and common helper functions.
"""

import json
from pathlib import Path
from typing import Tuple

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

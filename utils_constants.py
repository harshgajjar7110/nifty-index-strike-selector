"""
Shared constants and utilities for Nifty 50 Iron Condor pipeline.
Centralizes regime definitions, thresholds, and common helper functions.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
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


def assign_regime_series(vix_series: pd.Series, low_thresh: float, high_thresh: float) -> pd.Series:
    return pd.cut(
        vix_series,
        bins=[-float('inf'), low_thresh, high_thresh, float('inf')],
        labels=['low', 'mid', 'high'],
    )


def find_column(df: pd.DataFrame, names: tuple[str, ...], default_col: int = -1) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    if len(df.columns) > 0:
        return df.columns[default_col]
    return None


def chronological_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'log_range',
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.sort_index()
    n = len(df)
    split_idx = int(n * train_frac)

    if target_col in df.columns:
        y = df[target_col].values
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y[:split_idx] if y is not None else None
    y_test = y[split_idx:] if y is not None else None

    return X_train, X_test, y_train, y_test

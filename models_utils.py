"""
Shared model utilities for Nifty 50 Iron Condor pipeline.
Centralizes MAPIE interval handling, regime-based prediction wrapper,
and conformal coverage utilities.
"""

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import norm

from config import cfg


def extract_pis(y_pis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract lower and upper prediction interval bounds from MAPIE output.

    Handles both 3D shape (n_samples, 2, 1) and 2D shape (n_samples, 2).

    Parameters
    ----------
    y_pis : np.ndarray
        Prediction intervals from MAPIE SplitConformalRegressor.predict_interval()

    Returns
    -------
    tuple of (lower_bound, upper_bound) as 1D numpy arrays
    """
    if y_pis.ndim == 3:
        y_low = y_pis[:, 0, 0]
        y_high = y_pis[:, 1, 0]
    else:
        y_low = y_pis[:, 0]
        y_high = y_pis[:, 1]
    return y_low, y_high


def compute_breach_probability(
    log_range_p10: float,
    log_range_p90: float,
    short_put_log: float,
    short_call_log: float,
) -> dict:
    """Compute Gaussian breach probabilities and POP from quantile predictions.

    Uses the log-range Normal assumption:
        mu = (p10 + p90) / 2
        sigma = (p90 - p10) / (2 * z_alpha_p90)

    Parameters
    ----------
    log_range_p10 : float
        Predicted 10th percentile of weekly log-range
    log_range_p90 : float
        Predicted 90th percentile of weekly log-range
    short_put_log : float
        Log-price of short put strike relative to spot
    short_call_log : float
        Log-price of short call strike relative to spot

    Returns
    -------
    dict with keys: breach_prob_put, breach_prob_call, POP, mu, sigma
    """
    mu = (log_range_p10 + log_range_p90) / 2.0

    z_090 = norm.ppf(0.90)
    sigma = (log_range_p90 - log_range_p10) / (2 * z_090)

    if sigma <= 0:
        sigma = max(abs(log_range_p90 - log_range_p10) / 2, 0.001)

    breach_prob_put = norm.cdf((np.log(short_put_log) - mu) / sigma) if short_put_log > 0 else 0.0
    breach_prob_call = 1 - norm.cdf((np.log(short_call_log) - mu) / sigma) if short_call_log > 0 else 0.0

    pop = max(0.0, 1.0 - breach_prob_put - breach_prob_call)

    return {
        "breach_prob_put": float(breach_prob_put),
        "breach_prob_call": float(breach_prob_call),
        "POP": float(pop),
        "mu": float(mu),
        "sigma": float(sigma),
    }


class RegimeLGBQuantileWrapper(BaseEstimator, RegressorMixin):
    """Route predictions by VIX regime to per-regime LightGBM quantile models.

    Single canonical implementation used by both M5 (calibration)
    and M7b (walk-forward backtest). Computes the midpoint of P10/P90
    predictions and falls back to cross-regime median when a regime
    has no trained model.
    """

    def __init__(self, lgb_models: dict, feature_columns: list,
                 low_thresh: float = 15.0, high_thresh: float = 20.0):
        self.lgb_models = lgb_models
        self.feature_columns = feature_columns
        self.vix_col_idx = feature_columns.index("vix_level") if "vix_level" in feature_columns else 0
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.fitted_ = True

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        """Predict midpoint of P10/P90, routing by VIX regime.

        Returns array of midpoint predictions (shape: n_samples).
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        preds = np.zeros(n_samples)
        vix = X[:, self.vix_col_idx]

        mask_low = vix < self.low_thresh
        mask_mid = (vix >= self.low_thresh) & (vix < self.high_thresh)
        mask_high = vix >= self.high_thresh

        fallback_mask = np.zeros(n_samples, dtype=bool)

        for regime, mask in (("low", mask_low), ("mid", mask_mid), ("high", mask_high)):
            if not np.any(mask):
                continue
            if regime in self.lgb_models:
                model = self.lgb_models[regime]
                X_regime = X[mask]
                p10 = model["p10"].predict(X_regime)
                p90 = model["p90"].predict(X_regime)
                preds[mask] = (p10 + p90) / 2.0
            else:
                fallback_mask |= mask
                logger.warning(f"No model for regime '{regime}' \u2014 using training median")

        if np.any(fallback_mask):
            all_mid_preds = []
            for m in self.lgb_models.values():
                p10 = m["p10"].predict(X[fallback_mask])
                p90 = m["p90"].predict(X[fallback_mask])
                all_mid_preds.append((p10 + p90) / 2.0)
            preds[fallback_mask] = np.median(all_mid_preds, axis=0)

        return preds


def predict_mu_sigma_p90(log_range_p10: float, log_range_p90: float,
                         regime: str) -> tuple[float, float, float]:
    """Compute mu, sigma from P10/P90 using regime-specific alpha.

    Uses cfg.regime_alphas to get the correct alpha pair for the regime,
    computing sigma = (p90 - p10) / (norm.ppf(alpha_p90) - norm.ppf(alpha_p10)).

    Parameters
    ----------
    log_range_p10 : float
    log_range_p90 : float
    regime : str
        Regime name ('low', 'mid', 'high')

    Returns
    -------
    tuple of (mu, sigma, divisor)
    """
    alpha_p10, alpha_p90 = cfg.regime_alphas.get(regime, (0.10, 0.90))
    divisor = norm.ppf(alpha_p90) - norm.ppf(alpha_p10)
    mu = (log_range_p10 + log_range_p90) / 2.0
    sigma = (log_range_p90 - log_range_p10) / divisor
    return mu, sigma, divisor


def soft_vix_size_multiplier(vix_level: float) -> float:
    """Compute position size multiplier based on soft-VIX band.

    Between wf_soft_vix_lower and wf_max_vix_trade, returns wf_soft_size_mult.
    Outside this band (but still within tradeable range), returns 1.0.

    Parameters
    ----------
    vix_level : float
        Current VIX level

    Returns
    -------
    float
        Size multiplier (1.0 for normal, cfg.wf_soft_size_mult for soft band)
    """
    if cfg.wf_soft_vix_lower <= vix_level <= cfg.wf_max_vix_trade:
        logger.debug(
            f"Soft-VIX band: vix={vix_level:.1f} in "
            f"[{cfg.wf_soft_vix_lower}, {cfg.wf_max_vix_trade}] \u2014 "
            f"applying size multiplier {cfg.wf_soft_size_mult}"
        )
        return cfg.wf_soft_size_mult
    return 1.0

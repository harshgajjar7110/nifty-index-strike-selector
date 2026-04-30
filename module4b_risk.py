"""Pure risk analysis functions for iron condor strategy."""

import numpy as np
from scipy.stats import norm as sp_norm


def breach_probability(strike: float, mu: float, sigma: float, spot: float, side: str) -> float:
    """Compute P(Nifty breaches strike) given log_range ~ Normal(mu, sigma)."""
    if side == "call":
        half_range_needed = (strike - spot) / spot
        if half_range_needed <= 0:
            return 1.0
    elif side == "put":
        half_range_needed = (spot - strike) / spot
        if half_range_needed <= 0:
            return 1.0
    else:
        raise ValueError(f"side must be 'call' or 'put', got {side}")

    log_range_needed = np.log(1.0 + 2.0 * half_range_needed)
    return 1.0 - sp_norm.cdf(log_range_needed, loc=mu, scale=sigma)


def pop_from_chain_iv(
    short_strike: float,
    spot: float,
    dte_days: int,
    iv: float,
    r: float = 0.065,
    q: float = 0.015,
    side: str = "put",
) -> float:
    """P(option expires OTM) using per-strike chain IV and Black-Scholes d2.

    Under risk-neutral measure, N(d2) = P(S_T > K) and N(-d2) = P(S_T < K).
    Short put profits when S_T > K  → POP = N(d2).
    Short call profits when S_T < K → POP = N(-d2).
    """
    if iv <= 0 or dte_days <= 0:
        return 0.5

    T = dte_days / 365.0
    d2 = (np.log(spot / short_strike) + (r - q - 0.5 * iv ** 2) * T) / (iv * np.sqrt(T))

    if side == "put":
        return float(sp_norm.cdf(d2))
    else:  # call
        return float(sp_norm.cdf(-d2))


def cvar(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """Analytical CVaR for Normal(mu, sigma)."""
    z = sp_norm.ppf(1 - alpha)
    phi_z = sp_norm.pdf(z)
    return mu + sigma * phi_z / alpha


import json
from pathlib import Path

_THRESH_PATH = Path(__file__).parent / "models" / "regime_thresholds.json"

def _load_thresholds():
    """Load optimized regime thresholds from file."""
    if _THRESH_PATH.exists():
        try:
            d = json.loads(_THRESH_PATH.read_text())
            return d.get("low_thresh", 15.0), d.get("high_thresh", 20.0)
        except Exception:
            pass
    return 15.0, 20.0

def regime_risk_report(test_df, ngb_models: dict, feature_columns: list) -> dict:
    """Per-regime risk summary from NGB models and test data."""
    report = {}
    _low_t, _high_t = _load_thresholds()
    vix_thresholds = {"low": (0, _low_t), "mid": (_low_t, _high_t), "high": (_high_t, float('inf'))}

    for regime, (vix_min, vix_max) in vix_thresholds.items():
        # Filter by VIX regime
        regime_df = test_df[
            (test_df["vix_level"] >= vix_min) & (test_df["vix_level"] < vix_max)
        ].copy()

        if len(regime_df) == 0:
            report[regime] = {
                "n_rows": 0,
                "coverage": None,
                "cvar_95": None,
                "cvar_99": None,
                "mean_mu": None,
                "mean_sigma": None,
            }
            continue

        model = ngb_models[regime]
        mus = []
        sigmas = []
        coverages = []

        for idx, row in regime_df.iterrows():
            X_row = row[feature_columns].values.reshape(1, -1)
            
            # Use LightGBM P10/P90 to derive mu and sigma
            # P10 = mu - 1.28*sigma, P90 = mu + 1.28*sigma
            p10 = float(model["p10"].predict(X_row)[0])
            p90 = float(model["p90"].predict(X_row)[0])
            
            mu_row = (p10 + p90) / 2.0
            sigma_row = (p90 - p10) / 2.563  # 2 * 1.2815
            
            mus.append(mu_row)
            sigmas.append(sigma_row)

            # Check coverage: actual log_range within [P10, P90]
            actual_log_range = row["log_range"]
            if p10 <= actual_log_range <= p90:
                coverages.append(1)
            else:
                coverages.append(0)

        mean_mu = float(np.mean(mus))
        mean_sigma = float(np.mean(sigmas))
        coverage = float(np.mean(coverages)) if coverages else 0.0
        cvar_95 = cvar(mean_mu, mean_sigma, alpha=0.05)
        cvar_99 = cvar(mean_mu, mean_sigma, alpha=0.01)

        report[regime] = {
            "n_rows": len(regime_df),
            "coverage": coverage,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "mean_mu": mean_mu,
            "mean_sigma": mean_sigma,
        }

    return report

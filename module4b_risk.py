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


def cvar(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """Analytical CVaR for Normal(mu, sigma)."""
    z = sp_norm.ppf(1 - alpha)
    phi_z = sp_norm.pdf(z)
    return mu + sigma * phi_z / alpha


def regime_risk_report(test_df, ngb_models: dict, feature_columns: list) -> dict:
    """Per-regime risk summary from NGB models and test data."""
    report = {}
    vix_thresholds = {"low": (0, 15), "mid": (15, 20), "high": (20, float('inf'))}

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
            dist = model.pred_dist(X_row)
            mu_row = dist.loc[0]
            sigma_row = dist.scale[0]

            mus.append(mu_row)
            sigmas.append(sigma_row)

            # Compute P10, P90 from the distribution
            p10 = sp_norm.ppf(0.10, loc=mu_row, scale=sigma_row)
            p90 = sp_norm.ppf(0.90, loc=mu_row, scale=sigma_row)

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

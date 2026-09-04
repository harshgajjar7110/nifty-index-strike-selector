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

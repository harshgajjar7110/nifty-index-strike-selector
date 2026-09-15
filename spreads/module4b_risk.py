"""Pure risk analysis functions for iron condor strategy."""

import numpy as np
from scipy.stats import norm as sp_norm


def _default_rq(r: float | None, q: float | None) -> tuple[float, float]:
    from config import cfg
    return (cfg.risk_free_rate if r is None else r, cfg.dividend_yield if q is None else q)


def breach_probability(strike: float, mu: float, sigma: float, spot: float, side: str) -> float:
    """Compute P(log-range breaches strike) under Normal(mu, sigma) assumption.
    
    Uses standard log-return parameterization:
    - P(call breach) = 1 - Φ((ln(short_call/spot) - mu) / sigma)
    - P(put breach)  = Φ((ln(short_put/spot) - mu) / sigma)
    """
    if sigma <= 0:
        sigma = 1e-9
    
    if side == "call":
        if strike <= spot:
            return 1.0
        log_k_over_s = np.log(strike / spot)
        z = (log_k_over_s - mu) / sigma
        return float(1.0 - sp_norm.cdf(z))
    
    elif side == "put":
        if strike >= spot:
            return 1.0
        log_k_over_s = np.log(strike / spot)
        z = (log_k_over_s - mu) / sigma
        return float(sp_norm.cdf(z))
    
    else:
        raise ValueError(f"side must be 'call' or 'put', got {side}")


def pop_from_chain_iv(
    short_strike: float,
    spot: float,
    dte_days: int,
    iv: float,
    r: float | None = None,
    q: float | None = None,
    side: str = "put",
) -> float:
    """P(option expires OTM) using per-strike chain IV and Black-Scholes d2.

    Under risk-neutral measure, N(d2) = P(S_T > K) and N(-d2) = P(S_T < K).
    Short put profits when S_T > K  → POP = N(d2).
    Short call profits when S_T < K → POP = N(-d2).
    """
    r, q = _default_rq(r, q)
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

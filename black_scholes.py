"""
Shared Black-Scholes pricing utilities.
Centralizes option pricing to avoid duplication across modules.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_price(
    S: float,
    K: float,
    T_years: float,
    sigma_annual: float,
    option_type: str = "call",
    r: float = 0.065,
    q: float = 0.015,
) -> float:
    """Black-Scholes price for a European option.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T_years : float
        Time to expiry in years
    sigma_annual : float
        Annualized volatility
    option_type : str
        'call' or 'put'
    r : float
        Risk-free rate (default from config)
    q : float
        Dividend yield (default from config)

    Returns
    -------
    float
        Option price (>= 0)
    """
    if T_years <= 0:
        T_years = 1.0 / 365.0
    if sigma_annual <= 0:
        sigma_annual = 0.10

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_annual**2) * T_years) / (
        sigma_annual * np.sqrt(T_years)
    )
    d2 = d1 - sigma_annual * np.sqrt(T_years)

    if option_type == "call":
        price = S * np.exp(-q * T_years) * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * np.exp(-q * T_years) * norm.cdf(-d1)

    return max(price, 0.0)


def estimate_ic_premium_bs(
    spot: float,
    short_put: float,
    long_put: float,
    short_call: float,
    long_call: float,
    dte_days: int,
    vix_level: float,
    r: float = 0.065,
    q: float = 0.015,
) -> float:
    """Net credit for an iron condor via Black-Scholes.

    Sells short_put/short_call, buys long_put/long_call wings.
    """
    T = max(dte_days, 1) / 365.0
    sigma = max(vix_level / 100.0, 0.05)

    net = (
        black_scholes_price(spot, short_put, T, sigma, "put", r, q)
        - black_scholes_price(spot, long_put, T, sigma, "put", r, q)
        + black_scholes_price(spot, short_call, T, sigma, "call", r, q)
        - black_scholes_price(spot, long_call, T, sigma, "call", r, q)
    )
    return max(net, 0.0)


def bs_price_with_skew(
    S: float,
    K: float,
    T_years: float,
    sigma_annual: float,
    option_type: str = "put",
    r: float = 0.065,
    q: float = 0.015,
    vol_skew_factor: float = 0.0,
) -> float:
    """BS price with optional vol skew for OTM options.

    OTM puts/calls trade at higher IV proportional to OTM distance.
    """
    if T_years <= 0:
        T_years = 1.0 / 365.0
    if sigma_annual <= 0:
        sigma_annual = 0.10

    # Apply skew
    if vol_skew_factor > 0:
        if option_type == "put" and K < S:
            otm_pct = (S - K) / S
            sigma_annual = sigma_annual + vol_skew_factor * otm_pct
        elif option_type == "call" and K > S:
            otm_pct = (K - S) / S
            sigma_annual = sigma_annual + vol_skew_factor * otm_pct

    return black_scholes_price(S, K, T_years, sigma_annual, option_type, r, q)

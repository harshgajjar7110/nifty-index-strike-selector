"""
Module 10: NSE Transaction Costs & Slippage Engine
Calculates SEBI, STT, GST, and Stamp Duty for Nifty 50 Options.
Includes slippage modeling for realistic Net P&L.
"""

import os
from loguru import logger

def calculate_nse_charges(
    premium_pts: float,
    num_legs: int = 4,
    lot_size: int = 25,
    is_sell: bool = True,
    brokerage_per_order: float = 20.0
) -> dict:
    """
    Calculate total transaction costs for an NSE Options trade.
    
    Parameters:
    -----------
    premium_pts : float
        Total premium points involved in the transaction (sum of all legs).
    num_legs : int
        Number of legs (4 for Iron Condor, 2 for Credit Spread).
    lot_size : int
        Nifty lot size (default 25).
    is_sell : bool
        Whether this is the entry (sell) or exit (buy/expiry).
    """
    turnover = premium_pts * lot_size
    
    # 1. Brokerage (₹20 per executed order/leg)
    total_brokerage = num_legs * brokerage_per_order
    
    # 2. STT (Securities Transaction Tax)
    # 0.0625% on Sell Side Premium. 0% on Buy side.
    stt = (0.000625 * turnover) if is_sell else 0.0
    
    # 3. Exchange Transaction Charges (NSE)
    # Approx 0.053% of premium
    exchange_charges = 0.00053 * turnover
    
    # 4. SEBI Charges
    # ₹10 per crore (0.000001)
    sebi_fees = 0.000001 * turnover
    
    # 5. Stamp Duty
    # 0.003% on Buy side (for options)
    stamp_duty = (0.00003 * turnover) if not is_sell else 0.0
    
    # 6. GST
    # 18% on (Brokerage + Exchange Charges + SEBI Fees)
    gst = 0.18 * (total_brokerage + exchange_charges + sebi_fees)
    
    total_costs = total_brokerage + stt + exchange_charges + sebi_fees + stamp_duty + gst
    
    return {
        "total_costs": round(total_costs, 2),
        "breakdown": {
            "brokerage": round(total_brokerage, 2),
            "stt": round(stt, 2),
            "exchange": round(exchange_charges, 2),
            "gst": round(gst, 2),
            "stamp": round(stamp_duty, 2)
        },
        "cost_per_lot_pts": round(total_costs / lot_size, 2)
    }

def apply_slippage(premium_pts: float, num_legs: int, slippage_per_leg: float = 1.0) -> float:
    """
    Apply bid-ask slippage to the premium.
    For sellers: you receive LESS than the mid-price.
    For buyers: you pay MORE than the mid-price.
    """
    return premium_pts - (num_legs * slippage_per_leg)

def estimate_ic_premium(
    spot: float,
    short_put: float, long_put: float,
    short_call: float, long_call: float,
    dte_days: int,
    vix_level: float,
    r: float = 0.065,
    q: float = 0.015,       # Nifty dividend yield ~1.5%
) -> float:
    """Black-Scholes IC net credit: sell short legs, buy wing legs."""
    from scipy.stats import norm
    import numpy as np

    T = max(dte_days, 1) / 365.0
    sigma = max(vix_level / 100.0, 0.05)

    def _bs(S, K, opt):
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt == 'put':
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    net = (_bs(spot, short_put,  'put')
         - _bs(spot, long_put,   'put')
         + _bs(spot, short_call, 'call')
         - _bs(spot, long_call,  'call'))
    return max(net, 0.0)

if __name__ == "__main__":
    # Test for a 4-leg Iron Condor collecting 80 points
    costs = calculate_nse_charges(80, num_legs=4)
    print(f"NSE Charges for IC (80 pts): ₹{costs['total_costs']}")
    print(f"Cost in Points: {costs['cost_per_lot_pts']} pts")
    print(f"Breakdown: {costs['breakdown']}")

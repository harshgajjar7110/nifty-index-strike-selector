"""
Module 15: Volatility Skew & Term Structure Features
Computes Volatility Skew (25-Delta Put IV - 25-Delta Call IV) 
and Term Structure (Front month IV - Back month IV) using existing black_scholes library.
"""

from loguru import logger
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from black_scholes import implied_volatility, black_scholes_delta

def compute_option_metrics(spot: float, strike: float, price: float, dte_days: int, option_type: str) -> dict:
    """Computes IV and Delta for a given option price using existing black_scholes logic."""
    t_years = max(dte_days, 1) / 365.0
    iv = implied_volatility(price, spot, strike, t_years, option_type)
    delta = black_scholes_delta(spot, strike, t_years, iv, option_type) if iv > 0 else 0.0
    return {"iv": iv, "delta": delta}

def extract_skew_metrics(chain_data: dict, spot: float, dte_days: int) -> dict:
    """
    Given a dict of strikes -> prices, find 25-delta Put, 25-delta Call, and ATM IV.
    chain_data format: {strike: {'call_price': float, 'put_price': float}}
    """
    metrics = []
    for strike, prices in chain_data.items():
        cp = prices.get('call_price', 0.0)
        pp = prices.get('put_price', 0.0)

        c_met = compute_option_metrics(spot, strike, cp, dte_days, "call")
        p_met = compute_option_metrics(spot, strike, pp, dte_days, "put")

        # Skip strikes where IV could not be solved (price outside Brentq bounds)
        if (not isinstance(c_met["iv"], float) or (c_met["iv"] != c_met["iv"])) or \
           (not isinstance(p_met["iv"], float) or (p_met["iv"] != p_met["iv"])):
            continue

        metrics.append({
            "strike": strike,
            "call_iv": c_met["iv"],
            "call_delta": c_met["delta"],
            "put_iv": p_met["iv"],
            "put_delta": p_met["delta"],
        })
        
    if not metrics:
        return None
        
    # Find ATM (closest strike to spot)
    closest_atm = min(metrics, key=lambda x: abs(x["strike"] - spot))
    atm_iv = (closest_atm["call_iv"] + closest_atm["put_iv"]) / 2.0
    
    # Find 25-Delta Call (delta ~ 0.25)
    call_25 = min(metrics, key=lambda x: abs(x["call_delta"] - 0.25))
    
    # Find 25-Delta Put (delta ~ -0.25)
    put_25 = min(metrics, key=lambda x: abs(x["put_delta"] - (-0.25)))
    
    # Skew Index: 25-Delta Put IV minus 25-Delta Call IV
    skew_index = put_25["put_iv"] - call_25["call_iv"]
    
    return {
        "atm_iv": atm_iv,
        "put_25_iv": put_25["put_iv"],
        "call_25_iv": call_25["call_iv"],
        "skew_index": skew_index
    }

def analyze_surface(front_chain: dict, back_chain: dict, spot: float, dte_front: int, dte_back: int) -> dict:
    """
    Calculates Volatility Skew and Term Structure.
    Returns dictionary with front_skew_index, front_atm_iv, and term_structure_spread.
    """
    logger.info(f"Analyzing Volatility Surface for Spot={spot}")
    
    front_metrics = extract_skew_metrics(front_chain, spot, dte_front)
    back_metrics = extract_skew_metrics(back_chain, spot, dte_back) if back_chain else None
    
    result = {"spot": spot, "dte_front": dte_front}
    
    if front_metrics:
        result.update({
            "front_atm_iv": front_metrics["atm_iv"],
            "front_skew_index": front_metrics["skew_index"]
        })
        logger.info(f"Front Month Skew Index: {front_metrics['skew_index']:.4f}")
        
    if back_metrics and front_metrics:
        ts_spread = front_metrics["atm_iv"] - back_metrics["atm_iv"]
        result.update({
            "back_atm_iv": back_metrics["atm_iv"],
            "term_structure_spread": ts_spread
        })
        logger.info(f"Term Structure Spread (Front - Back): {ts_spread:.4f}")
        
    return result

if __name__ == "__main__":
    # Quick sanity test
    logger.info("module15_iv_skew initialized.")

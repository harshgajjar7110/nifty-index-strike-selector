import pytest
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from black_scholes import black_scholes_price, implied_volatility, black_scholes_delta

def test_implied_volatility_atm_call():
    # TC-01: ATM Call verification
    S = 24000
    K = 24000
    T_years = 7 / 365.0
    r = 0.065
    q = 0.015
    true_iv = 0.125
    
    price = black_scholes_price(S, K, T_years, true_iv, "call", r, q)
    calc_iv = implied_volatility(price, S, K, T_years, "call", r, q)
    
    assert abs(calc_iv - true_iv) < 1e-4

def test_implied_volatility_otm_put():
    # Verify accurate calculation for skewed out-of-the-money puts
    S = 24000
    K = 23000
    T_years = 14 / 365.0
    true_iv = 0.18
    
    price = black_scholes_price(S, K, T_years, true_iv, "put")
    calc_iv = implied_volatility(price, S, K, T_years, "put")
    
    assert abs(calc_iv - true_iv) < 1e-4

def test_implied_volatility_zero_price():
    # Edge case: zero price should return zero IV without crashing
    calc_iv = implied_volatility(0.0, 24000, 24000, 7/365.0, "call")
    assert calc_iv == 0.0

def test_black_scholes_delta():
    # ATM call delta should be approximately 0.5
    delta = black_scholes_delta(24000, 24000, 30/365.0, 0.15, "call")
    assert 0.45 < delta < 0.55

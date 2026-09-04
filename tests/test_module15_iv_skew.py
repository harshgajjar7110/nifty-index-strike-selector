import pytest
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from features_skew.module15_iv_skew import extract_skew_metrics
from black_scholes import black_scholes_price

def test_extract_skew_metrics():
    # TC-02: Verify 25-Delta Skew calculation
    spot = 24000
    dte = 14
    t_years = dte / 365.0
    
    # Mock an option chain: Puts are skewed higher (20% IV) vs Calls (12% IV)
    mock_chain = {}
    for strike in range(23000, 25100, 100):
        if strike < spot:
            iv = 0.20 # Skewed put
        elif strike > spot:
            iv = 0.12 # Normal call
        else:
            iv = 0.15 # ATM
            
        c_price = black_scholes_price(spot, strike, t_years, iv, "call")
        p_price = black_scholes_price(spot, strike, t_years, iv, "put")
        mock_chain[strike] = {"call_price": c_price, "put_price": p_price}
        
    metrics = extract_skew_metrics(mock_chain, spot, dte)
    
    assert metrics is not None
    assert "atm_iv" in metrics
    assert "skew_index" in metrics
    
    # Since Put IV is 20% and Call IV is 12%, Skew Index should be around 0.08
    assert metrics["skew_index"] > 0.0
    assert abs(metrics["skew_index"] - 0.08) < 0.02

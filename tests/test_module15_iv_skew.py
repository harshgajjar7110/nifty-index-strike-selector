import pytest
from utils.features_skew.module15_iv_skew import extract_skew_metrics
from utils.black_scholes import black_scholes_price

def test_extract_skew_metrics():
    spot = 24000
    dte = 14
    t_years = dte / 365.0
    
    mock_chain = {}
    for strike in range(23000, 25100, 100):
        if strike < spot:
            iv = 0.20
        elif strike > spot:
            iv = 0.12
        else:
            iv = 0.15
            
        c_price = black_scholes_price(spot, strike, t_years, iv, "call")
        p_price = black_scholes_price(spot, strike, t_years, iv, "put")
        mock_chain[strike] = {"call_price": c_price, "put_price": p_price}
        
    metrics = extract_skew_metrics(mock_chain, spot, dte)
    
    assert metrics is not None
    assert "atm_iv" in metrics
    assert "skew_index" in metrics
    assert metrics["skew_index"] > 0.0
    assert abs(metrics["skew_index"] - 0.08) < 0.02

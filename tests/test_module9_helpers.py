import pytest
from module9_spreads import _get_strike_iv


def test_direct_hit_put():
    oi = {23200: {"put_iv": 0.25, "call_iv": 0.18, "put_oi": 2000, "call_oi": 1000}}
    assert _get_strike_iv(oi, 23200, 23800, "put", atm_iv_fallback=0.17) == 0.25


def test_direct_hit_call():
    oi = {24400: {"put_iv": 0.15, "call_iv": 0.17, "put_oi": 500, "call_oi": 1200}}
    assert _get_strike_iv(oi, 24400, 23800, "call", atm_iv_fallback=0.17) == 0.17


def test_interpolation():
    oi = {
        23100: {"put_iv": 0.28, "call_iv": 0.20, "put_oi": 1000, "call_oi": 500},
        23300: {"put_iv": 0.24, "call_iv": 0.18, "put_oi": 1500, "call_oi": 800},
    }
    iv = _get_strike_iv(oi, 23200, 23800, "put", atm_iv_fallback=0.17)
    assert abs(iv - 0.26) < 0.001


def test_empty_chain_fallback():
    iv = _get_strike_iv({}, 23200, 23800, "put", atm_iv_fallback=0.17)
    assert iv == 0.17


def test_zero_iv_falls_through_to_fallback():
    oi = {23200: {"put_iv": 0.0, "call_iv": 0.0, "put_oi": 100, "call_oi": 50}}
    iv = _get_strike_iv(oi, 23200, 23800, "put", atm_iv_fallback=0.17)
    assert iv == 0.17

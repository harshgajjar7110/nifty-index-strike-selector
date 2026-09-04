import pytest
from module4b_risk import pop_from_chain_iv


def test_pop_short_put_atm():
    # ATM short put → ~50% POP
    pop = pop_from_chain_iv(short_strike=23800, spot=23800, dte_days=7, iv=0.17, side="put")
    assert 0.45 < pop < 0.55


def test_pop_short_put_otm():
    # 2.5% OTM put, 19 DTE, 25% IV (with skew) → realistic 65-85%
    pop = pop_from_chain_iv(short_strike=23200, spot=23800, dte_days=19, iv=0.25, side="put")
    assert 0.65 < pop < 0.85


def test_pop_short_call_otm():
    # 2.5% OTM call, 19 DTE, 17% ATM IV → realistic 70-85%
    pop = pop_from_chain_iv(short_strike=24400, spot=23800, dte_days=19, iv=0.17, side="call")
    assert 0.65 < pop < 0.90


def test_pop_zero_iv_fallback():
    # Zero IV → returns 0.5 sentinel
    pop = pop_from_chain_iv(short_strike=23200, spot=23800, dte_days=10, iv=0.0, side="put")
    assert pop == 0.5


def test_pop_itm_put():
    # ITM short put (strike > spot) → POP < 0.5
    pop = pop_from_chain_iv(short_strike=24000, spot=23800, dte_days=10, iv=0.20, side="put")
    assert pop < 0.5

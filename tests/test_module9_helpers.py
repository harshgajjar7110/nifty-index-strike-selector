import pytest
from module9_spreads import _get_strike_iv, generate_credit_spread


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


# Test generate_credit_spread with chain IV
OI_STRIKES_SAMPLE = {
    23200: {"put_iv": 0.25, "call_iv": 0.18, "put_oi": 10000, "call_oi": 5000},
    23800: {"put_iv": 0.17, "call_iv": 0.17, "put_oi": 8000, "call_oi": 8000},
    24400: {"put_iv": 0.14, "call_iv": 0.17, "put_oi": 3000, "call_oi": 12000},
    24800: {"put_iv": 0.13, "call_iv": 0.16, "put_oi": 1000, "call_oi": 6000},
}


def test_pop_realistic_bull_put():
    spread = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=19, vix_level=17.4, garch_vol=0.01,
        direction="bull_put", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes=OI_STRIKES_SAMPLE,
    )
    # POP should be realistic, not 99.5%
    assert 0.55 < spread["pop_pct"] < 0.95, f"POP={spread['pop_pct']} not realistic"


def test_pop_realistic_bear_call():
    spread = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=19, vix_level=17.4, garch_vol=0.01,
        direction="bear_call", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes=OI_STRIKES_SAMPLE,
    )
    assert 0.55 < spread["pop_pct"] < 0.95, f"POP={spread['pop_pct']} not realistic"


def test_no_oi_falls_back_gracefully():
    spread = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=19, vix_level=17.4, garch_vol=0.01,
        direction="bull_put", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes={},  # empty — should fall back gracefully
    )
    assert spread["pop_pct"] is not None
    assert 0.0 < spread["pop_pct"] < 1.0

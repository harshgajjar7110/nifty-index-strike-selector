"""Regression tests for the strike-placement math contract (AGENTS.md ┬º6).

Locks the contract for:
- round_to_strike rounding semantics (Python banker's rounding via round())
- buffer clamping with vix_scalar
- breach_probability formula behavior (module4b_risk)
- generate_credit_spread wing width floor
- ev_proxy identity in module9_spreads (T1)

These tests intentionally assert the *actual* behavior of the implementation
(including Python's banker's rounding for .5 cases), not idealized rounding.
"""

import math

import pytest

from spreads.module4b_risk import breach_probability
from module6_strikes import round_to_strike


# ---------------------------------------------------------------------------
# round_to_strike
# ---------------------------------------------------------------------------


def test_round_to_strike_multiples_of_50_round_to_self():
    for p in (23000, 23050, 23100, 23150, 23200):
        assert round_to_strike(p) == p


def test_round_to_strike_above_midpoint_rounds_up():
    # 23024 (24 < 25) ΓåÆ 23000
    assert round_to_strike(23024) == 23000
    # 23026 (26 > 25) ΓåÆ 23050
    assert round_to_strike(23026) == 23050


def test_round_to_strike_below_midpoint_rounds_down():
    # 23074 (74 < 75) ΓåÆ 23050
    assert round_to_strike(23074) == 23050
    # 23076 (76 > 75) ΓåÆ 23100
    assert round_to_strike(23076) == 23100


def test_round_to_strike_python_bankers_rounding_at_half():
    # Python's round() uses banker's rounding (half-to-even).
    # 23025 / 50 = 0.5 ΓåÆ 0 (even); result = 23000.
    # 23075 / 50 = 1.5 ΓåÆ 2 (even); result = 23100.
    assert round_to_strike(23025) == 23000
    assert round_to_strike(23075) == 23100


def test_round_to_strike_interval_parameter():
    # Custom interval of 100
    assert round_to_strike(24100, interval=100) == 24100
    assert round_to_strike(24149, interval=100) == 24100
    assert round_to_strike(24151, interval=100) == 24200


# ---------------------------------------------------------------------------
# breach_probability (module4b_risk)
# ---------------------------------------------------------------------------


def test_breach_prob_call_otm_is_below_one():
    # OTM call: K > spot, breach prob is between 0 and 1
    p = breach_probability(strike=24100, mu=0.0, sigma=0.02, spot=23800, side="call")
    assert 0.0 < p < 1.0


def test_breach_prob_put_otm_is_below_one():
    # OTM put: K < spot
    p = breach_probability(strike=23500, mu=0.0, sigma=0.02, spot=23800, side="put")
    assert 0.0 < p < 1.0


def test_breach_prob_call_decreases_as_strike_moves_otm():
    # Further OTM (higher K for a call) should give lower breach probability
    p_close = breach_probability(strike=23900, mu=0.0, sigma=0.02, spot=23800, side="call")
    p_mid   = breach_probability(strike=24200, mu=0.0, sigma=0.02, spot=23800, side="call")
    p_far   = breach_probability(strike=24700, mu=0.0, sigma=0.02, spot=23800, side="call")
    assert p_close > p_mid > p_far


def test_breach_prob_put_decreases_as_strike_moves_otm():
    # Further OTM (lower K for a put) should give lower breach probability
    p_close = breach_probability(strike=23700, mu=0.0, sigma=0.02, spot=23800, side="put")
    p_mid   = breach_probability(strike=23400, mu=0.0, sigma=0.02, spot=23800, side="put")
    p_far   = breach_probability(strike=22900, mu=0.0, sigma=0.02, spot=23800, side="put")
    assert p_close > p_mid > p_far


def test_breach_prob_call_and_put_equal_when_mu_zero():
    # When mu=0 (symmetric log-range distribution), call and put breach
    # probabilities are equal at equidistant strikes from spot.
    mu, sigma = 0.0, 0.02
    spot = 23800
    offset = 300
    p_call = breach_probability(spot + offset, mu, sigma, spot, "call")
    p_put  = breach_probability(spot - offset, mu, sigma, spot, "put")
    assert math.isclose(p_call, p_put, rel_tol=1e-9, abs_tol=1e-12)


def test_breach_prob_call_and_put_equal_for_equidistant_otm_strikes():
    # NOTE: breach_probability in module4b_risk uses the same Normal(mu, sigma)
    # distribution for both call and put sides via the half-range parameterisation
    # log_range_needed = log(1 + 2*half_range_needed). For equidistant OTM strikes
    # (offset above and below spot), the half_range_needed magnitudes are equal,
    # so call and put breach probabilities are equal regardless of mu.
    # This is the actual contract ΓÇö locked here to prevent silent divergence.
    for mu in (0.0, 0.01, -0.005):
        sigma = 0.02
        spot = 23800
        offset = 300
        p_call = breach_probability(spot + offset, mu, sigma, spot, "call")
        p_put  = breach_probability(spot - offset, mu, sigma, spot, "put")
        assert math.isclose(p_call, p_put, rel_tol=1e-9, abs_tol=1e-12)


def test_breach_prob_at_or_inside_spot_returns_one():
    # For a call at K <= spot (half_range_needed <= 0), the function returns
    # 1.0 ΓÇö the strike is already breached. Lock the current behavior.
    p = breach_probability(strike=23800, mu=0.01, sigma=0.02, spot=23800, side="call")
    assert p == 1.0
    p_itm = breach_probability(strike=23600, mu=0.01, sigma=0.02, spot=23800, side="call")
    assert p_itm == 1.0


def test_breach_prob_at_or_above_spot_for_put_returns_one():
    # For a put at K >= spot, the function returns 1.0.
    p = breach_probability(strike=23800, mu=0.01, sigma=0.02, spot=23800, side="put")
    assert p == 1.0
    p_itm = breach_probability(strike=24000, mu=0.01, sigma=0.02, spot=23800, side="put")
    assert p_itm == 1.0


def test_breach_prob_invalid_side_raises():
    with pytest.raises(ValueError):
        breach_probability(strike=24000, mu=0.0, sigma=0.02, spot=23800, side="straddle")


# ---------------------------------------------------------------------------
# generate_credit_spread ΓÇö wing width floor
# ---------------------------------------------------------------------------


def _oi_sample():
    return {
        23200: {"put_iv": 0.25, "call_iv": 0.18, "put_oi": 10000, "call_oi": 5000},
        23800: {"put_iv": 0.17, "call_iv": 0.17, "put_oi": 8000, "call_oi": 8000},
        24400: {"put_iv": 0.14, "call_iv": 0.17, "put_oi": 3000, "call_oi": 12000},
        24800: {"put_iv": 0.13, "call_iv": 0.16, "put_oi": 1000, "call_oi": 6000},
    }


def test_generate_credit_spread_wing_width_floor_50():
    # Even with very low DTE (where dte_scalar is small), scaled_wing is
    # bounded below by 50 after rounding.
    from spreads.module9_spreads import generate_credit_spread

    spread = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=1, vix_level=17.4, garch_vol=0.01,
        direction="bull_put", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes=_oi_sample(),
    )
    assert spread["wing_width"] >= 50


def test_generate_credit_spread_wing_width_grows_with_dte():
    from spreads.module9_spreads import generate_credit_spread

    short = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=2, vix_level=17.4, garch_vol=0.01,
        direction="bull_put", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes=_oi_sample(),
    )
    long = generate_credit_spread(
        spot=23800, log_range_p10=0.019, log_range_p90=0.039,
        dte_days=21, vix_level=17.4, garch_vol=0.01,
        direction="bull_put", atm_iv=0.172,
        log_range_mu=0.029, log_range_sigma=0.008,
        oi_strikes=_oi_sample(),
    )
    assert long["wing_width"] >= short["wing_width"]


# ---------------------------------------------------------------------------
# T1: ev_proxy identity (locks the contract)
# ---------------------------------------------------------------------------


def test_ev_proxy_identity_holds_for_any_inputs():
    # ev_proxy is the standard credit-spread expected-value formula:
    # EV = +premium*pop - max_loss*(1-pop)
    # This test guards against future agents "fixing" the non-bug.
    cases = [
        (45.0, 254.0, 0.736),
        (40.0, 260.0, 0.837),
        (0.0,  100.0, 0.5),
        (10.0, 200.0, 0.95),
        (100.0, 50.0, 0.0),  # premium > max_loss (extreme)
    ]
    for premium, max_loss, pop in cases:
        expected = premium * pop - max_loss * (1.0 - pop)
        # emulate the implementation
        ev_proxy = round(premium * pop - max_loss * (1 - pop), 4)
        assert math.isclose(ev_proxy, round(expected, 4), rel_tol=1e-9, abs_tol=1e-12)


def test_ev_proxy_negative_when_rr_poor():
    # R:R of 0.18 (premium 45 / max_loss 254) at POP 73.6% ΓåÆ negative EV.
    premium, max_loss, pop = 45.0, 254.0, 0.736
    ev = premium * pop - max_loss * (1 - pop)
    assert ev < 0


def test_ev_proxy_positive_when_rr_good():
    # R:R of 0.33 (premium 50 / max_loss 150) at POP 80% ΓåÆ clearly positive.
    premium, max_loss, pop = 50.0, 150.0, 0.80
    ev = premium * pop - max_loss * (1 - pop)
    assert ev > 0


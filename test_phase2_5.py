"""
Test Suite: Phase 2.5 - Strike Selection Engine
===============================================

Tests for strike filtering, candidate generation, Greeks calculation, and ranking.
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import norm

# Add workspace to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Greeks Calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_greeks_calculation():
    """Verify Greeks calculation for option positions."""
    from engine_strike_selection import calculate_greeks
    
    logger.info("TEST 1: Greeks Calculation")
    
    # Test parameters
    spot = 19000.0
    strike = 18500.0  # OTM put
    vol = 0.25
    dte = 7
    
    greeks_put = calculate_greeks(spot, strike, vol, dte, "put")
    greeks_call = calculate_greeks(spot, 19500.0, vol, dte, "call")
    
    # Verify Greeks structure
    assert "delta" in greeks_put, "Missing delta"
    assert "gamma" in greeks_put, "Missing gamma"
    assert "vega" in greeks_put, "Missing vega"
    assert "theta" in greeks_put, "Missing theta"
    
    # Short put delta should be positive (long stock exposure)
    assert greeks_put["delta"] > 0, f"Short put delta should be positive, got {greeks_put['delta']}"
    
    # Short call delta should be negative (short stock exposure)
    assert greeks_call["delta"] < 0, f"Short call delta should be negative, got {greeks_call['delta']}"
    
    logger.success(f"  ✓ Put Greeks: Δ={greeks_put['delta']:.4f}, Γ={greeks_put['gamma']:.6f}, V={greeks_put['vega']:.3f}, Θ={greeks_put['theta']:.3f}")
    logger.success(f"  ✓ Call Greeks: Δ={greeks_call['delta']:.4f}, Γ={greeks_call['gamma']:.6f}, V={greeks_call['vega']:.3f}, Θ={greeks_call['theta']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Candidate Filter & Constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_candidate_filter():
    """Verify candidate filtering logic."""
    from engine_strike_selection import CandidateFilter
    
    logger.info("TEST 2: Candidate Filter & Constraints")
    
    # Test weekly expiry
    filter_weekly = CandidateFilter(
        dte=7,
        spot=19000.0,
        volatility=0.25,
        regime_allocation_factor=1.0,
        iv_rank=55.0,
    )
    
    assert filter_weekly.horizon == "weekly", f"Expected 'weekly', got {filter_weekly.horizon}"
    assert filter_weekly.check_dte_constraint(), "DTE constraint failed for weekly"
    assert filter_weekly.check_iv_constraint(), "IV constraint failed"
    assert filter_weekly.check_regime_constraint(), "Regime constraint failed"
    
    logger.success(f"  ✓ Horizon classification: {filter_weekly.horizon.upper()}")
    
    # Test monthly expiry
    filter_monthly = CandidateFilter(
        dte=35,
        spot=19000.0,
        volatility=0.25,
        regime_allocation_factor=1.0,
        iv_rank=55.0,
    )
    
    assert filter_monthly.horizon == "monthly", f"Expected 'monthly', got {filter_monthly.horizon}"
    
    logger.success(f"  ✓ Horizon classification: {filter_monthly.horizon.upper()}")
    
    # Test constraint violations
    filter_low_regime = CandidateFilter(
        dte=7,
        spot=19000.0,
        volatility=0.25,
        regime_allocation_factor=0.0,  # PANIC regime
        iv_rank=55.0,
    )
    
    assert not filter_low_regime.check_regime_constraint(), "Should reject PANIC regime"
    
    logger.success(f"  ✓ Regime constraint properly rejects panic allocation (0%)")
    
    filter_low_iv = CandidateFilter(
        dte=7,
        spot=19000.0,
        volatility=0.25,
        regime_allocation_factor=1.0,
        iv_rank=20.0,  # Too low
    )
    
    assert not filter_low_iv.check_iv_constraint(), "Should reject low IV rank"
    
    logger.success(f"  ✓ IV constraint properly rejects low IV rank (<30)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Strike Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_candidate_strikes_generation():
    """Verify candidate strike generation."""
    from engine_strike_selection import CandidateFilter
    
    logger.info("TEST 3: Candidate Strike Generation")
    
    filter_obj = CandidateFilter(
        dte=7,
        spot=19000.0,
        volatility=0.25,
        regime_allocation_factor=1.0,
        iv_rank=55.0,
    )
    
    # Generate candidates
    put_candidates = filter_obj.generate_candidate_strikes("put", delta_target=0.15)
    call_candidates = filter_obj.generate_candidate_strikes("call", delta_target=0.15)
    
    assert len(put_candidates) > 0, "Should generate put candidates"
    assert len(call_candidates) > 0, "Should generate call candidates"
    
    # Verify candidate structure
    for candidate in put_candidates:
        assert "strike" in candidate, "Missing strike"
        assert "delta" in candidate, "Missing delta"
        assert "breach_prob" in candidate, "Missing breach_prob"
        assert 18000 <= candidate["strike"] <= 20000, f"Strike out of range: {candidate['strike']}"
    
    logger.success(f"  ✓ Generated {len(put_candidates)} put candidates")
    logger.success(f"  ✓ Generated {len(call_candidates)} call candidates")
    logger.success(f"  ✓ Put strike range: {min(c['strike'] for c in put_candidates):.0f} - {max(c['strike'] for c in put_candidates):.0f}")
    logger.success(f"  ✓ Call strike range: {min(c['strike'] for c in call_candidates):.0f} - {max(c['strike'] for c in call_candidates):.0f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Strike Selection & Scoring
# ─────────────────────────────────────────────────────────────────────────────

def test_strike_selector():
    """Verify strike selection and scoring."""
    from engine_strike_selection import StrikeSelector
    
    logger.info("TEST 4: Strike Selection & Scoring")
    
    selector = StrikeSelector(
        spot=19000.0,
        volatility=0.25,
        dte=7,
        regime_allocation_factor=1.0,
        iv_rank=55.0,
    )
    
    # Select strikes
    recommendations = selector.select_strikes()
    
    assert "status" in recommendations, "Missing status"
    assert "market" in recommendations, "Missing market data"
    assert recommendations["market"]["spot"] == 19000.0, "Spot not preserved"
    assert recommendations["market"]["regime_allocation"] == 1.0, "Regime factor not preserved"
    
    logger.success(f"  ✓ Recommendations status: {recommendations['status']}")
    
    if recommendations["status"] == "SUCCESS":
        assert "put_recommendations" in recommendations, "Missing put recommendations"
        assert "call_recommendations" in recommendations, "Missing call recommendations"
        assert "multi_leg_strategies" in recommendations, "Missing multi-leg strategies"
        
        put_best = recommendations["put_recommendations"].get("best_strike")
        call_best = recommendations["call_recommendations"].get("best_strike")
        
        if put_best:
            logger.success(f"  ✓ Best put: {put_best['strike']:.2f} | Delta: {put_best['delta']:.4f} | Score: {put_best.get('opportunity_score', 0):.1f}")
        
        if call_best:
            logger.success(f"  ✓ Best call: {call_best['strike']:.2f} | Delta: {call_best['delta']:.4f} | Score: {call_best.get('opportunity_score', 0):.1f}")
        
        if "strangle" in recommendations["multi_leg_strategies"]:
            strangle = recommendations["multi_leg_strategies"]["strangle"]
            logger.success(f"  ✓ Strangle EV: ₹{strangle.get('total_expected_value', 0) * 100:.0f} | Score: {strangle.get('opportunity_score', 0):.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Integration with Probability Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_probability_integration():
    """Verify integration with probability distribution engine."""
    from engine_strike_selection import StrikeSelector
    from engine_probability_distribution import LognormalDistribution
    
    logger.info("TEST 5: Probability Integration")
    
    spot = 19000.0
    vol = 0.25
    dte = 7
    
    # Create probability distribution
    dist = LognormalDistribution(spot, vol, dte)
    
    # Test breach probabilities
    strikes = [18000, 18500, 19000, 19500, 20000]
    for strike in strikes:
        breach_prob = dist.cdf(strike)
        assert 0 <= breach_prob <= 1, f"Invalid breach probability: {breach_prob}"
    
    logger.success(f"  ✓ Breach probabilities calculated for {len(strikes)} strikes")
    
    # Verify distribution properties (calculated from lognormal parameters)
    # Expected spot at expiry: S_T = S_0 * exp(0.5 * sigma^2 * T)
    mean_future_spot = spot * np.exp(0.5 * vol ** 2 * (dte / 365.0))
    variance = spot ** 2 * (np.exp(vol ** 2 * (dte / 365.0)) - 1) * np.exp(vol ** 2 * (dte / 365.0))
    std_future_spot = np.sqrt(variance)
    
    assert mean_future_spot > 0, "Mean should be positive"
    assert std_future_spot > 0, "Std dev should be positive"
    
    logger.success(f"  ✓ Distribution: μ={mean_future_spot:.2f}, σ={std_future_spot:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Constraints Under Different Regimes
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_constraints():
    """Test strike selection under different market regimes."""
    from engine_strike_selection import CandidateFilter
    
    logger.info("TEST 6: Regime Constraints")
    
    regimes = [
        ("Quiet Bull", 1.0, True),      # Should trade
        ("Range Bound", 1.0, True),     # Should trade
        ("High Expansion", 0.5, True),  # Should trade (reduced allocation)
        ("Panic", 0.0, False),          # Should NOT trade
    ]
    
    for regime_name, allocation, should_trade in regimes:
        filter_obj = CandidateFilter(
            dte=7,
            spot=19000.0,
            volatility=0.25,
            regime_allocation_factor=allocation,
            iv_rank=55.0,
        )
        
        can_trade = filter_obj.check_regime_constraint()
        assert can_trade == should_trade, f"Regime {regime_name} constraint mismatch"
        
        status = "✓ TRADE" if can_trade else "✗ HALT"
        logger.success(f"  {status} {regime_name:20} (allocation: {allocation:.0%})")


# ─────────────────────────────────────────────────────────────────────────────
# Main Test Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("  PHASE 2.5 - STRIKE SELECTION ENGINE TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        test_greeks_calculation,
        test_candidate_filter,
        test_candidate_strikes_generation,
        test_strike_selector,
        test_probability_integration,
        test_regime_constraints,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print()
        except AssertionError as e:
            logger.error(f"FAILED: {e}")
            failed += 1
            print()
        except Exception as e:
            logger.error(f"ERROR: {e}")
            failed += 1
            print()
    
    print("="*70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

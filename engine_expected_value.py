"""
Expected Value Engine
====================

Evaluates every strike and strategy mathematically by computing:

  EV = Premium Received - Expected Loss

For multi-leg strategies (PE/CE/Strangle/IC/Spreads):
  - Per-leg EV calculation
  - Multi-leg portfolio EV
  - Risk-adjusted EV scoring
  - Tail-risk penalty application
  - Final opportunity scoring (ranks by risk-adjusted EV)

Outputs: Strike recommendations ranked by EV, suitable for backtesting and live trading.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from loguru import logger

from engine_probability_distribution import (
    LognormalDistribution,
    MonteCarloDistribution,
    generate_probability_table,
    generate_multi_leg_probabilities,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EV_ANALYSIS_PATH = DATA_DIR / "ev_analysis_latest.json"


# ---------------------------------------------------------------------------
# Single-Leg EV Calculation
# ---------------------------------------------------------------------------

def calculate_single_leg_ev(
    strike: float,
    side: str,
    premium: float,
    spot: float,
    volatility: float,
    dte: float,
    method: str = "mc",
) -> dict:
    """
    Calculate expected value for a single leg (PE or CE).
    
    Args:
        strike: Strike price
        side: 'put' or 'call'
        premium: Premium collected/paid
        spot: Current spot
        volatility: Forecast volatility
        dte: Days to expiry
        method: 'mc' or 'lognormal'
        
    Returns:
        dict with EV, breach prob, expected loss, Sharpe score
    """
    
    # Get distribution
    if method == "mc":
        dist = MonteCarloDistribution(spot, volatility, dte)
    else:
        dist = LognormalDistribution(spot, volatility, dte)
    
    # Calculate breach probability
    breach_prob = dist.breach_probability(strike, side=side)
    
    # Estimate expected loss
    if side.lower() == "put":
        # Loss is max(strike - final_price, 0) * number of contracts
        # For estimation, approximate as: strike * (1 - breach_prob * 0.5)
        if method == "mc":
            payoffs = np.maximum(strike - dist.final_prices, 0)
            expected_loss = np.mean(payoffs)
        else:
            # Approximate for lognormal
            expected_loss = strike * breach_prob * 0.3  # Rough heuristic
    else:
        # Call: loss is max(final_price - strike, 0)
        if method == "mc":
            payoffs = np.maximum(dist.final_prices - strike, 0)
            expected_loss = np.mean(payoffs)
        else:
            expected_loss = strike * (1 - breach_prob) * 0.3
    
    # Expected value
    ev = premium - expected_loss
    
    # Probability of profit (premium fully retained)
    pop = 1 - breach_prob
    
    # Sharpe-like score (reward / risk)
    max_loss = strike if side == "put" else (spot * 5)  # Approximate max loss
    sharpe_score = ev / (max_loss + 0.001) if max_loss > 0 else 0
    
    return {
        "strike": float(strike),
        "side": side,
        "premium": float(premium),
        "breach_prob": float(breach_prob),
        "expected_loss": float(expected_loss),
        "expected_value": float(ev),
        "prob_of_profit": float(pop),
        "sharpe_like_score": float(sharpe_score),
    }


# ---------------------------------------------------------------------------
# Multi-Leg EV Calculation
# ---------------------------------------------------------------------------

def calculate_strangle_ev(
    put_strike: float,
    call_strike: float,
    put_premium: float,
    call_premium: float,
    spot: float,
    volatility: float,
    dte: float,
    method: str = "mc",
) -> dict:
    """
    Calculate EV for short strangle (short put + short call).
    
    Args:
        put_strike: Short put strike
        call_strike: Short call strike
        put_premium: Premium received for put
        call_premium: Premium received for call
        spot, volatility, dte: Market parameters
        method: 'mc' or 'lognormal'
        
    Returns:
        dict with strangle EV and components
    """
    
    # Get individual leg EVs
    put_ev_dict = calculate_single_leg_ev(put_strike, "put", put_premium, spot, volatility, dte, method)
    call_ev_dict = calculate_single_leg_ev(call_strike, "call", call_premium, spot, volatility, dte, method)
    
    # Combined metrics
    total_premium = put_premium + call_premium
    total_expected_loss = put_ev_dict["expected_loss"] + call_ev_dict["expected_loss"]
    total_ev = total_premium - total_expected_loss
    
    # Probability both legs expire worthless
    pe_breach_prob = put_ev_dict["breach_prob"]
    ce_breach_prob = call_ev_dict["breach_prob"]
    prob_both_ok = (1 - pe_breach_prob) * (1 - ce_breach_prob)
    
    # Max loss (if both breached)
    max_loss_put = put_strike - (put_strike * 0.8)  # Heuristic
    max_loss_call = spot * 5 - call_strike  # Heuristic
    max_loss = max_loss_put + max_loss_call
    
    # Risk-adjusted metrics
    sharpe_like = total_ev / (max_loss + 0.001) if max_loss > 0 else 0
    
    return {
        "strategy": "strangle",
        "put_strike": float(put_strike),
        "call_strike": float(call_strike),
        "put_premium": float(put_premium),
        "call_premium": float(call_premium),
        "total_premium": float(total_premium),
        "total_expected_loss": float(total_expected_loss),
        "expected_value": float(total_ev),
        "prob_of_profit": float(prob_both_ok),
        "put_breach_prob": float(pe_breach_prob),
        "call_breach_prob": float(ce_breach_prob),
        "max_loss_estimate": float(max_loss),
        "sharpe_like_score": float(sharpe_like),
        "risk_reward_ratio": float(total_ev / max_loss) if max_loss > 0 else 0,
    }


def calculate_iron_condor_ev(
    put_strike: float,
    put_long_strike: float,
    call_strike: float,
    call_long_strike: float,
    put_premium: float,
    put_long_premium: float,
    call_premium: float,
    call_long_premium: float,
    spot: float,
    volatility: float,
    dte: float,
    method: str = "mc",
) -> dict:
    """
    Calculate EV for iron condor (short call spread + short put spread).
    
    Args:
        put_strike: Short put strike
        put_long_strike: Long put strike (protective)
        call_strike: Short call strike
        call_long_strike: Long call strike (protective)
        [premiums]: Premium for each leg
        spot, volatility, dte: Market parameters
        method: 'mc' or 'lognormal'
        
    Returns:
        dict with iron condor EV
    """
    
    # Net premium
    net_premium = (put_premium + call_premium) - (put_long_premium + call_long_premium)
    
    # Max profit = net premium (if all legs expire worthless)
    max_profit = net_premium
    
    # Max loss = width of spread - net premium
    put_spread_width = put_strike - put_long_strike
    call_spread_width = call_long_strike - call_strike
    max_loss = min(put_spread_width, call_spread_width) - net_premium
    max_loss = max(max_loss, 0)  # Can't be negative
    
    # Simplified: probability neither short strike breached
    if method == "mc":
        dist = MonteCarloDistribution(spot, volatility, dte)
        breaches = np.sum((dist.final_prices < put_strike) | (dist.final_prices > call_strike))
        prob_loss = breaches / len(dist.final_prices)
    else:
        dist = LognormalDistribution(spot, volatility, dte)
        prob_pe_breach = dist.cdf(put_strike)
        prob_ce_breach = 1 - dist.cdf(call_strike)
        prob_loss = prob_pe_breach + prob_ce_breach - (prob_pe_breach * prob_ce_breach)
    
    prob_profit = 1 - prob_loss
    
    # Expected value
    ev = max_profit * prob_profit - max_loss * prob_loss
    
    return {
        "strategy": "iron_condor",
        "put_strike": float(put_strike),
        "put_long_strike": float(put_long_strike),
        "call_strike": float(call_strike),
        "call_long_strike": float(call_long_strike),
        "net_premium": float(net_premium),
        "max_profit": float(max_profit),
        "max_loss": float(max_loss),
        "expected_value": float(ev),
        "prob_of_profit": float(prob_profit),
        "sharpe_like_score": float(ev / (max_loss + 0.001)) if max_loss > 0 else 0,
        "risk_reward_ratio": float(max_profit / max_loss) if max_loss > 0 else 0,
    }


# ---------------------------------------------------------------------------
# EV Scoring & Ranking
# ---------------------------------------------------------------------------

class EVScorer:
    """Score and rank strike opportunities by risk-adjusted EV."""
    
    def __init__(
        self,
        regime_allocation_factor: float = 1.0,
        tail_risk_threshold: float = 0.15,
    ):
        """
        Args:
            regime_allocation_factor: From regime detection (1.0 = full, 0.0 = no trades)
            tail_risk_threshold: Breach prob threshold to apply penalty
        """
        self.regime_factor = regime_allocation_factor
        self.tail_risk_threshold = tail_risk_threshold
    
    def score_single_leg(self, ev_dict: dict) -> dict:
        """
        Score a single-leg EV with risk adjustments.
        
        Returns ev_dict with added fields:
          - ev_score: Risk-adjusted EV
          - tail_risk_penalty: Penalty for tail risk
          - opportunity_score: Final ranking score (0-100)
        """
        
        result = ev_dict.copy()
        
        # Base score: raw EV
        base_ev = ev_dict["expected_value"]
        
        # Tail risk penalty: penalize high breach probability
        breach_prob = ev_dict["breach_prob"]
        if breach_prob > self.tail_risk_threshold:
            tail_penalty = base_ev * (breach_prob - self.tail_risk_threshold) * 0.5
        else:
            tail_penalty = 0
        
        # Risk-adjusted EV
        risk_adj_ev = base_ev - tail_penalty
        
        # Regime adjustment
        regime_adj_ev = risk_adj_ev * self.regime_factor
        
        # Opportunity score (0-100, where 50 is neutral)
        # Normalize EV to [-1, 1] range, then scale to [0, 100]
        opportunity_score = max(0, min(100, 50 + regime_adj_ev * 50))
        
        result["ev_score"] = float(risk_adj_ev)
        result["tail_risk_penalty"] = float(tail_penalty)
        result["regime_factor"] = float(self.regime_factor)
        result["opportunity_score"] = float(opportunity_score)
        
        return result
    
    def score_multi_leg(self, ev_dict: dict) -> dict:
        """Score multi-leg strategy (strangle, condor, etc)."""
        
        result = ev_dict.copy()
        
        # Base score
        base_ev = ev_dict["expected_value"]
        
        # Tail risk: both legs breach
        breach_probs = [
            ev_dict.get("put_breach_prob", 0.1),
            ev_dict.get("call_breach_prob", 0.1),
        ]
        both_breach_prob = np.prod([p for p in breach_probs if p is not None])
        
        if both_breach_prob > self.tail_risk_threshold:
            tail_penalty = base_ev * (both_breach_prob - self.tail_risk_threshold) * 0.7
        else:
            tail_penalty = 0
        
        # Risk-adjusted
        risk_adj_ev = base_ev - tail_penalty
        regime_adj_ev = risk_adj_ev * self.regime_factor
        
        # Opportunity score
        opportunity_score = max(0, min(100, 50 + regime_adj_ev * 30))
        
        result["ev_score"] = float(risk_adj_ev)
        result["tail_risk_penalty"] = float(tail_penalty)
        result["regime_factor"] = float(self.regime_factor)
        result["opportunity_score"] = float(opportunity_score)
        
        return result


# ---------------------------------------------------------------------------
# Main Interface: Generate EV Analysis Report
# ---------------------------------------------------------------------------

def generate_ev_analysis_report(
    spot: float,
    volatility: float,
    dte: float,
    regime_allocation_factor: float = 1.0,
    premium_points: dict | None = None,
) -> dict:
    """
    Generate comprehensive EV analysis for all candidate strikes.
    
    Args:
        spot: Current spot price
        volatility: Forecast volatility
        dte: Days to expiry
        regime_allocation_factor: Risk allocation from regime (0-1)
        premium_points: dict with 'pe', 'ce' premium per strike (defaults applied if None)
        
    Returns:
        dict with analyzed strikes and recommendations
    """
    
    logger.info(f"Generating EV analysis: spot={spot:.2f}, vol={volatility:.2%}, DTE={dte}, regime={regime_allocation_factor:.1%}")
    
    if premium_points is None:
        # Default premium model: base + VIX scaling
        vix = volatility * 100
        vix_factor = max(0.6, min(1.5, vix / 16.0))  # Scale around baseline VIX=16
        base_premium = 80  # points
        premium_points = {
            "pe": int(base_premium * vix_factor),
            "ce": int(base_premium * vix_factor),
        }
    
    # Initialize scorer
    scorer = EVScorer(regime_allocation_factor=regime_allocation_factor)
    
    # Generate candidate strikes (±3% from spot in 0.5% increments)
    candidate_strikes = [
        spot * (1 + pct / 100)
        for pct in np.arange(-3, 3.5, 0.5)
    ]
    
    # Analyze each strike
    strikes_analysis = []
    for strike in candidate_strikes:
        # Single-leg analyses
        pe_ev = calculate_single_leg_ev(
            strike, "put", premium_points["pe"] / 100 * spot,
            spot, volatility, dte, method="lognormal"
        )
        pe_scored = scorer.score_single_leg(pe_ev)
        
        ce_ev = calculate_single_leg_ev(
            strike, "call", premium_points["ce"] / 100 * spot,
            spot, volatility, dte, method="lognormal"
        )
        ce_scored = scorer.score_single_leg(ce_ev)
        
        strikes_analysis.append({
            "strike": float(strike),
            "delta_pct": float((strike / spot - 1) * 100),
            "put": pe_scored,
            "call": ce_scored,
        })
    
    # Find best strikes
    best_pe = max([s["put"] for s in strikes_analysis], key=lambda x: x["opportunity_score"])
    best_ce = max([s["call"] for s in strikes_analysis], key=lambda x: x["opportunity_score"])
    
    # Calculate multi-leg strategies using best strikes
    strangle_ev = calculate_strangle_ev(
        best_pe["strike"],
        best_ce["strike"],
        best_pe["premium"],
        best_ce["premium"],
        spot, volatility, dte,
    )
    strangle_scored = scorer.score_multi_leg(strangle_ev)
    
    return {
        "generated_at": datetime.now().isoformat(),
        "spot": float(spot),
        "volatility": float(volatility),
        "dte": float(dte),
        "regime_allocation_factor": float(regime_allocation_factor),
        "premium_model": premium_points,
        "strikes_analysis": strikes_analysis,
        "best_pe_strike": best_pe,
        "best_ce_strike": best_ce,
        "strangle_recommendation": strangle_scored,
        "top_opportunities": sorted(
            strikes_analysis,
            key=lambda x: max(x["put"]["opportunity_score"], x["call"]["opportunity_score"]),
            reverse=True,
        )[:5],
    }


def save_ev_analysis(report: dict) -> None:
    """Persist EV analysis to JSON."""
    with open(EV_ANALYSIS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"EV analysis saved → {EV_ANALYSIS_PATH}")


def load_ev_analysis() -> dict:
    """Load latest EV analysis."""
    if not EV_ANALYSIS_PATH.exists():
        logger.warning(f"No EV analysis found at {EV_ANALYSIS_PATH}")
        return {}
    
    with open(EV_ANALYSIS_PATH, "r") as f:
        report = json.load(f)
    
    logger.info(f"EV analysis loaded from {EV_ANALYSIS_PATH}")
    return report
